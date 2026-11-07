"""Layer 5: grounded generation + confidence scoring.

Takes Layer 4's RetrievalOutcome and turns it into an Answer: text written
only from the evidence handed over, every claim tied to a citation marker,
and a confidence score whose parts are all visible.

Ways this layer refuses to answer, on purpose:
1. Layer 4 already abstained (succeeded=False) - no LLM call at all.
2. The model reads the evidence and says it doesn't answer the question
   (e.g. "iPhone units sold" when the evidence only has iPhone revenue).
3. The model answers but cites nothing real - an uncited answer breaks the
   one promise this layer makes, so it is not shown as an answer.
4. The model call itself fails - said plainly instead of crashing.

And one way it answers but warns: a number in the answer that matches
nothing in the cited evidence forces the confidence to LOW.
"""

from __future__ import annotations

import itertools
import json
import re
import time

from .config import Config, get_config
from .models import (
    Answer,
    Citation,
    ConfidenceBreakdown,
    GradedChunk,
    RetrievalOutcome,
)
from .self_healing import ProgressFn, emit, get_groq_client, search_parts, self_heal
from .structured import SEGMENT_LABELS, describe_fact, segments_mentioned

# --- Step 1: lay out the evidence with markers -----------------------------


def _clean_facts(outcome: RetrievalOutcome) -> list[dict]:
    """Only structured facts that resolved to one real value. Flagged ones
    (ambiguous, no year given, year not in the data) carry no number, so
    there is nothing in them for the model to cite.
    """
    return [f for f in outcome.structured_facts if "value" in f]


def _build_evidence(
    chunks: list[GradedChunk], facts: list[dict]
) -> tuple[str, dict[str, GradedChunk | dict]]:
    """Number every piece of evidence. Narrative chunks get [1], [2], ...;
    structured facts get [S1], [S2], ... Returns the text block for the
    prompt plus a lookup from marker to the evidence it stands for, which is
    how citations get checked after the model answers.
    """
    lines: list[str] = []
    lookup: dict[str, GradedChunk | dict] = {}

    for i, g in enumerate(chunks, start=1):
        marker = f"[{i}]"
        lookup[marker] = g
        c = g.retrieved.chunk
        lines.append(f"{marker} ({c.item} / {c.subsection or c.section})\n{c.text}")

    for i, fact in enumerate(facts, start=1):
        marker = f"[S{i}]"
        lookup[marker] = fact
        # Same wording Layer 4's grader saw, from one shared function.
        lines.append(f"{marker} (financial statement data)\n{describe_fact(fact)}")

    return "\n\n".join(lines), lookup


# --- Step 2: ask the model, allowing it to say "this doesn't answer it" ----

_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        # Asking for markers only inside the prose wasn't enough: on a
        # one-number answer the model skipped them entirely. A required
        # field in the response format can't be quietly left out.
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answerable", "answer", "citations"],
    "additionalProperties": False,
}

_ANSWER_PROMPT = """You answer questions about a company's 10-K filing using \
ONLY the evidence below. Do not use any outside knowledge.

Question: {question}

Evidence:
{evidence}

Rules:
- Every factual claim must end with the marker of the evidence it comes \
from, like [1] or [S1]. Use only markers that appear above. This applies \
even to a one-number answer, e.g. "$73,733 million [S1]".
- Also list every marker you used in "citations", e.g. ["[S1]"].
- If financial statement data [S#] is provided, use it for every number in \
your answer, and show the numbers a comparison or trend rests on.
- Check that the evidence answers what was actually asked. A number about \
the right topic is not enough if it measures something else (for example, \
revenue in dollars does not answer a question about units sold).
- If the evidence does not answer the question, set "answerable" to false \
and use "answer" to say briefly what is missing.
- Be concise. State numbers exactly as given, always with their unit. \
Dollar amounts in a 10-K's financial tables are in millions unless the table \
says otherwise, even where a passage doesn't repeat "(in millions)": write a \
table's 12,345 as "$12,345 million", never "$12,345". Per-share amounts are \
in dollars ("$7.46 per share"). Share counts are not dollars: give them in the \
unit their table states."""


def _ask_model(question: str, evidence: str, cfg: Config) -> dict:
    client = get_groq_client(cfg)
    last_exc = None
    # Groq sometimes rejects the call itself with 400 "json_validate_failed"
    # (the model wrote JSON that does not match the schema). A retry at the same
    # temperature tends to repeat the mistake, so the second try changes it.
    # After two failures the error goes up and the caller says so plainly.
    for temperature in (cfg.generation.temperature, 0.5):
        try:
            resp = client.chat.completions.create(
                model=cfg.generation.model,
                messages=[
                    {
                        "role": "user",
                        "content": _ANSWER_PROMPT.format(question=question, evidence=evidence),
                    }
                ],
                max_tokens=cfg.generation.max_tokens,
                temperature=temperature,
                reasoning_effort=cfg.generation.effort,
                response_format={
                    "type": "json_schema",
                    # strict: Groq guarantees the reply matches the schema.
                    "json_schema": {"name": "grounded_answer", "schema": _ANSWER_SCHEMA, "strict": True},
                },
            )
            break
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is this one error
            if "json_validate_failed" not in str(exc):
                raise
            last_exc = exc
    else:
        raise last_exc
    return json.loads(resp.choices[0].message.content)

# --- Step 3: check which citations are real ---------------------------------

_MARKER_RE = re.compile(r"\[S?\d+\]")
# A new sentence starts after . ! or ? and a space.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _check_citations(
    text: str, lookup: dict[str, GradedChunk | dict], cfg: Config
) -> tuple[list[Citation], list[str], list[str]]:
    """Return (real citations, markers that point to nothing, notes).

    The model is told to use only the markers it was given, but nothing
    forces it to. A marker that points to no evidence is a made-up source -
    exactly what this layer exists to prevent - so it is dropped and noted,
    never turned into a citation.
    """
    used = list(dict.fromkeys(_MARKER_RE.findall(text)))  # unique, in order
    citations: list[Citation] = []
    invented: list[str] = []
    notes: list[str] = []

    for marker in used:
        ev = lookup.get(marker)
        if ev is None:
            invented.append(marker)
            continue
        if isinstance(ev, dict):  # structured fact
            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=f"structured:{ev['table']}:{ev['label']}:{ev['year']}",
                    source=(
                        f"{cfg.company.ticker} | {cfg.company.form} | FY{ev['year']} | "
                        f"{ev['table'].replace('_', ' ')} - {ev['label']}"
                    ),
                    quote=describe_fact(ev),
                )
            )
        else:  # narrative chunk
            c = ev.retrieved.chunk
            citations.append(
                Citation(marker=marker, chunk_id=c.chunk_id, source=c.citation())
            )

    if invented:
        notes.append(f"model cited evidence that doesn't exist: {', '.join(invented)} (removed)")
    return citations, invented, notes


# --- Step 4: check the numbers ----------------------------------------------

# A number, optionally with $ in front and thousand/million/billion/% after
# ("M" and "B" too: "$112,887M"; share counts are tabled "in thousands").
# Not preceded by a letter or digit, so "FY2025", "Q2" and "[S1]" aren't
# read as numbers.
_NUM_RE = re.compile(
    r"(?<![A-Za-z\d.,])(\$?)(\d[\d,]*(?:\.\d+)?)\s*(thousand|million|billion|bn\b|[mb]\b|%)?", re.IGNORECASE
)
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "m": 1e6, "b": 1e9, "bn": 1e9}


def _numbers(text: str) -> list[tuple[float, float, str]]:
    """Money-like numbers in a text, as (value, rounding tolerance, as written).

    The tolerance is half a unit of the last digit written: "$6.0 billion"
    may stand for anything from 5.95 to 6.05 billion, so rounding isn't
    flagged, but "$11.2 billion" for 112,010 million still is.
    Skipped: bare numbers like years (2025) and counts (3-year), which
    aren't financial figures.
    """
    out = []
    for dollar, digits, unit in _NUM_RE.findall(_MARKER_RE.sub(" ", text)):
        digits = digits.rstrip(",.")
        unit = unit.lower()
        if not (dollar or unit or "," in digits):
            continue
        value = float(digits.replace(",", ""))
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        scale = _SCALE.get(unit, 1)
        written = f"{dollar}{digits}{unit if unit == '%' else (' ' + unit if unit else '')}"
        out.append((value * scale, 0.5 * 10 ** -decimals * scale, written))
    return out


def _with_pairs(values: list[float]) -> list[float]:
    """The values plus the difference and sum of any two - "a decrease of
    $1,319 million" is $37,005 minus $35,686 million from the same passage."""
    pairs = list(itertools.combinations(values, 2))
    return values + [abs(a - b) for a, b in pairs] + [a + b for a, b in pairs]


def _allowed_values(cited: list[GradedChunk | dict]) -> tuple[list[float], list[float]]:
    """Every value an answer may state, given what it cites, as (in dollars,
    as table cells). A "$" amount with no unit may only match the first.

    Facts: the value itself, plus sums, differences and % changes of up to
    three cited facts - the model is allowed to calculate ("liabilities fell
    by $22,522,000,000" is FY2024 minus FY2025, and appears in no source).
    Passages: every number in their text, read as millions and billions,
    because 10-K tables are stated "in millions".
    """
    fact_values = [f["value"] for f in cited if isinstance(f, dict)]
    # Outflows are stored negative ("cash used in financing": -120,686) but
    # written positive in an answer, so the size of each value counts too.
    dollars = list(fact_values) + [abs(v) for v in fact_values]
    for r in range(2, min(3, len(fact_values)) + 1):
        for combo in itertools.combinations(fact_values, r):
            for signs in itertools.product((1, -1), repeat=r):
                dollars.append(abs(sum(s * v for s, v in zip(signs, combo))))
    # The same figures restated in millions or billions without the unit, as
    # 10-K tables do: "(285,508 + 73,733 = 359,241)" checks the balance sheet
    # with Total liabilities 285,508,000,000 written in millions.
    cells = [v / 1e6 for v in dollars] + [v / 1e9 for v in dollars]
    for a, b in itertools.permutations(fact_values, 2):
        if a:
            cells.append(abs(b - a) / abs(a) * 100)  # % change, either direction

    # Passages: every number in their text, including bare table cells
    # ("112,010"). Only amounts with cents ("7.46" per share) are also dollars
    # as written: table cells in millions are whole numbers, and "$10.2" for
    # "$10.2 billion" has lost its unit just the same.
    numbers, cents = [], []
    for g in cited:
        if isinstance(g, dict):
            continue
        for line in g.retrieved.chunk.text.split("\n"):
            row = []
            for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)(\s*(?:million|billion))?", line, re.IGNORECASE):
                digits = m.group(1).rstrip(",.")
                row.append(float(digits.replace(",", "")))
                if "." in digits and not m.group(2):
                    cents.append(row[-1])
            numbers += row
            # % change between two numbers in the same row: "diluted EPS rose
            # 22.7%" is $6.08 to $7.46 on one line of the EPS table. Only within
            # a row - across a whole passage, almost any percentage would match.
            for a, b in itertools.permutations(row, 2):
                if a:
                    cells.append(abs(b - a) / abs(a) * 100)
    for v in _with_pairs(numbers):
        cells.append(v)
        # Share counts are tabled "in thousands", amounts "in millions".
        dollars += [v * 1e3, v * 1e6, v * 1e9]
    dollars += _with_pairs(cents)
    return dollars, cells


def _unverified_numbers(text: str, cited: list[GradedChunk | dict]) -> list[str]:
    """Numbers in the answer that match nothing the answer cites.

    A real citation next to a wrong number is the worst failure for this
    tool: "$11.2 billion [S1]" where S1 is 112,010 million would otherwise
    pass, because the citation itself is real. So would "$64,377" for a
    table's 64,377 million: the digits match, but it reads as $64 thousand.
    """
    dollars, cells = _allowed_values(cited)
    bad = []
    for value, tol, written in _numbers(text):
        unitless_dollars = written.startswith("$") and written[-1].isdigit()
        allowed = dollars if unitless_dollars else dollars + cells
        if not any(abs(value - a) <= max(tol, 0.001 * abs(a)) for a in allowed):
            bad.append(written)
    return list(dict.fromkeys(bad))


# --- Step 5: confidence -----------------------------------------------------

# Starting score for an answer that rests on database facts; the checks in
# _confidence() can then lower it. Starting points, not measured truths.
_FACT_SCORE = {"keyword": 0.90, "fuzzy": 0.60}
_INCOMPLETE_PENALTY = 0.30


def _loop_cleanliness(outcome: RetrievalOutcome, cfg: Config) -> float:
    """1.0 if Layer 4 succeeded on the first try, falling to 0.0 if it
    needed every attempt. More rewrites = the original question matched the
    filing less directly."""
    max_attempts = cfg.self_healing.max_attempts
    if max_attempts <= 1:
        return 1.0
    return 1.0 - (len(outcome.attempts) - 1) / (max_attempts - 1)


def _search_signals(chunks: list[GradedChunk]) -> tuple[float, float]:
    """(share found by both searches, strongest reranker score) for these passages."""
    # Signal 1: share of the passages that BOTH dense and keyword search found.
    hybrid = sum(1 for g in chunks if g.retrieved.found_by_both) / len(chunks) if chunks else 0.0
    # Signal 2: strongest cross-encoder score among them. Measured on the eval
    # set: answerable questions topped out at 0.41-0.999, unanswerable ones at
    # 0.0005-0.022, so the raw 0-1 score is used as-is, no rescaling.
    rerank = max((g.retrieved.rerank_score or 0.0 for g in chunks), default=0.0)
    return hybrid, rerank


def _repeats_a_figure(g: GradedChunk, facts: list[dict]) -> bool:
    """Does this passage state one of the cited figures - the same number,
    or the same number in millions or billions as 10-K tables write it
    ("112,010" for $112,010,000,000)? Then it backs that figure up instead
    of adding anything of its own.
    """
    # Thousands too: share counts are tabled "in thousands".
    targets = [abs(f["value"]) / scale for f in facts if f["value"] for scale in (1, 1e3, 1e6, 1e9)]
    for d in re.findall(r"\d[\d,]*(?:\.\d+)?", g.retrieved.chunk.text):
        d = d.rstrip(",.")
        n = float(d.replace(",", ""))
        # Half a unit of the last digit written, as in _numbers().
        tol = 0.5 * 10 ** -(len(d.split(".")[1]) if "." in d else 0)
        if any(abs(n - t) <= tol for t in targets):
            return True
    return False


def _band(score: float, cfg: Config) -> str:
    b = cfg.confidence.bands
    if score >= b.high:
        return "high"
    if score >= b.medium:
        return "medium"
    return "low"


def _confidence(
    outcome: RetrievalOutcome,
    answer_text: str,
    citations: list[Citation],
    lookup: dict[str, GradedChunk | dict],
    cfg: Config,
) -> ConfidenceBreakdown:
    """Score only the evidence the answer actually cites, not everything
    retrieved - a strong chunk the answer never used shouldn't raise
    confidence in that answer. Then apply checks that can pull it down.
    """
    cited = [lookup[c.marker] for c in citations]
    cited_chunks = [e for e in cited if not isinstance(e, dict)]
    cited_facts = [e for e in cited if isinstance(e, dict)]

    hybrid, rerank = _search_signals(cited_chunks)
    # Signal 3: did Layer 4 get there on the first try?
    loop = _loop_cleanliness(outcome, cfg)

    w = cfg.confidence.weights
    bands = cfg.confidence.bands
    notes: list[str] = []

    if cited_facts:
        # The formula above only describes narrative search, so an answer
        # resting on database values starts from its own score - but no
        # longer keeps it automatically: the checks below can lower it.
        fuzzy = any(f.get("_matched_via") == "fuzzy" for f in cited_facts)
        score = _FACT_SCORE["fuzzy" if fuzzy else "keyword"]
        notes.append(
            "answer rests on exact values from the financial statements"
            + (" (matched by fuzzy wording - check the label)" if fuzzy else "")
        )
        # The figure is exact, but is it the one the question asked for? Layer
        # 4's grade says; below the "strong" line the score drops with it, so
        # a figure graded 0.5 lands in MEDIUM.
        strong = cfg.self_healing.strong_chunk_threshold
        weakest = min(f.get("_relevance", 1.0) for f in cited_facts)
        if weakest < strong:
            score *= weakest / strong
            notes.append(f"weakest cited figure was graded {weakest:.2f}")
        # A cited passage that only restates a figure adds nothing to doubt.
        # One that adds something else (a reason, a comparison) carries part
        # of the answer, so its search confidence counts for its share.
        own = [g for g in cited_chunks if not _repeats_a_figure(g, cited_facts)]
        if own:
            own_hybrid, own_rerank = _search_signals(own)
            passage_score = (
                w.hybrid_agreement * own_hybrid + w.rerank_strength * own_rerank + w.loop_cleanliness * loop
            )
            score = (score * len(cited_facts) + passage_score * len(own)) / (len(cited_facts) + len(own))
            notes.append(
                f"{len(own)} cited passage(s) carry part of the answer; "
                f"their search confidence ({passage_score:.2f}) is blended in"
            )
        # Was every requested line item x year found? A flagged fact (year
        # not in the data, ambiguous label) means part of the question wasn't.
        missing = [f for f in outcome.structured_facts if "value" not in f]
        if missing:
            score -= _INCOMPLETE_PENALTY
            notes.append(f"{len(missing)} requested figure(s) not found in the financial statements")
    else:
        score = w.hybrid_agreement * hybrid + w.rerank_strength * rerank + w.loop_cleanliness * loop

    # Facts were handed over, but the answer cites none of them: it took its
    # numbers from prose instead, or gave none. Capped at medium.
    if _clean_facts(outcome) and not cited_facts:
        score = min(score, bands.high - 0.01)
        notes.append("answer ignored the database figures it was given")

    # The question names a product or region, but the answer rests on
    # company-wide figures alone: right if the total is what was wanted
    # ("tariffs in Europe ... gross margin"), wrong if it stands in for the
    # product's own ("how much gross margin did the iPhone make"). Wording
    # can't tell which, so say so instead of calling it HIGH.
    named = segments_mentioned(outcome.original_query)
    company_wide = [f["label"] for f in cited_facts if f["label"] not in SEGMENT_LABELS]
    if named and company_wide and len(company_wide) == len(cited_facts):
        score = min(score, bands.high - 0.01)
        notes.append(
            f"question is about {', '.join(named)}, but {', '.join(dict.fromkeys(company_wide))} "
            "is Apple's company-wide total - check it answers what was asked"
        )

    # Every number must match something the answer cites. If one doesn't,
    # the answer is still shown, but marked LOW - never quietly trusted.
    bad = _unverified_numbers(answer_text, cited)
    if bad:
        score = min(score, bands.medium - 0.01)
        notes.append(f"number(s) not found in cited sources: {', '.join(bad)}")

    if len(outcome.attempts) > 1:
        notes.append(f"needed {len(outcome.attempts)} retrieval attempts")

    score = max(score, 0.0)
    return ConfidenceBreakdown(
        score=round(score, 3),
        band=_band(score, cfg),
        hybrid_agreement=round(hybrid, 3),
        rerank_strength=round(rerank, 3),
        loop_cleanliness=round(loop, 3),
        notes=notes,
    )


# --- Entry point -------------------------------------------------------


def _model_failed(query: str, exc: Exception, outcome: RetrievalOutcome | None = None) -> Answer:
    """A rate limit, an outage or an unreadable reply is said plainly
    instead of crashing the command. The first line of the error is kept: the
    type alone ("OSError") doesn't say whether it was a rate limit or a model
    that wouldn't load."""
    detail = next(iter(str(exc).strip().splitlines()), "")[:200]
    return Answer(
        query=query,
        answered=False,
        text=f"Model call failed ({type(exc).__name__}{': ' + detail if detail else ''}). Try again in a minute.",
        outcome=outcome,
    )


def generate_answer(
    outcome: RetrievalOutcome, cfg: Config | None = None, on_progress: ProgressFn | None = None
) -> Answer:
    """Layer 5's core: RetrievalOutcome in, Answer out."""
    conf = cfg or get_config()
    query = outcome.original_query

    # Refusal 1: Layer 4 already decided the evidence is too weak.
    if not outcome.succeeded:
        return Answer(query=query, answered=False, text=outcome.abstain_reason, outcome=outcome)

    chunks = outcome.relevant_chunks
    facts = _clean_facts(outcome)
    evidence, lookup = _build_evidence(chunks, facts)
    emit(on_progress, "write", passages=len(chunks), facts=len(facts))
    try:
        result = _ask_model(query, evidence, conf)
    except Exception as exc:  # noqa: BLE001 - rate limit, outage, unreadable reply
        return _model_failed(query, exc, outcome)

    # Refusal 2: the evidence is on-topic but doesn't answer what was asked.
    if not result["answerable"]:
        return Answer(query=query, answered=False, text=result["answer"], outcome=outcome, refused_by_model=True)

    # Check markers from both places: inline in the text, and the required
    # "citations" list. Wrap bare "S1" as "[S1]" in case the model drops the
    # brackets in the list.
    listed = [m if m.startswith("[") else f"[{m}]" for m in (s.strip() for s in result.get("citations", []))]
    citations, invented, notes = _check_citations(result["answer"] + " " + " ".join(listed), lookup, conf)

    # An invented marker is dropped from the sources AND from the text, so
    # the reader never sees a citation that leads nowhere.
    text = result["answer"]
    for m in invented:
        text = text.replace(m, "")
    text = re.sub(r" +([.,;])", r"\1", text).strip()

    # Refusal 3: an answer with no real citations is an unsupported claim.
    if not citations:
        return Answer(
            query=query,
            answered=False,
            text="The model produced an answer without citing any real evidence, so it was not shown.",
            outcome=outcome,
        )

    # The citations list says which sources were used; the text must also show
    # where. A one-sentence answer can only come from its cited sources, so the
    # markers go on its end. A longer one could mix them up - leave it as it
    # is, but don't call it HIGH.
    untied = not _MARKER_RE.search(text)
    if untied and len(_SENTENCE_RE.split(text)) == 1:
        markers = "".join(c.marker for c in citations)
        text = f"{text[:-1]} {markers}." if text.endswith(".") else f"{text} {markers}"
        untied = False

    confidence = _confidence(outcome, text, citations, lookup, conf)
    confidence.notes = notes + confidence.notes
    if untied:
        confidence.score = min(confidence.score, conf.confidence.bands.high - 0.01)
        confidence.band = _band(confidence.score, conf)
        confidence.notes.append("sources are not tied to specific sentences - check them before relying on it")
    return Answer(
        query=query,
        answered=True,
        text=text,
        citations=citations,
        confidence=confidence,
        outcome=outcome,
    )


def answer(query: str, cfg: Config | None = None, on_progress: ProgressFn | None = None) -> Answer:
    """Full pipeline in one call: Layers 3 + 4 (self_heal) then Layer 5.

    on_progress, if given, is told about each step as it happens - used by
    Layer 7's live room. Leaving it out changes nothing.
    """
    conf = cfg or get_config()
    start = time.perf_counter()
    try:
        outcome = self_heal(query, conf, on_progress)
    except Exception as exc:  # noqa: BLE001 - Layer 4's grader and rewriter call Groq too
        result = _model_failed(query, exc)
    else:
        result = generate_answer(outcome, conf, on_progress)
        # No answer from the question's own words - whether Layer 4 refused the
        # evidence or the answer model found it doesn't answer the question:
        # one more search, with the question split into parts. Both happen to
        # eval Q20 on different runs, depending on how the grader scores the
        # net income lines. A failed model call has nothing to gain from it.
        if result.refused_by_model or not outcome.succeeded:
            try:
                retried = search_parts(outcome, conf, on_progress)
            except Exception:  # noqa: BLE001 - the extra search is an extra try; keep the refusal
                retried = None
            if retried is not None and retried.succeeded:
                result = generate_answer(retried, conf, on_progress)
                if result.confidence is not None:
                    result.confidence.notes.append(
                        "found on a second search: the first evidence didn't answer the question, so it was "
                        f"split into parts ({retried.final_query})"
                    )
    result.elapsed_s = round(time.perf_counter() - start, 2)
    return result