"""Layer 4: self-healing retrieval loop.

Layer 3 retrieves chunks unconditionally - it has no idea whether what it
found is actually good. This layer is the checkpoint: grade what came back,
and if it's not good enough, reword the question and try again. If it's
still not good enough after a few tries, stop and say so honestly instead of
generating an answer from weak evidence. That honesty is the entire point -
most RAG systems skip this and hallucinate confidently on bad retrieval.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from groq import Groq

from .config import Config, get_config, get_secrets
from .models import AttemptRecord, Chunk, GradedChunk, RetrievalOutcome, RetrievedChunk
from .retrieval import hybrid_retrieve
from .structured import MAX_FACTS, describe_fact, get_structured_facts

_CLIENT: Groq | None = None


def get_groq_client(cfg: Config) -> Groq:
    """Singleton, same reasoning as the Qdrant client in indexing.py -
    no need to re-authenticate a new client on every call in a loop that may
    call the API several times per question.
    """
    global _CLIENT
    if _CLIENT is None:
        secrets = get_secrets()
        _CLIENT = Groq(
            api_key=secrets.groq_api_key,
            max_retries=cfg.generation.max_retries,
        )
    return _CLIENT


# --- Step 1: grade every retrieved chunk, in ONE call ----------------------

_GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "grades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "relevance": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["item", "relevance", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["grades"],
    "additionalProperties": False,
}

_GRADE_PROMPT = """You are grading whether evidence from a company's 10-K \
filing actually answers a question - not whether it's topically related, \
but whether it contains the specific information asked for.

Question: {question}
{parts}
Evidence:
{passages}

After its marker, each passage shows where it sits in the filing, in \
parentheses. Read the \
WHOLE passage, not just its first sentence: a passage can open with the end \
of an unrelated paragraph and still answer the question further on.

Items marked [S1], [S2]... are figures from the financial statements; the \
others are passages. Score each item on its own.

Passages, on this scale:
1.0 - states exactly what was asked: the right company, measure and period.
{strong} - contains the answer, but one small step is needed (e.g. picking \
the right year column of a table, or the answer is not the passage's main \
point).
{relevant} - has part of what is needed: one of several things asked for, \
or the right measure for a different period.
{weak} - same area, but none of the information asked for.
0.0 - unrelated.
Use a value in between when a passage falls between two steps.

Figures, on this scale instead:
1.0 - exactly a measure and period the question names. A change, trend or \
comparison needs several figures, one per measure and year: each of them is \
complete for its part and scores 1.0, even though it does not answer the \
question alone.
{relevant} - the right measure for a different period.
{weak} - the right topic but the wrong measure: revenue in dollars is not \
"units sold", and a company-wide total is not a figure for one product or \
segment.
0.0 - unrelated.

For EACH item give its marker exactly as shown (e.g. "[0]" or "[S1]"), its \
score and a one-sentence reason. Return exactly {n} grades, one per item."""


def _where(c: Chunk) -> str:
    """Where a passage sits in the filing, e.g. "Item 8 / CONSOLIDATED
    BALANCE SHEETS" - a table of bare numbers can't be judged without it.
    The reranker (retrieval.py) is given the same context."""
    return f"{c.item} / {c.subsection or c.section}"


def _clamp(x: object) -> float:
    """Scores must be 0-1 for GradedChunk; a model writing 1.2 or "0.8"
    shouldn't crash the question."""
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def _marker(item: object) -> str:
    """The marker a grade names: "[0]" -> "0", "[S1]" / "s1" -> "S1".
    Anything unrecognisable -> "" (matches no item)."""
    m = re.search(r"\b(S?\d+)\b", str(item or "").upper())
    return m.group(1) if m else ""


# Shown to the grader on the second search (search_parts), whose evidence came
# from the question's parts. Without it, a passage holding one part - the tax
# reconciliation showing a $10,246 million State Aid charge in FY2024 - was
# graded 0.0 against "what caused the FY2024 net income dip?" as "tax
# reconciliation; no cause of dip", and the question was still refused.
_PARTS_NOTE = """
To find the evidence, the question was split into these parts, each searched \
on its own:
{parts}
An item that answers one of these parts is evidence for the question, even \
though it does not state the whole answer: score it as if that part were the \
question.
"""


def grade_chunks(
    query: str, chunks: list[RetrievedChunk], cfg: Config, facts: list[dict] | None = None,
    parts: list[str] | None = None,
) -> tuple[list[GradedChunk], list[float]]:
    """Grade every chunk - and any database facts - in ONE call.

    Config calls this out directly: Groq's free tier caps at 8K tokens/minute,
    and N separate calls resend the full instructions N times - at this scale
    that overhead is most of the budget, not the actual chunk content being
    graded. One call, one JSON array back, is what `batch_grading: true` means.

    Facts are graded with the passages so a number has to earn its place
    like any passage: "iPhone revenue" is a real fact but no answer to "how
    many iPhone units". Returns (graded chunks, one score per fact).
    """
    facts = facts or []
    n = len(chunks) + len(facts)
    if n == 0:
        return [], []

    items = [f"[{i}] ({_where(c.chunk)}) {c.chunk.text}" for i, c in enumerate(chunks)]
    items += [f"[S{i}] {describe_fact(f)}" for i, f in enumerate(facts, start=1)]
    # The scale's steps are the thresholds self_heal() decides with, so a
    # score means the same thing to the grader as it does to the code.
    sh = cfg.self_healing
    prompt = _GRADE_PROMPT.format(
        question=query, passages="\n\n".join(items), n=n,
        parts=_PARTS_NOTE.format(parts="\n".join(f"- {p}" for p in parts)) if parts else "",
        strong=f"{sh.strong_chunk_threshold:g}",
        relevant=f"{sh.relevance_threshold:g}",
        weak=f"{sh.give_up_below:g}",
    )

    def _ask(temperature: float):
        return get_groq_client(cfg).chat.completions.create(
            model=cfg.generation.utility_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=cfg.generation.utility_max_tokens,
            temperature=temperature,
            # Same fix as rewrite_query: without it the model can spend its whole
            # token budget thinking and return nothing.
            reasoning_effort=cfg.generation.utility_effort,
            response_format={
                "type": "json_schema",
                # strict: Groq guarantees the reply matches the schema. Without
                # it, the reply can be invalid JSON.
                "json_schema": {"name": "chunk_grades", "schema": _GRADE_SCHEMA, "strict": True},
            },
        )

    # Groq sometimes rejects the call itself with 400 "json_validate_failed"
    # (the model wrote broken JSON). At temperature 0 it repeats the same
    # mistake, so the retry has to change something: one more try at 0.5.
    # Any other error is real and still raises.
    resp = None
    for temperature in (0.0, 0.5):
        try:
            resp = _ask(temperature)
            break
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is this one error
            if "json_validate_failed" not in str(exc):
                raise

    markers = [f"{i}" for i in range(len(chunks))] + [f"S{i}" for i in range(1, len(facts) + 1)]
    try:
        # resp is None when both tries failed: json.loads("") then fails and
        # every item is failed honestly below.
        reply = json.loads((resp.choices[0].message.content if resp else "") or "")["grades"]
        # Match each grade to its item by the marker the model wrote, never by
        # position: the model can list correct grades in the wrong order, and
        # matching by position would then hand them to the wrong passages.
        by_marker: dict[str, dict] = {}
        for g in reply:
            m = _marker(g.get("item"))
            if m in markers and m not in by_marker:
                by_marker[m] = {"relevance": _clamp(g.get("relevance")), "reason": str(g.get("reason", ""))}
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        # Unreadable reply: fail every item rather than crash. The question
        # is then refused honestly ("0 relevant") instead of erroring out.
        by_marker = {m: {"relevance": 0.0, "reason": "grader failed"} for m in markers}

    # An item the model skipped fails rather than borrowing a guess - a wrong
    # guess here silently corrupts the scores. If it graded one twice, the
    # first grade counts.
    grades = [by_marker.get(m, {"relevance": 0.0, "reason": "ungraded"}) for m in markers]

    graded = [
        GradedChunk(retrieved=c, relevance=g["relevance"], reason=g["reason"])
        for c, g in zip(chunks, grades)
    ]
    return graded, [g["relevance"] for g in grades[len(chunks):]]


# --- Step 2: rewrite the query when grading comes back weak ----------------

_REWRITE_PROMPT = """A search for this question returned mostly irrelevant \
results from a company's 10-K filing.

Original question: {original}
Query just tried: {tried}

Why the results were weak:
{reasons}

Write ONE alternative search query - different wording, different angle, or \
more specific terms - likely to retrieve the actually relevant passage. \
Reply with ONLY the new query text, nothing else."""


def rewrite_query(
    original_query: str, tried_query: str, graded: list[GradedChunk], cfg: Config
) -> str:
    """Ask the model for a differently-worded query, using the weak results'
    own stated reasons as the signal for what to change.
    """
    client = get_groq_client(cfg)
    reasons = "\n".join(f"- {g.reason}" for g in graded if not g.reason == "")
    prompt = _REWRITE_PROMPT.format(
        original=original_query, tried=tried_query, reasons=reasons or "(no detail given)"
    )

    resp = client.chat.completions.create(
        model=cfg.generation.utility_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.0,
        # gpt-oss spends tokens "thinking" before it answers. Without a low
        # effort setting it can use up the whole budget thinking and return
        # an empty reply - which then gets searched and graded as if it were
        # a real question.
        reasoning_effort=cfg.generation.utility_effort,
    )
    new_query = (resp.choices[0].message.content or "").strip()
    # Never hand back an empty query. Falling back to the last query means
    # the next attempt returns the same chunks, and the stagnation check
    # ends the loop honestly instead of grading against nothing.
    return new_query or tried_query


# --- Step 3: the loop itself ------------------------------------------------


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# Called as on_progress(stage, details) at each step, so a live view (Layer 7)
# can show what the pipeline is doing. Optional everywhere: None = silent.
ProgressFn = Callable[[str, dict[str, Any]], None]


def emit(on_progress: ProgressFn | None, stage: str, **details: Any) -> None:
    """Report one step. A broken listener must never break the answer, so
    any error it raises is swallowed here."""
    if on_progress is None:
        return
    try:
        on_progress(stage, details)
    except Exception:  # noqa: BLE001 - progress reporting is best-effort
        pass


def self_heal(
    query: str, cfg: Config | None = None, on_progress: ProgressFn | None = None
) -> RetrievalOutcome:
    """Retrieve, grade, and retry with a reworded query up to `max_attempts`
    times. Stops early on success, on stagnation (a retry that returns
    essentially the same chunks isn't making progress), or after the last
    attempt - and abstains honestly in the last two cases rather than
    handing Layer 5 evidence too weak to answer from.
    """
    conf = cfg or get_config()
    sh = conf.self_healing

    # Structured lookup is cheap (no LLM call, no embeddings) and based on
    # the ORIGINAL question only - rewording is a narrative-search strategy,
    # it has nothing to do with which DuckDB row the question asked about.
    # Run it once, up front, regardless of how the narrative loop below goes.
    structured_facts = get_structured_facts(query, conf)

    # Facts with a real value get graded like passages (on attempt 1, against
    # the original question). Flagged ones (ambiguous, no year...) have no
    # number to grade; they pass through untouched for Layer 5 to see.
    clean_facts = [f for f in structured_facts if "value" in f]
    flagged_facts = [f for f in structured_facts if "value" not in f]
    relevant_facts: list[dict] = []
    fact_scores: list[float] = []
    emit(on_progress, "lookup", facts=[f["label"] for f in clean_facts])

    attempts: list[AttemptRecord] = []
    current_query = query
    prev_chunk_ids: set[str] | None = None
    graded: list[GradedChunk] = []
    best: tuple[int, float, list[GradedChunk], str] | None = None  # (n_relevant, mean, chunks, query)

    for attempt_num in range(1, sh.max_attempts + 1):
        emit(on_progress, "search", attempt=attempt_num, query=current_query)
        retrieved = hybrid_retrieve(current_query, conf)
        emit(on_progress, "grade", attempt=attempt_num, passages=len(retrieved),
             facts=len(clean_facts) if attempt_num == 1 else 0)
        if attempt_num == 1:
            graded, fact_scores = grade_chunks(current_query, retrieved, conf, clean_facts)
            # A fact that doesn't answer the question isn't passed on: iPhone
            # revenue graded low for "iPhone units" never reaches Layer 5.
            relevant_facts = [
                {**f, "_relevance": s}
                for f, s in zip(clean_facts, fact_scores)
                if s >= sh.relevance_threshold
            ]
        else:
            graded, _ = grade_chunks(current_query, retrieved, conf)

        chunk_ids = {g.retrieved.chunk.chunk_id for g in graded}
        n_relevant_chunks = sum(1 for g in graded if g.relevance >= sh.relevance_threshold)
        # Relevant facts count as evidence just like relevant passages.
        n_relevant = n_relevant_chunks + len(relevant_facts)
        mean_relevance = sum(g.relevance for g in graded) / len(graded) if graded else 0.0

        stagnated = (
            prev_chunk_ids is not None
            and _jaccard(chunk_ids, prev_chunk_ids) >= sh.stagnation_jaccard
        )

        attempts.append(
            AttemptRecord(
                attempt=attempt_num,
                query=current_query,
                strategy="original" if attempt_num == 1 else "rewrite",
                n_retrieved=len(retrieved),
                n_relevant=n_relevant_chunks,
                mean_relevance=mean_relevance,
                chunk_ids=sorted(chunk_ids),
                stagnated=stagnated,
                note=(
                    f"{len(relevant_facts)} of {len(clean_facts)} database facts relevant"
                    if clean_facts else ""
                ),
            )
        )
        emit(
            on_progress, "graded", attempt=attempt_num, relevant=n_relevant_chunks, total=len(graded),
            # Facts are only graded on attempt 1; report them there.
            facts_relevant=len(relevant_facts) if attempt_num == 1 else None,
            facts_total=len(clean_facts) if attempt_num == 1 else None,
        )

        if best is None or (n_relevant_chunks, mean_relevance) > best[:2]:
            best = (n_relevant_chunks, mean_relevance, graded, current_query)

        # One very strong item is enough on its own: some answers sit in a
        # single paragraph (or a single figure), and requiring a second one
        # made the result depend on how a weaker, loosely related item graded.
        scores = [g.relevance for g in graded] + [f["_relevance"] for f in relevant_facts]
        has_strong = any(s >= sh.strong_chunk_threshold for s in scores)

        if n_relevant >= sh.min_relevant_chunks or has_strong:
            return RetrievalOutcome(
                original_query=query,
                final_query=current_query,
                succeeded=True,
                chunks=graded,
                attempts=attempts,
                structured_facts=relevant_facts + flagged_facts,
                relevance_threshold=sh.relevance_threshold,
            )

        if stagnated:
            emit(on_progress, "stagnated", attempt=attempt_num)
            break  # rewriting again would just burn the remaining attempts for nothing

        # Nothing on the first try even comes close: the filing almost
        # certainly doesn't cover this, and rewording rarely rescues it.
        # Saves two rewrites (~40 s) on questions that end in a refusal anyway.
        if attempt_num == 1 and max(scores + fact_scores, default=0.0) < sh.give_up_below:
            emit(on_progress, "give_up", best=round(max(scores + fact_scores, default=0.0), 2),
                 below=sh.give_up_below)
            break

        prev_chunk_ids = chunk_ids
        if attempt_num < sh.max_attempts:
            current_query = rewrite_query(query, current_query, graded, conf)
            emit(on_progress, "rewrite", attempt=attempt_num + 1, query=current_query)

    # Abstain with the BEST attempt's evidence, not just the last one's.
    best_n, _, best_chunks, best_query = best
    return RetrievalOutcome(
        original_query=query,
        final_query=best_query,
        succeeded=False,
        abstain_reason=(
            f"only {best_n} relevant chunk(s) found after {len(attempts)} attempt(s) "
            f"(need {sh.min_relevant_chunks}, or one scoring {sh.strong_chunk_threshold}+)"
        ),
        chunks=best_chunks,
        attempts=attempts,
        structured_facts=relevant_facts + flagged_facts,
        relevance_threshold=sh.relevance_threshold,
    )


# --- Step 4: one more search when the answer step finds nothing answers it --

# A question asking WHY a figure changed matches the figure, not its cause:
# "what caused the FY2024 net income dip?" finds the net income lines, and the
# answer step rightly says nothing there explains the dip. The cause (a tax
# charge) sits in a table the question's words never reach. Splitting the
# question into what the figure is calculated from - one search per part -
# does reach it. Tested on the real model before this was written: the
# split reached the tax reconciliation in 2 of 2 runs; rewording the
# question as one query reached it in 0 of 2.
_SPLIT_PROMPT = """A question about a company's 10-K filing was searched \
as written, and the evidence found did not answer it. It may need several \
separate facts that no single search reaches.

Question: {question}

List the separate facts needed to answer it, as at most {max_parts} short \
search queries: one per figure or topic, naming the line item as a 10-K \
states it and the fiscal years that matter.

When the question asks WHY a figure changed, the parts must cover what that \
figure is calculated from, so the cause can be found: a per-share figure \
comes from earnings and the share count; net income from operating income, \
other income and the provision for income taxes; a margin from sales and \
costs. Add a part for one-time items or charges in the year that changed, \
naming the kinds of item (for example an impairment, a legal settlement, a \
restructuring or a tax charge).

Ask only for what the question needs, and only for line items a 10-K \
actually reports - never invent one. Never swap in a different measure: if \
the question asks for units sold, a part asks for units sold, not revenue.

If the question needs only one fact, return an empty list."""

_SPLIT_SCHEMA = {
    "type": "object",
    "properties": {"parts": {"type": "array", "items": {"type": "string"}}},
    "required": ["parts"],
    "additionalProperties": False,
}

MAX_PARTS = 4
# Best few passages per part: 4 parts x 3 bounds the one grading call.
PASSAGES_PER_PART = 3


def split_question(query: str, cfg: Config) -> list[str]:
    """The question's separate facts as short search queries; [] when it is
    one fact. The extra call is an extra try, so a failed one returns []
    and leaves the answer as it was."""
    try:
        resp = get_groq_client(cfg).chat.completions.create(
            model=cfg.generation.utility_model,
            messages=[{"role": "user", "content": _SPLIT_PROMPT.format(question=query, max_parts=MAX_PARTS)}],
            max_tokens=cfg.generation.utility_max_tokens,
            temperature=0.0,
            reasoning_effort=cfg.generation.utility_effort,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "question_parts", "schema": _SPLIT_SCHEMA, "strict": True},
            },
        )
        parts = json.loads(resp.choices[0].message.content or "")["parts"]
    except Exception:  # noqa: BLE001
        return []
    clean = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
    return list(dict.fromkeys(clean))[:MAX_PARTS]


def search_parts(
    outcome: RetrievalOutcome, cfg: Config | None = None, on_progress: ProgressFn | None = None
) -> RetrievalOutcome | None:
    """The question's own search found no answer - Layer 4 refused the
    evidence, or Layer 5 found it doesn't answer the question: split the
    question, search each part, and grade what's found together with what was
    already relevant - against the question as asked, with the parts shown.
    None if the question can't be split; called at most once per question.
    """
    conf = cfg or get_config()
    sh = conf.self_healing
    query = outcome.original_query
    parts = split_question(query, conf)
    if not parts:
        return None
    attempt_num = len(outcome.attempts) + 1
    emit(on_progress, "split", attempt=attempt_num, parts=parts)

    passages = {g.retrieved.chunk.chunk_id: g.retrieved for g in outcome.relevant_chunks}
    figures = {(f["table"], f["label"], f["year"]): {k: v for k, v in f.items() if k != "_relevance"}
               for f in outcome.structured_facts if "value" in f}
    for part in parts:
        for rc in hybrid_retrieve(part, conf)[:PASSAGES_PER_PART]:
            passages.setdefault(rc.chunk.chunk_id, rc)
        for f in get_structured_facts(part, conf):
            if "value" in f:
                figures.setdefault((f["table"], f["label"], f["year"]), f)
    retrieved, facts = list(passages.values()), list(figures.values())[:MAX_FACTS]

    emit(on_progress, "grade", attempt=attempt_num, passages=len(retrieved), facts=len(facts))
    graded, fact_scores = grade_chunks(query, retrieved, conf, facts, parts)
    relevant_facts = [{**f, "_relevance": s} for f, s in zip(facts, fact_scores) if s >= sh.relevance_threshold]
    n_relevant = sum(1 for g in graded if g.relevance >= sh.relevance_threshold)
    emit(on_progress, "graded", attempt=attempt_num, relevant=n_relevant, total=len(graded),
         facts_relevant=len(relevant_facts), facts_total=len(facts))
    scores = [g.relevance for g in graded] + [f["_relevance"] for f in relevant_facts]
    succeeded = (n_relevant + len(relevant_facts) >= sh.min_relevant_chunks
                 or any(s >= sh.strong_chunk_threshold for s in scores))
    record = AttemptRecord(
        attempt=attempt_num, query="; ".join(parts), strategy="decompose",
        n_retrieved=len(retrieved), n_relevant=n_relevant,
        mean_relevance=sum(g.relevance for g in graded) / len(graded) if graded else 0.0,
        chunk_ids=sorted(passages), note=f"split into {len(parts)} parts after the first search found no answer",
    )
    return RetrievalOutcome(
        original_query=query,
        final_query="; ".join(parts),
        succeeded=succeeded,
        abstain_reason="" if succeeded else f"searching the question's {len(parts)} parts found too little",
        chunks=graded,
        attempts=outcome.attempts + [record],
        structured_facts=relevant_facts + [f for f in outcome.structured_facts if "value" not in f],
        relevance_threshold=sh.relevance_threshold,
    )