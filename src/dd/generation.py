"""Layer 5: grounded generation + confidence scoring.

Takes Layer 4's RetrievalOutcome and turns it into an Answer: text written
only from the evidence handed over, every claim tied to a citation marker,
and a confidence score whose parts are all visible.

Three ways this layer refuses to answer, on purpose:
1. Layer 4 already abstained (succeeded=False) - no LLM call at all.
2. The model reads the evidence and says it doesn't answer the question
   (e.g. "iPhone units sold" when the evidence only has iPhone revenue).
3. The model answers but cites nothing real - an uncited answer breaks the
   one promise this layer makes, so it is not shown as an answer.
"""

from __future__ import annotations

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
from .self_healing import get_groq_client, self_heal

# --- Step 1: lay out the evidence with markers -----------------------------


def _is_per_share(fact: dict) -> bool:
    """EPS and other per-share amounts are already in dollars per share and
    must never be scaled. Checked by XBRL concept first, label as a backup."""
    concept = str(fact.get("concept") or "")
    return "PerShare" in concept or "per share" in fact["label"].lower()


def _format_fact_value(fact: dict) -> str:
    """Show a database value in the SAME unit as the 10-K's own tables.

    The warehouse stores whole dollars (112010000000.0), but the filing's
    tables - which reach the prompt as narrative chunks - state "(in millions)".
    Putting both side by side made the model convert between them, where a
    slip is a 1000x error. Every dollar fact is therefore shown in millions:
    112010000000.0 -> '$112,010 million'; -120686000000.0 -> '-$120,686 million'.
    Per-share amounts stay as they are: 7.46 -> '$7.46 per share'.
    The stored value is untouched - only what the model reads changes.
    """
    value = float(fact["value"])
    sign = "-" if value < 0 else ""
    if _is_per_share(fact):
        return f"{sign}${abs(value):,.2f} per share"
    millions = abs(value) / 1_000_000
    shown = f"{millions:,.0f}" if millions.is_integer() else f"{millions:,.2f}"
    return f"{sign}${shown} million"


def _clean_facts(outcome: RetrievalOutcome) -> list[dict]:
    """Only structured facts that resolved to one real value. Flagged ones
    (ambiguous, no year given, year not in the data) carry no number, so
    there is nothing in them for the model to cite.
    """
    return [f for f in outcome.structured_facts if "value" in f]


def _fact_line(fact: dict) -> str:
    """Spell out exactly what the number is, so the model can tell
    'iPhone revenue in dollars' apart from a question about iPhone units.
    """
    table = fact["table"].replace("_", " ")
    return (
        f"{fact['label']}, fiscal year {fact['year']}: "
        f"{_format_fact_value(fact)} (from Apple's {table})"
    )


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
        lines.append(f"{marker} ({c.item} / {c.subsection})\n{c.text}")

    for i, fact in enumerate(facts, start=1):
        marker = f"[S{i}]"
        lookup[marker] = fact
        lines.append(f"{marker} (financial statement data)\n{_fact_line(fact)}")

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
- Check that the evidence answers what was actually asked. A number about \
the right topic is not enough if it measures something else (for example, \
revenue in dollars does not answer a question about units sold).
- If the evidence does not answer the question, set "answerable" to false \
and use "answer" to say briefly what is missing.
- Be concise. State numbers exactly as given, always with their unit. \
Financial tables in a 10-K are often stated "in millions" - if the evidence \
says so, say so (e.g. "$73,733 million"), never a bare number.
- Financial statement data [S1], [S2], ... is already in millions of US \
dollars, the same unit as the filing's tables (per-share amounts are in \
dollars per share). Keep every number in the unit shown; do not convert."""


def _ask_model(question: str, evidence: str, cfg: Config) -> dict:
    client = get_groq_client(cfg)
    resp = client.chat.completions.create(
        model=cfg.generation.model,
        messages=[
            {
                "role": "user",
                "content": _ANSWER_PROMPT.format(question=question, evidence=evidence),
            }
        ],
        max_tokens=cfg.generation.max_tokens,
        temperature=cfg.generation.temperature,
        reasoning_effort=cfg.generation.effort,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "grounded_answer", "schema": _ANSWER_SCHEMA},
        },
    )
    return json.loads(resp.choices[0].message.content)


# --- Step 3: check which citations are real ---------------------------------

_MARKER_RE = re.compile(r"\[S?\d+\]")


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
                    quote=_fact_line(ev),
                )
            )
        else:  # narrative chunk
            c = ev.retrieved.chunk
            citations.append(
                Citation(marker=marker, chunk_id=c.chunk_id, source=c.citation())
            )

    if invented:
        notes.append(f"model cited evidence that doesn't exist: {', '.join(invented)} (dropped)")
    return citations, invented, notes


# --- Step 4: confidence -----------------------------------------------------


def _loop_cleanliness(outcome: RetrievalOutcome, cfg: Config) -> float:
    """1.0 if Layer 4 succeeded on the first try, falling to 0.0 if it
    needed every attempt. More rewrites = the original question matched the
    filing less directly."""
    max_attempts = cfg.self_healing.max_attempts
    if max_attempts <= 1:
        return 1.0
    return 1.0 - (len(outcome.attempts) - 1) / (max_attempts - 1)


def _band(score: float, cfg: Config) -> str:
    b = cfg.confidence.bands
    if score >= b.high:
        return "high"
    if score >= b.medium:
        return "medium"
    return "low"


def _confidence(
    outcome: RetrievalOutcome,
    citations: list[Citation],
    lookup: dict[str, GradedChunk | dict],
    cfg: Config,
) -> ConfidenceBreakdown:
    """Score only the evidence the answer actually cites, not everything
    retrieved - a strong chunk the answer never used shouldn't raise
    confidence in that answer.
    """
    cited_chunks = [lookup[c.marker] for c in citations if not isinstance(lookup[c.marker], dict)]
    cited_facts = [lookup[c.marker] for c in citations if isinstance(lookup[c.marker], dict)]

    # Signal 1: share of cited chunks that BOTH dense and keyword search found.
    hybrid = (
        sum(1 for g in cited_chunks if g.retrieved.found_by_both) / len(cited_chunks)
        if cited_chunks else 0.0
    )
    # Signal 2: strongest cross-encoder score among cited chunks. Measured on
    # the eval set: answerable questions topped out at 0.41-0.999, unanswerable
    # ones at 0.0005-0.022, so the raw 0-1 score is used as-is, no rescaling.
    rerank = max(
        (g.retrieved.rerank_score or 0.0 for g in cited_chunks), default=0.0
    )
    # Signal 3: did Layer 4 get there on the first try?
    loop = _loop_cleanliness(outcome, cfg)

    w = cfg.confidence.weights
    notes: list[str] = []

    if cited_facts:
        # The formula above only describes narrative search. An answer that
        # cites an exact database value would score low whenever the prose
        # around that number was weak - a correct, exact answer labelled
        # "low". So a cited structured fact gets its own score instead.
        fuzzy = any(f.get("_matched_via") == "fuzzy" for f in cited_facts)
        score = 0.60 if fuzzy else 0.90
        notes.append(
            "answer rests on an exact value from the financial statements"
            + (" (matched by fuzzy wording - check the label)" if fuzzy else "")
        )
    else:
        score = w.hybrid_agreement * hybrid + w.rerank_strength * rerank + w.loop_cleanliness * loop

    if len(outcome.attempts) > 1:
        notes.append(f"needed {len(outcome.attempts)} retrieval attempts")

    return ConfidenceBreakdown(
        score=round(score, 3),
        band=_band(score, cfg),
        hybrid_agreement=round(hybrid, 3),
        rerank_strength=round(rerank, 3),
        loop_cleanliness=round(loop, 3),
        notes=notes,
    )


# --- Entry point -------------------------------------------------------


def generate_answer(outcome: RetrievalOutcome, cfg: Config | None = None) -> Answer:
    """Layer 5's core: RetrievalOutcome in, Answer out."""
    conf = cfg or get_config()
    query = outcome.original_query

    # Refusal 1: Layer 4 already decided the evidence is too weak.
    if not outcome.succeeded:
        return Answer(query=query, answered=False, text=outcome.abstain_reason, outcome=outcome)

    chunks = outcome.relevant_chunks
    facts = _clean_facts(outcome)
    evidence, lookup = _build_evidence(chunks, facts)
    result = _ask_model(query, evidence, conf)

    # Refusal 2: the evidence is on-topic but doesn't answer what was asked.
    if not result["answerable"]:
        return Answer(query=query, answered=False, text=result["answer"], outcome=outcome)

    # Check markers from both places: inline in the text, and the required
    # "citations" list. Wrap bare "S1" as "[S1]" in case the model drops the
    # brackets in the list.
    listed = [m if m.startswith("[") else f"[{m}]" for m in (s.strip() for s in result.get("citations", []))]
    citations, _, notes = _check_citations(result["answer"] + " " + " ".join(listed), lookup, conf)

    # Refusal 3: an answer with no real citations is an unsupported claim.
    if not citations:
        return Answer(
            query=query,
            answered=False,
            text="The model produced an answer without citing any real evidence, so it was not shown.",
            outcome=outcome,
        )

    confidence = _confidence(outcome, citations, lookup, conf)
    confidence.notes = notes + confidence.notes
    return Answer(
        query=query,
        answered=True,
        text=result["answer"],
        citations=citations,
        confidence=confidence,
        outcome=outcome,
    )


def answer(query: str, cfg: Config | None = None) -> Answer:
    """Full pipeline in one call: Layers 3 + 4 (self_heal) then Layer 5."""
    conf = cfg or get_config()
    start = time.perf_counter()
    result = generate_answer(self_heal(query, conf), conf)
    result.elapsed_s = round(time.perf_counter() - start, 2)
    return result