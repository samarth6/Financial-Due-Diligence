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

from groq import Groq

from .config import Config, get_config, get_secrets
from .models import AttemptRecord, GradedChunk, RetrievalOutcome, RetrievedChunk
from .retrieval import hybrid_retrieve
from .structured import describe_fact, get_structured_facts

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
                    "relevance": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["relevance", "reason"],
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

Evidence:
{passages}

Items marked [S1], [S2]... are figures from the financial statements. A \
figure is relevant only if it is one of the numbers needed to answer the \
question - the right measure, not just the right topic (revenue in dollars \
is not "units sold").

For EACH item, in order, give a relevance score from 0.0 (does not answer \
the question) to 1.0 (directly answers it), and a one-sentence reason.
Return exactly {n} grades, in the same order as the items."""


def _clamp(x: object) -> float:
    """Scores must be 0-1 for GradedChunk; a model writing 1.2 or "0.8"
    shouldn't crash the question."""
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def grade_chunks(
    query: str, chunks: list[RetrievedChunk], cfg: Config, facts: list[dict] | None = None
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

    items = [f"[{i}] {c.chunk.text}" for i, c in enumerate(chunks)]
    items += [f"[S{i}] {describe_fact(f)}" for i, f in enumerate(facts, start=1)]
    prompt = _GRADE_PROMPT.format(question=query, passages="\n\n".join(items), n=n)

    resp = get_groq_client(cfg).chat.completions.create(
        model=cfg.generation.utility_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=cfg.generation.utility_max_tokens,
        temperature=0.0,
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

    try:
        grades = json.loads(resp.choices[0].message.content or "")["grades"]
        grades = [{"relevance": _clamp(g.get("relevance")), "reason": str(g.get("reason", ""))} for g in grades]
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        # Unreadable reply: fail every item rather than crash. The question
        # is then refused honestly ("0 relevant") instead of erroring out.
        grades = [{"relevance": 0.0, "reason": "grader failed"}] * n

    if len(grades) != n:
        # The model didn't return one grade per item as instructed. Treat
        # every ungraded item as failing rather than guess which one it
        # meant - a wrong guess here silently corrupts every score above it.
        grades = (grades + [{"relevance": 0.0, "reason": "ungraded"}] * n)[:n]

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


def self_heal(query: str, cfg: Config | None = None) -> RetrievalOutcome:
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

    attempts: list[AttemptRecord] = []
    current_query = query
    prev_chunk_ids: set[str] | None = None
    graded: list[GradedChunk] = []
    best: tuple[int, float, list[GradedChunk], str] | None = None  # (n_relevant, mean, chunks, query)

    for attempt_num in range(1, sh.max_attempts + 1):
        retrieved = hybrid_retrieve(current_query, conf)
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
            break  # rewriting again would just burn the remaining attempts for nothing

        # Nothing on the first try even comes close: the filing almost
        # certainly doesn't cover this, and rewording rarely rescues it.
        # Saves two rewrites (~40 s) on questions that end in a refusal anyway.
        if attempt_num == 1 and max(scores + fact_scores, default=0.0) < sh.give_up_below:
            break

        prev_chunk_ids = chunk_ids
        if attempt_num < sh.max_attempts:
            current_query = rewrite_query(query, current_query, graded, conf)

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