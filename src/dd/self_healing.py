

from __future__ import annotations

import json

from groq import Groq

from .config import Config, get_config, get_secrets
from .models import AttemptRecord, GradedChunk, RetrievalOutcome, RetrievedChunk
from .retrieval import hybrid_retrieve

_CLIENT: Groq | None = None


def get_groq_client(cfg: Config) -> Groq:
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
            },
        }
    },
    "required": ["grades"],
}

_GRADE_PROMPT = """You are grading whether passages from a company's 10-K \
filing actually answer a question - not whether they're topically related, \
but whether they contain the specific information asked for.

Question: {question}

Passages:
{passages}

For EACH passage, in order, give a relevance score from 0.0 (does not answer \
the question) to 1.0 (directly answers it), and a one-sentence reason.
Return exactly {n} grades, in the same order as the passages."""


def grade_chunks(
    query: str, chunks: list[RetrievedChunk], cfg: Config
) -> list[GradedChunk]:
    """Grade every chunk in ONE call instead of one call per chunk.

    Config calls this out directly: Groq's free tier caps at 8K tokens/minute,
    and N separate calls resend the full instructions N times - at this scale
    that overhead is most of the budget, not the actual chunk content being
    graded. One call, one JSON array back, is what `batch_grading: true` means.
    """
    if not chunks:
        return []

    client = get_groq_client(cfg)
    passages = "\n\n".join(f"[{i}] {c.chunk.text}" for i, c in enumerate(chunks))
    prompt = _GRADE_PROMPT.format(question=query, passages=passages, n=len(chunks))

    resp = client.chat.completions.create(
        model=cfg.generation.utility_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=cfg.generation.utility_max_tokens,
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "chunk_grades", "schema": _GRADE_SCHEMA},
        },
    )

    parsed = json.loads(resp.choices[0].message.content)
    grades = parsed["grades"]

    if len(grades) != len(chunks):
        # The model didn't return one grade per passage as instructed. Treat
        # every ungraded chunk as failing rather than guess which one it
        # meant - a wrong guess here silently corrupts every score above it.
        grades = (grades + [{"relevance": 0.0, "reason": "ungraded"}] * len(chunks))[
            : len(chunks)
        ]

    return [
        GradedChunk(retrieved=c, relevance=g["relevance"], reason=g["reason"])
        for c, g in zip(chunks, grades)
    ]


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
        reasoning_effort=cfg.generation.utility_effort,
    )
    new_query = (resp.choices[0].message.content or "").strip()

    # If the model still returns nothing usable, fall back to the query that
    # was actually tried rather than search on an empty string. Reusing it
    # means the next attempt's chunk set will look identical to this one,
    # which correctly triggers stagnation and ends the loop honestly -
    # instead of silently burning an attempt on a search for nothing.
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

    attempts: list[AttemptRecord] = []
    current_query = query
    prev_chunk_ids: set[str] | None = None
    graded: list[GradedChunk] = []

    for attempt_num in range(1, sh.max_attempts + 1):
        retrieved = hybrid_retrieve(current_query, conf)
        graded = grade_chunks(current_query, retrieved, conf)

        # Read the threshold from config here rather than trust
        # GradedChunk.is_relevant's own hardcoded 0.5 - that property and this
        # config value happen to agree today, but only one of them is the
        # actual source of truth, and it's this one.
        chunk_ids = {g.retrieved.chunk.chunk_id for g in graded}
        n_relevant = sum(1 for g in graded if g.relevance >= sh.relevance_threshold)
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
                n_relevant=n_relevant,
                mean_relevance=mean_relevance,
                chunk_ids=sorted(chunk_ids),
                stagnated=stagnated,
            )
        )

        if n_relevant >= sh.min_relevant_chunks:
            return RetrievalOutcome(
                original_query=query,
                final_query=current_query,
                succeeded=True,
                chunks=graded,
                attempts=attempts,
            )

        if stagnated:
            break  # rewriting again would just burn the remaining attempts for nothing

        prev_chunk_ids = chunk_ids
        if attempt_num < sh.max_attempts:
            current_query = rewrite_query(query, current_query, graded, conf)

    n_relevant_final = sum(1 for g in graded if g.relevance >= sh.relevance_threshold)
    return RetrievalOutcome(
        original_query=query,
        final_query=current_query,
        succeeded=False,
        abstain_reason=(
            f"only {n_relevant_final} relevant chunk(s) found "
            f"after {len(attempts)} attempt(s) (need {sh.min_relevant_chunks})"
        ),
        chunks=graded,
        attempts=attempts,
    )