"""Shared data contracts passed between layers.

Layer 2 emits Chunk. Layer 3 emits RetrievedChunk. Layer 4 emits GradedChunk
and RetrievalOutcome. Layer 5 emits Answer. Nothing else crosses layer lines.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class SourceKind(str, Enum):
    NARRATIVE = "narrative"   # prose from the filing, retrieved by hybrid search
    STRUCTURED = "structured"  # a number from DuckDB/XBRL, retrieved by SQL


class Chunk(BaseModel):
    """One indexed unit of the filing."""

    chunk_id: str
    text: str
    kind: SourceKind = SourceKind.NARRATIVE

    # Provenance - every one of these ends up in a citation.
    company: str
    ticker: str
    form: str
    fiscal_year: str
    item: str = ""          # e.g. "Item 1A"
    section: str = ""       # e.g. "Risk Factors"
    subsection: str = ""    # nearest heading above the chunk
    char_start: int = 0
    char_end: int = 0

    def citation(self) -> str:
        bits = [self.ticker, self.form, self.fiscal_year]
        if self.item:
            bits.append(self.item)
        if self.section and self.section != self.item:
            bits.append(self.section)
        return " | ".join(b for b in bits if b)


class RetrievedChunk(BaseModel):
    """A chunk plus every score that led to it being retrieved.

    These scores are the raw material for the Layer 5 confidence calculation,
    so they are carried all the way through rather than discarded after ranking.
    """

    chunk: Chunk

    vector_rank: int | None = None   # 1-based rank in the dense list, None if absent
    vector_score: float | None = None
    bm25_rank: int | None = None     # 1-based rank in the keyword list, None if absent
    bm25_score: float | None = None

    rrf_score: float = 0.0
    rerank_score: float | None = None  # cross-encoder logit, higher is better

    @property
    def found_by_both(self) -> bool:
        """True when dense and sparse retrieval independently surfaced this chunk.

        Agreement between two methods that fail differently is the single
        strongest cheap signal that a chunk is genuinely on-topic.
        """
        return self.vector_rank is not None and self.bm25_rank is not None


class GradedChunk(BaseModel):
    """A retrieved chunk after the Layer 4 relevance grader has judged it."""

    retrieved: RetrievedChunk
    relevance: float = Field(ge=0.0, le=1.0)
    reason: str = ""

class AttemptRecord(BaseModel):
    """What happened on one pass through the retrieve-and-grade loop."""

    attempt: int
    query: str
    strategy: str                 # "original" | "rewrite" | "decompose"
    n_retrieved: int
    n_relevant: int
    mean_relevance: float
    chunk_ids: list[str]
    stagnated: bool = False       # returned essentially the same chunks as last time
    note: str = ""


class RetrievalOutcome(BaseModel):
    """The full Layer 3 + Layer 4 result handed to Layer 5."""

    original_query: str
    final_query: str
    succeeded: bool               # False means: abstain, do not generate an answer
    abstain_reason: str = ""
    # The threshold Layer 4 judged these chunks against. Required, with no
    # default: a default would let a caller forget it and silently filter at a
    # different cutoff than the loop that decided `succeeded`.
    relevance_threshold: float = Field(ge=0.0, le=1.0)
    chunks: list[GradedChunk] = Field(default_factory=list)
    structured_facts: list[dict[str, Any]] = Field(default_factory=list)
    attempts: list[AttemptRecord] = Field(default_factory=list)

    @property
    def relevant_chunks(self) -> list[GradedChunk]:
        """Chunks that cleared the same threshold Layer 4 used to decide success."""
        return [c for c in self.chunks if c.relevance >= self.relevance_threshold]


class Citation(BaseModel):
    marker: str        # "[1]"
    chunk_id: str
    source: str        # human-readable provenance line
    quote: str = ""    # the span the claim rests on


class ConfidenceBreakdown(BaseModel):
    """Deliberately transparent: the user sees what produced the number.

    This is a heuristic signal, not a calibrated probability. It only becomes
    trustworthy once it has been validated against a labelled eval set.
    """

    score: float
    band: Literal["high", "medium", "low"]
    hybrid_agreement: float
    rerank_strength: float
    loop_cleanliness: float
    notes: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    query: str
    answered: bool
    text: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: ConfidenceBreakdown | None = None
    outcome: RetrievalOutcome | None = None
    elapsed_s: float = 0.0
