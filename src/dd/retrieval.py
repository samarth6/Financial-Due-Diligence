"""Layer 3: hybrid retrieval.

Takes a question, gets it answered by BOTH indexes Layer 2 built, fuses the
two ranked lists with Reciprocal Rank Fusion (RRF), then re-scores the fused
shortlist with a cross-encoder for precision.

Two-step design, same reasoning as `cli.py`'s `probe` command already
demonstrated: RRF is fast but rough (good for narrowing hundreds of
candidates down to ~25), the cross-encoder is slow but accurate (only run on
that small shortlist, never on the whole index).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .chunking import read_chunks
from .config import CHUNKS_PATH, Config, get_config
from .embedding import embed_query
from .indexing import get_qdrant_client, load_bm25
from .models import Chunk, RetrievedChunk

# --- Step 1: independent dense + sparse retrieval -------------------------


def dense_search(query: str, cfg: Config, top_k: int) -> list[tuple[str, float]]:
    """Top-k (chunk_id, score) from Qdrant. 1-based rank is the list position."""
    client = get_qdrant_client(cfg)
    hits = client.query_points(
        collection_name=cfg.vector_store.collection,
        query=embed_query(query, cfg).tolist(),
        limit=top_k,
    ).points
    return [(h.payload["chunk_id"], h.score) for h in hits]


def sparse_search(query: str, cfg: Config, top_k: int) -> list[tuple[str, float]]:
    """Top-k (chunk_id, score) from BM25. Already stops at score <= 0."""
    return load_bm25().search(query, top_k)


# --- Step 2: Reciprocal Rank Fusion ----------------------------------------


def reciprocal_rank_fusion(
    dense: list[tuple[str, float]],
    sparse: list[tuple[str, float]],
    rrf_k: int,
) -> dict[str, dict[str, Any]]:
    """Fuse two incomparable-score rankings into one, by rank position.

    Cosine similarity (~0-1) and BM25 (unbounded) cannot be added directly -
    a 0.69 and a 12.98 mean nothing next to each other. Rank position is the
    one thing both lists share a scale for: 1st place means the same thing
    regardless of which scoring produced it. `rrf_k` (60, the constant from
    the original RRF paper) dampens the gap between adjacent ranks so one
    method's rank-1-vs-rank-2 noise doesn't swing the fused result.

    Returns chunk_id -> {rrf_score, vector_rank, vector_score, bm25_rank,
    bm25_score}, i.e. every field `RetrievedChunk` needs, so nothing found
    here has to be recomputed later.
    """
    out: dict[str, dict[str, Any]] = {}

    for rank, (chunk_id, score) in enumerate(dense, start=1):
        row = out.setdefault(chunk_id, {})
        row["vector_rank"] = rank
        row["vector_score"] = score
        row["rrf_score"] = row.get("rrf_score", 0.0) + 1.0 / (rrf_k + rank)

    for rank, (chunk_id, score) in enumerate(sparse, start=1):
        row = out.setdefault(chunk_id, {})
        row["bm25_rank"] = rank
        row["bm25_score"] = score
        row["rrf_score"] = row.get("rrf_score", 0.0) + 1.0 / (rrf_k + rank)

    return out


# --- Step 3: cross-encoder reranking ---------------------------------------


@lru_cache(maxsize=1)
def get_reranker(model_name: str):
    """Load the cross-encoder once per process. Imported lazily, same reason
    as `embedding.get_encoder`: torch import cost shouldn't be paid by code
    that never touches the reranker.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    cfg: Config,
) -> list[RetrievedChunk]:
    """Score each (query, chunk_text) pair jointly, then re-sort.

    Unlike dense/sparse retrieval, which score the query and the chunk
    separately and compare the results, a cross-encoder reads both together
    in one forward pass - slower, but it can weigh how the specific words in
    the question relate to the specific words in this specific chunk, not
    just whether they land near each other in a general-purpose vector space.
    """
    if not candidates:
        return candidates

    model = get_reranker(cfg.reranker.model)
    pairs = [
    (query, f"[SEC 10-K filing, {c.chunk.item} - {c.chunk.section}] {c.chunk.text}")
    for c in candidates
    ]
    scores = model.predict(pairs, batch_size=cfg.reranker.batch_size)

    for c, score in zip(candidates, scores):
        c.rerank_score = float(score)

    return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)


# --- Entry point -------------------------------------------------------


def hybrid_retrieve(query: str, cfg: Config | None = None) -> list[RetrievedChunk]:
    """Layer 3's entry point. Question in, ranked RetrievedChunks out."""
    conf = cfg or get_config()
    r = conf.retrieval

    dense = dense_search(query, conf, r.vector_top_k)
    sparse = sparse_search(query, conf, r.bm25_top_k)
    fused = reciprocal_rank_fusion(dense, sparse, r.rrf_k)

    # Rank by fused score, keep only the top `rerank_candidates` - the
    # cross-encoder is the expensive step, so it only ever sees the shortlist
    # RRF has already narrowed things down to, never the full candidate pool.
    ranked_ids = sorted(fused, key=lambda cid: fused[cid]["rrf_score"], reverse=True)
    shortlist_ids = ranked_ids[: r.rerank_candidates]

    by_id: dict[str, Chunk] = {c.chunk_id: c for c in read_chunks(CHUNKS_PATH)}

    candidates = [
        RetrievedChunk(chunk=by_id[cid], **fused[cid])
        for cid in shortlist_ids
        if cid in by_id  # defensive: a stale index could reference a dropped chunk
    ]
    candidates = _drop_duplicates(candidates)

    reranked = rerank(query, candidates, conf)

    # The reranker picks the final passages - it reads each one WITH the
    # question, so it reliably throws out boilerplate that merely shares words
    # with it ("Available Information", the signature page). One exception: a
    # passage BOTH searches rank in their top few is always kept. On the
    # tariffs question the reranker scored the MD&A tariff passage 0.0005
    # although it was vector #2 and BM25 #4, and cut it. (Giving the searches
    # a full vote instead let boilerplate in and broke the cybersecurity
    # question.)
    top = r.keep_if_both_top
    agreed = [
        c for c in reranked
        if c.vector_rank is not None and c.vector_rank <= top
        and c.bm25_rank is not None and c.bm25_rank <= top
    ][: r.final_top_k]
    rest = [c for c in reranked if c not in agreed]
    final = agreed + rest[: r.final_top_k - len(agreed)]
    return sorted(final, key=lambda c: c.rerank_score, reverse=True)


def _drop_duplicates(candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """The 10-K repeats some paragraphs word for word (e.g. Item 1 and Item 8
    both describe component supply). Each copy is its own chunk, and two
    copies took two of the six final slots. Keep the first, i.e. the one
    ranked higher by the searches."""
    seen: set[str] = set()
    out = []
    for c in candidates:
        key = " ".join(c.chunk.text.split()).lower()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out