"""Tests for Layer 3's final ranking, with the searches and the reranker
replaced by stand-ins - offline, no models, no index.

Why this file exists: for "What risks does Apple disclose related to tariffs?"
both searches ranked the MD&A tariff passages near the top (vector #2,
BM25 #4), but the reranker scored them ~0.0005 - below an unrelated
credit-risk passage - and, as the only judge of the final 6, cut them.

Run from the project root:   pytest
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import dd.retrieval as ret
from dd.config import get_config
from dd.models import Chunk


def chunk(cid: str, text: str | None = None) -> Chunk:
    return Chunk(chunk_id=cid, text=text or f"text of {cid}", company="Apple Inc.",
                 ticker="AAPL", form="10-K", fiscal_year="FY2025", item="Item 1A",
                 section="Risk Factors")


@pytest.fixture
def world(monkeypatch):
    """setup(dense ids, bm25 ids, reranker scores by id, optional chunk texts)"""
    def setup(dense, sparse, rerank_scores, texts=None):
        texts = texts or {}
        ids = dict.fromkeys(dense + sparse)
        monkeypatch.setattr(ret, "dense_search", lambda q, cfg, k: [(c, 1.0) for c in dense])
        monkeypatch.setattr(ret, "sparse_search", lambda q, cfg, k: [(c, 1.0) for c in sparse])
        monkeypatch.setattr(ret, "read_chunks", lambda path: [chunk(c, texts.get(c)) for c in ids])

        def predict(pairs, batch_size):
            # the chunk id is in the text the reranker reads
            return [next(s for c, s in rerank_scores.items()
                         if t.endswith(texts.get(c, f"text of {c}")))
                    for _, t in pairs]
        monkeypatch.setattr(ret, "get_reranker", lambda name: SimpleNamespace(predict=predict))
    return setup


def ids(results) -> list[str]:
    return [r.chunk.chunk_id for r in results]


def test_reranker_alone_cannot_drop_what_both_searches_rank_highly(world):
    # "tariffs" is #2 in both searches; the reranker scores it lowest of all.
    # Six filler passages the reranker likes more, found by only one search.
    fillers = [f"f{i}" for i in range(6)]
    world(dense=["a", "tariffs", *fillers[:3]],
          sparse=["a", "tariffs", *fillers[3:]],
          rerank_scores={"a": 0.9, "tariffs": 0.0004, **{f: 0.01 for f in fillers}})
    top = ids(ret.hybrid_retrieve("tariff risks?", get_config()))
    assert len(top) == get_config().retrieval.final_top_k
    assert "tariffs" in top, "both searches agreed on it - the reranker gets a vote, not a veto"


def test_reranker_still_decides_between_otherwise_equal_passages(world):
    # Same search ranks for x and y (each found by one search at #2): the
    # reranker's opinion breaks the tie.
    world(dense=["a", "x"], sparse=["a", "y"], rerank_scores={"a": 0.9, "x": 0.1, "y": 0.8})
    assert ids(ret.hybrid_retrieve("q?", get_config())) == ["a", "y", "x"]


def test_identical_passages_take_only_one_slot(world):
    same = "Although most components essential to the Company's business are generally available."
    world(dense=["item1", "item8", "b"], sparse=["item1", "item8", "b"],
          rerank_scores={"item1": 0.5, "item8": 0.5, "b": 0.4},
          texts={"item1": same, "item8": "  " + same.replace(" ", "  "), "b": "text of b"})
    assert ids(ret.hybrid_retrieve("components?", get_config())) == ["item1", "b"]


def test_passages_only_one_search_likes_cannot_outvote_the_reranker(world):
    # Cybersecurity question: each search had boilerplate at #2 ("Available
    # Information", a different chunk in each), and the reranker rightly put
    # the real risk passages first. Giving the searches a vote let five
    # boilerplate passages in and the question was refused.
    # As in the real probe, the risk passages are found by vector search only
    # (#4-#9) and each search has its own boilerplate at the top.
    relevant = [f"risk{i}" for i in range(6)]
    world(dense=["junk_d1", "junk_d2", "junk_d3", *relevant],
          sparse=["junk_s1", "junk_s2", "junk_s3"],
          rerank_scores={**{r: 0.9 for r in relevant},
                         **{j: 0.001 for j in ["junk_d1", "junk_d2", "junk_d3", "junk_s1", "junk_s2", "junk_s3"]}})
    assert sorted(ids(ret.hybrid_retrieve("cybersecurity risks?", get_config()))) == relevant
