"""Layer 2c: build the two indexes Layer 3 fuses.

Two indexes, because they fail differently. Dense vectors find "risks from
import duties" in a passage that only ever says "tariffs"; BM25 finds the
literal string "iPhone 17 Pro" that a 384-dim vector blurs into every other
phone. Layer 3's RRF only has something to fuse if both exist here.

The dense side lives in Qdrant (embedded by default, no Docker needed). The
sparse side is a pickled BM25 index over the same chunk list, in the same
order, so a rank from either side maps back to the same chunk_id.
"""

from __future__ import annotations

import atexit
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    BM25_PATH,
    CHUNKS_PATH,
    Config,
    get_config,
    get_secrets,
)
from .embedding import embed_passages
from .models import Chunk

# BM25 needs tokens, and financial text is full of tokens that a naive
# \w+ split destroys: "$112,010" -> ["112", "010"], "10-K" -> ["10", "K"].
# Keeping money, percentages and hyphenated forms intact is what lets the
# keyword index answer the numeric questions the vector index cannot.
TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*|\$?\d[\d,\.]*%?")

# Words this common in a 10-K carry no signal but inflate every BM25 score.
STOPWORDS = frozenset(
    """a an and are as at be been by can for from had has have in into is it its
    may not of on or other our that the their these this to was were which will
    with would company companys""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase tokens for BM25, preserving money/percent/hyphen forms."""
    return [
        t
        for t in (m.group(0).lower().rstrip(".") for m in TOKEN_RE.finditer(text))
        if t and t not in STOPWORDS
    ]


@dataclass
class Bm25Index:
    """A BM25 index plus the chunk_ids its integer positions refer to."""

    bm25: Any
    chunk_ids: list[str]

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return (chunk_id, score) for the top_k keyword matches."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(
            range(len(scores)), key=lambda i: float(scores[i]), reverse=True
        )
        out: list[tuple[str, float]] = []
        for i in ranked[:top_k]:
            # A zero score means no query term appears at all. Returning it
            # would hand Layer 3 a padded list of non-matches that RRF then
            # treats as evidence.
            if scores[i] <= 0:
                break
            out.append((self.chunk_ids[i], float(scores[i])))
        return out


def build_bm25(chunks: list[Chunk], path: Path = BM25_PATH) -> Bm25Index:
    from rank_bm25 import BM25Okapi

    corpus = [tokenize(c.text) for c in chunks]
    index = Bm25Index(bm25=BM25Okapi(corpus), chunk_ids=[c.chunk_id for c in chunks])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(index, fh)
    return index


def load_bm25(path: Path = BM25_PATH) -> Bm25Index:
    if not path.exists():
        raise FileNotFoundError(f"no BM25 index at {path} - run `dd index` first")
    with path.open("rb") as fh:
        return pickle.load(fh)


# --- dense side ----------------------------------------------------------


_CLIENT: Any = None


def get_qdrant_client(cfg: Config | None = None):
    """Embedded (local path) or server (Docker) per QDRANT_MODE in .env.

    A process-wide singleton, because embedded Qdrant takes a file lock on its
    storage directory and a second client would block on the first one's lock.
    Closed via atexit rather than __del__: Qdrant's own finaliser releases that
    lock during interpreter teardown, by which point msvcrt has been unloaded
    and it raises a spurious ModuleNotFoundError on every exit.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    from qdrant_client import QdrantClient

    secrets = get_secrets()
    if secrets.qdrant_mode == "server":
        _CLIENT = QdrantClient(url=secrets.qdrant_url)
    else:
        path = secrets.qdrant_abs_path
        path.mkdir(parents=True, exist_ok=True)
        _CLIENT = QdrantClient(path=str(path))

    atexit.register(_CLIENT.close)
    return _CLIENT


def build_vector_index(chunks: list[Chunk], cfg: Config | None = None) -> int:
    """Embed every chunk and (re)create the Qdrant collection from scratch.

    Recreated rather than upserted: a partial index whose vectors were produced
    by a different model or chunking config is worse than no index, because
    retrieval still "works" and the eval numbers quietly stop meaning anything.
    """
    from qdrant_client import models as qm

    conf = cfg or get_config()
    client = get_qdrant_client(conf)
    collection = conf.vector_store.collection

    vectors = embed_passages([c.text for c in chunks], conf)

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(
            size=conf.embedding.dim,
            distance=qm.Distance[conf.vector_store.distance.upper()],
        ),
    )

    client.upsert(
        collection_name=collection,
        points=[
            qm.PointStruct(
                # Qdrant ids must be int or UUID, so the position is the id and
                # chunk_id travels in the payload.
                id=i,
                vector=vectors[i].tolist(),
                payload=chunk.model_dump(mode="json"),
            )
            for i, chunk in enumerate(chunks)
        ],
    )
    return len(chunks)


def build_all(cfg: Config | None = None) -> dict[str, int]:
    """Build both indexes from the chunks on disk. Layer 2's entry point."""
    from .chunking import read_chunks

    conf = cfg or get_config()
    chunks = read_chunks(CHUNKS_PATH)
    n_vec = build_vector_index(chunks, conf)
    build_bm25(chunks)
    return {"chunks": len(chunks), "vectors": n_vec}
