"""Layer 2b: the embedding model, loaded once and shared.

bge-*-en-v1.5 is asymmetric by design: queries are prefixed with an
instruction, passages are not. Getting this backwards (or dropping the prefix)
measurably degrades recall, so the two call sites are separate functions rather
than one function with a flag that someone will eventually pass wrong.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import Config, get_config


@lru_cache(maxsize=1)
def get_encoder(model_name: str):
    """Load the sentence-transformer once per process.

    Imported lazily: torch takes seconds to import, and the chunker and the
    config layer must stay usable without paying that cost.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_passages(texts: list[str], cfg: Config | None = None) -> np.ndarray:
    """Embed chunk text for indexing. No instruction prefix."""
    conf = cfg or get_config()
    model = get_encoder(conf.embedding.model)
    vectors = model.encode(
        texts,
        batch_size=conf.embedding.batch_size,
        normalize_embeddings=True,  # cosine distance assumes unit vectors
        show_progress_bar=len(texts) > 200,
        convert_to_numpy=True,
    )
    _check_dim(vectors, conf)
    return vectors


def embed_query(query: str, cfg: Config | None = None) -> np.ndarray:
    """Embed a user question for search. Instruction prefix applied."""
    conf = cfg or get_config()
    model = get_encoder(conf.embedding.model)
    vectors = model.encode(
        [conf.embedding.query_prefix + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    _check_dim(vectors, conf)
    return vectors[0]


def _check_dim(vectors: np.ndarray, conf: Config) -> None:
    """Fail loudly on a model/config dim mismatch.

    Swapping embedding.model without updating embedding.dim would otherwise
    surface as a confusing Qdrant error, or worse, as a silently rebuilt
    collection that no longer matches the stored vectors.
    """
    got = int(vectors.shape[-1])
    if got != conf.embedding.dim:
        raise ValueError(
            f"{conf.embedding.model} produces {got}-dim vectors but "
            f"config.yaml embedding.dim is {conf.embedding.dim}"
        )
