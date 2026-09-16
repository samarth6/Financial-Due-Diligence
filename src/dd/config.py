"""Configuration loading: config.yaml for tunables, .env for secrets."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
BM25_PATH = PROCESSED_DIR / "bm25.pkl"
DUCKDB_PATH = DATA_DIR / "warehouse.duckdb"
EVAL_SET_PATH = PROJECT_ROOT / "eval" / "eval_set.json"

# The three XBRL statements Layer 1 writes and Layer 3's structured path reads.
# Each name doubles as the DuckDB table name and the CSV file-name suffix.
STATEMENTS: tuple[str, ...] = ("income_statement", "balance_sheet", "cash_flow")


class Secrets(BaseSettings):
    """Values from .env / the environment. Never committed."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = ""
    edgar_identity: str = ""
    qdrant_mode: Literal["embedded", "server"] = "embedded"
    qdrant_path: str = "data/qdrant"
    qdrant_url: str = "http://localhost:6333"

    @property
    def qdrant_abs_path(self) -> Path:
        p = Path(self.qdrant_path)
        return p if p.is_absolute() else PROJECT_ROOT / p


# --- config.yaml schema ---------------------------------------------------


class CompanyCfg(BaseModel):
    ticker: str = "AAPL"
    form: str = "10-K"
    year: str | int = "latest"
    # File-name stem for this company's artifacts under data/raw. Not derived
    # from the ticker: Layer 1 named its output after the company ("apple_10k"),
    # and eval_set.json cites those exact file names.
    slug: str = "apple"
    name: str = "Apple Inc."


class ChunkingCfg(BaseModel):
    target_chars: int = 900
    overlap_chars: int = 150
    min_chars: int = 120
    max_chars: int = 2000
    exclude_items: list[str] = Field(default_factory=lambda: ["Item 15"])


class EmbeddingCfg(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    dim: int = 384
    batch_size: int = 32
    query_prefix: str = "Represent this sentence for searching relevant passages: "


class VectorStoreCfg(BaseModel):
    collection: str = "filings"
    distance: str = "Cosine"


class RetrievalCfg(BaseModel):
    vector_top_k: int = 30
    bm25_top_k: int = 30
    rrf_k: int = 60
    rerank_candidates: int = 25
    final_top_k: int = 6


class RerankerCfg(BaseModel):
    model: str = "BAAI/bge-reranker-base"
    batch_size: int = 16


class SelfHealingCfg(BaseModel):
    max_attempts: int = 3
    relevance_threshold: float = 0.5
    min_relevant_chunks: int = 2
    stagnation_jaccard: float = 0.9
    batch_grading: bool = True


class GenerationCfg(BaseModel):
    model: str = "openai/gpt-oss-120b"
    max_tokens: int = 4000
    # gpt-oss exposes reasoning_effort natively: low | medium | high.
    effort: Literal["low", "medium", "high"] = "high"
    # Grounded generation must reproduce the retrieved text, not embellish it.
    temperature: float = 0.0
    utility_model: str = "openai/gpt-oss-20b"
    utility_max_tokens: int = 2000
    utility_effort: Literal["low", "medium", "high"] = "low"
    # Groq's OpenAI-compatible endpoint; overridable for a proxy or gateway.
    base_url: str = "https://api.groq.com/openai/v1"
    # Free tier is 8K tokens/minute, and Layer 4 can issue several calls per
    # question. 429s are expected, not exceptional - retry with backoff.
    max_retries: int = 5


class ConfidenceWeights(BaseModel):
    hybrid_agreement: float = 0.3
    rerank_strength: float = 0.45
    loop_cleanliness: float = 0.25


class ConfidenceBands(BaseModel):
    high: float = 0.70
    medium: float = 0.45


class ConfidenceCfg(BaseModel):
    weights: ConfidenceWeights = Field(default_factory=ConfidenceWeights)
    bands: ConfidenceBands = Field(default_factory=ConfidenceBands)


class Config(BaseModel):
    company: CompanyCfg = Field(default_factory=CompanyCfg)
    chunking: ChunkingCfg = Field(default_factory=ChunkingCfg)
    embedding: EmbeddingCfg = Field(default_factory=EmbeddingCfg)
    vector_store: VectorStoreCfg = Field(default_factory=VectorStoreCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    reranker: RerankerCfg = Field(default_factory=RerankerCfg)
    self_healing: SelfHealingCfg = Field(default_factory=SelfHealingCfg)
    generation: GenerationCfg = Field(default_factory=GenerationCfg)
    confidence: ConfidenceCfg = Field(default_factory=ConfidenceCfg)


@lru_cache(maxsize=1)
def get_config() -> Config:
    if not CONFIG_PATH.exists():
        return Config()
    raw: dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return Config.model_validate(raw)


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    return Secrets()


def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def filing_text_path(cfg: Config | None = None) -> Path:
    """Where Layer 1 drops the narrative text and Layer 2 reads it from."""
    c = (cfg or get_config()).company
    return RAW_DIR / f"{c.slug}_{c.form.lower().replace('-', '')}.txt"


def statement_csv_path(table: str, cfg: Config | None = None) -> Path:
    """Where Layer 1 drops one XBRL statement CSV."""
    if table not in STATEMENTS:
        raise KeyError(f"unknown statement {table!r}; expected one of {STATEMENTS}")
    c = (cfg or get_config()).company
    return RAW_DIR / f"{c.slug}_{table}.csv"
