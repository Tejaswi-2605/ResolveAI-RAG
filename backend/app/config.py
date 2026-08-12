"""
config.py — THE SINGLE SOURCE OF TRUTH FOR EVERY TUNABLE VALUE.

Why this file exists
--------------------
A retrieval system has a lot of knobs: chunk size, overlap, how many lexical
candidates, how many semantic candidates, the RRF constant, the final top-K,
which embedding model, where the index lives. If those numbers are sprinkled
through the code as literals ("magic numbers"), you cannot answer the two
questions that matter in production:

    "What settings produced this result?"   and   "How do I change one?"

So every knob is declared exactly once, here, with a documented default and
an environment-variable override.

Design notes
------------
* `Settings` is a FROZEN dataclass — once built it cannot be mutated, so no
  part of the system can secretly change configuration at runtime.
* `get_settings()` reads the environment on EVERY call rather than caching.
  Reading ~25 environment variables costs microseconds, and it means a test
  can `monkeypatch.setenv(...)` and immediately see the new value without any
  cache-invalidation dance. (The expensive objects — the embedding model and
  the vector index — are cached separately, where caching actually pays.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# The repository's backend/ directory — used to resolve default relative paths
# so the app behaves the same regardless of the current working directory.
BACKEND_DIR = Path(__file__).resolve().parent.parent


def _env_str(key: str, default: str) -> str:
    value = os.environ.get(key)
    return value if value not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    """Read an int from the environment, falling back to `default` if unset or junk."""
    raw = os.environ.get(key)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _resolve(path_str: str) -> Path:
    """Turn a possibly-relative configured path into an absolute one."""
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (BACKEND_DIR / p).resolve()


@dataclass(frozen=True)
class Settings:
    """Every tunable value in ResolveAI-RAG, resolved from the environment."""

    # ── database ──────────────────────────────────────────────────
    database_path: str

    # ── model provider (the LLM) ──────────────────────────────────
    model_provider: str          # "mock" (offline, deterministic) or "anthropic"
    anthropic_model: str
    mock_failure_mode: str       # forces a failure path; used by tests only

    # ── agent bounds ──────────────────────────────────────────────
    agent_max_steps: int         # hard cap on loop iterations
    agent_timeout_s: float       # per-model-call timeout
    agent_max_retries: int       # retries for TRANSIENT provider errors only

    # ── retrieval: master switches ────────────────────────────────
    rag_enabled: bool
    rag_mode: str                # "hybrid" | "lexical" | "semantic"

    # ── retrieval: chunking ───────────────────────────────────────
    chunk_size_words: int
    chunk_overlap_words: int

    # ── retrieval: embeddings ─────────────────────────────────────
    embedding_provider: str      # "sentence-transformers" | "hashing"
    embedding_model: str         # model id, e.g. "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int
    hashing_embedding_dim: int   # dimension for the dependency-free fallback

    # ── retrieval: vector index ───────────────────────────────────
    vector_backend: str          # "faiss" | "numpy"
    vector_index_path: str       # directory holding the derived index artifacts

    # ── retrieval: candidate counts and fusion ────────────────────
    top_k_lexical: int           # candidates BM25 returns into the fusion stage
    top_k_semantic: int          # candidates vector search returns into fusion
    top_k_final: int             # evidence chunks actually shown to the LLM
    rrf_k: int                   # the RRF smoothing constant
    # Per-arm RRF vote weights. 1.0/1.0 is textbook RRF (both arms trusted
    # equally). The shipped defaults were chosen from the retrieval evaluation,
    # not by feel — see docs/hybrid_rag.md.
    rrf_weight_lexical: float
    rrf_weight_semantic: float

    # ── retrieval: reranking ──────────────────────────────────────
    reranker: str                # "heuristic" | "cross-encoder" | "none"
    cross_encoder_model: str
    rerank_candidates: int       # how many fused candidates enter the reranker

    # ── grounding ─────────────────────────────────────────────────
    # Cosine similarity floor for a SEMANTIC candidate. Cosine is bounded in
    # [-1, 1], so a fixed threshold here is meaningful — unlike BM25 or RRF
    # scores, which are unbounded and corpus-dependent and therefore must
    # never be compared against a hard-coded constant.
    min_semantic_score: float
    require_citations: bool      # strip uncited factual replies and escalate

    # ── observability ─────────────────────────────────────────────
    log_level: str
    cors_origins: str

    # ── derived helpers ───────────────────────────────────────────
    @property
    def index_dir(self) -> Path:
        """Absolute path to the directory holding derived retrieval artifacts."""
        return _resolve(self.vector_index_path)

    @property
    def chunks_path(self) -> Path:
        """The chunk store: the authoritative list of chunks the index was built from."""
        return self.index_dir / "chunks.json"

    @property
    def manifest_path(self) -> Path:
        """Build metadata: model, dimension, corpus fingerprint, timestamp."""
        return self.index_dir / "manifest.json"

    @property
    def faiss_path(self) -> Path:
        return self.index_dir / "index.faiss"

    @property
    def vectors_path(self) -> Path:
        """Raw vectors (numpy backend, and a portable copy for the faiss backend)."""
        return self.index_dir / "vectors.npy"

    @property
    def db_file(self) -> Path:
        return _resolve(self.database_path)


def get_settings() -> Settings:
    """
    Build a Settings object from the current environment.

    Called fresh each time on purpose — see the module docstring. Any default
    changed here changes the documented default everywhere in the project.
    """
    return Settings(
        # database
        database_path=_env_str("DATABASE_PATH", "./resolveai.db"),

        # provider
        model_provider=_env_str("MODEL_PROVIDER", "mock").lower(),
        anthropic_model=_env_str("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        mock_failure_mode=_env_str("MOCK_FAILURE_MODE", ""),

        # agent bounds
        agent_max_steps=_env_int("AGENT_MAX_STEPS", 6),
        agent_timeout_s=_env_float("AGENT_TIMEOUT_S", 30.0),
        agent_max_retries=_env_int("AGENT_MAX_RETRIES", 2),

        # retrieval switches
        rag_enabled=_env_bool("RAG_ENABLED", True),
        rag_mode=_env_str("RAG_MODE", "hybrid").lower(),

        # chunking
        chunk_size_words=_env_int("CHUNK_SIZE_WORDS", 110),
        chunk_overlap_words=_env_int("CHUNK_OVERLAP_WORDS", 25),

        # embeddings
        embedding_provider=_env_str("EMBEDDING_PROVIDER", "sentence-transformers"),
        embedding_model=_env_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 32),
        hashing_embedding_dim=_env_int("HASHING_EMBEDDING_DIM", 512),

        # vector index
        vector_backend=_env_str("VECTOR_BACKEND", "faiss").lower(),
        vector_index_path=_env_str("VECTOR_INDEX_PATH", "./data/index"),

        # candidates & fusion
        top_k_lexical=_env_int("TOP_K_LEXICAL", 10),
        top_k_semantic=_env_int("TOP_K_SEMANTIC", 10),
        top_k_final=_env_int("TOP_K_FINAL", 4),
        rrf_k=_env_int("RRF_K", 60),
        # 0.8 / 1.0 chosen from the retrieval evaluation, not by feel: equal
        # weights scored recall@1 0.875, a 0.8 lexical weight scored 0.938 while
        # keeping exact-identifier recall@1 at 1.00. See docs/hybrid_rag.md.
        rrf_weight_lexical=_env_float("RRF_WEIGHT_LEXICAL", 0.8),
        rrf_weight_semantic=_env_float("RRF_WEIGHT_SEMANTIC", 1.0),

        # reranking
        # OFF by default, and that is a measured decision rather than an
        # oversight. On this corpus the heuristic reranker LOWERED recall@1 at
        # every RRF weighting tested (0.938 → 0.875), because its bag-of-words
        # signals systematically penalise the paraphrase matches the semantic
        # arm exists to find. The stage is implemented, tested and pluggable —
        # set RERANKER=heuristic or cross-encoder to re-enable and re-measure.
        reranker=_env_str("RERANKER", "none").lower(),
        cross_encoder_model=_env_str("CROSS_ENCODER_MODEL",
                                     "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        rerank_candidates=_env_int("RERANK_CANDIDATES", 12),

        # grounding
        min_semantic_score=_env_float("MIN_SEMANTIC_SCORE", 0.15),
        require_citations=_env_bool("REQUIRE_CITATIONS", True),

        # observability
        log_level=_env_str("LOG_LEVEL", "INFO"),
        cors_origins=_env_str("CORS_ORIGINS", "http://localhost:5173"),
    )
