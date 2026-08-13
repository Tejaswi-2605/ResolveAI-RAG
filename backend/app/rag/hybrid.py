"""
hybrid.py — THE RETRIEVAL ORCHESTRATOR. This is the heart of the RAG system.

THE PIPELINE

        query
          │
    ┌─────┴─────┐
    ▼           ▼
  BM25       embed → vector index          (two independent arms)
    │           │
    └─────┬─────┘
          ▼
    RRF fusion                              (merge two rankings into one)
          ▼
    reranker                                (second opinion on the short list)
          ▼
    top-K evidence + RetrievalTrace

WHY TWO ARMS — THE ONE-PARAGRAPH ARGUMENT
Lexical search fails when the customer uses different words from the docs
("licences" vs "seats"). Semantic search fails on rare exact strings, because
an embedding compresses text into a few hundred numbers and ERR-4029 and
ERR-3007 compress to almost the same place. Their failure modes are
UNCORRELATED, which is precisely the condition under which combining two
retrievers beats tuning either one. A system that only embeds is not hybrid,
whatever it calls itself.

WHY RRF RATHER THAN ADDING THE SCORES
The obvious idea — `0.5 * bm25 + 0.5 * cosine` — does not work, because the two
numbers are not comparable. BM25 is unbounded and corpus-dependent (a score of
14 means nothing on its own); cosine sits in [-1, 1]. Any fixed weighting is
really a hidden bet on the score distributions, and it breaks when the corpus
changes.

Reciprocal Rank Fusion sidesteps this by throwing the scores away and using
only RANK — the one thing both arms agree on the meaning of:

        RRF(d) = Σ  1 / (k + rank_of_d_in_list_i)
                 i

  * A chunk ranked #1 by both arms scores 2/(k+1) — the maximum.
  * A chunk ranked #1 by one arm and absent from the other still scores
    1/(k+1), so a strong single-arm result is not thrown away.
  * `k` (default 60, the value from the original TREC paper) damps the
    difference between the top ranks. Without it, rank 1 would be worth
    infinitely more than rank 2; with it, 1/61 vs 1/62 — close, so agreement
    across arms matters more than a one-place lead within one arm.

RRF is parameter-light, needs no training data and no score calibration, and
is what most production hybrid systems actually use.

FALLBACKS ARE VISIBLE, NEVER SILENT
If the vector index is missing or the embedding model will not load, retrieval
continues on BM25 alone — but `mode_used` becomes "lexical" and the reason is
appended to `trace.fallbacks`. The system never reports "hybrid" for work it
did not do.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from app.config import Settings, get_settings
from app.database import db
from app.rag.chunking import chunk_articles
from app.rag.embeddings import (EmbeddingProvider, EmbeddingUnavailable,
                                get_embedding_provider)
from app.rag.lexical import BM25Index
from app.rag.models import (LEXICAL, SEMANTIC, Chunk, RetrievalTrace,
                            RetrievedChunk)
from app.rag.reranker import get_reranker
from app.rag.vector_store import (VectorIndex, VectorIndexUnavailable,
                                  load_index)

logger = logging.getLogger("resolveai.rag.hybrid")

CHUNKS_FILE = "chunks.json"

VALID_MODES = ("hybrid", "lexical", "semantic")


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[tuple[str, float]]],
    rrf_k: int,
    chunk_map: dict[str, Chunk],
    weights: dict[str, float] | None = None,
) -> list[RetrievedChunk]:
    """
    Merge several ranked lists into one using Reciprocal Rank Fusion.

    `ranked_lists` maps a method name ("lexical", "semantic") to that method's
    results, ALREADY ordered best-first. Position in the list is the rank, so
    the raw scores are never compared across methods — that is the entire point.

        RRF(d) = Σ  weight_i / (k + rank_of_d_in_list_i)
                 i

    WEIGHTS default to 1.0, which is textbook RRF and treats both arms as
    equally trustworthy. That assumption is not always right: if one arm is
    measurably stronger on your corpus, an equal vote lets the weaker arm pull
    a bad result to the top. The weights are configurable so that decision can
    be made from evaluation numbers rather than by feel — see
    `evaluation/retrieval_eval.py` and the tuning note in `docs/hybrid_rag.md`.

    Duplicates are merged by `chunk_id`: one output entry per chunk, whose
    `retrieval_methods` is the union of the arms that found it and whose
    `ranks` / `method_scores` keep each arm's own numbers for the trace.

    Deterministic: equal scores break on chunk_id.
    """
    weights = weights or {}
    totals: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    method_scores: dict[str, dict[str, float]] = {}
    methods: dict[str, list[str]] = {}

    for method in sorted(ranked_lists):          # sorted → stable method order
        weight = weights.get(method, 1.0)
        for position, (chunk_id, score) in enumerate(ranked_lists[method]):
            rank = position + 1                  # RRF ranks are 1-based
            totals[chunk_id] = totals.get(chunk_id, 0.0) + weight / (rrf_k + rank)
            ranks.setdefault(chunk_id, {})[method] = rank
            method_scores.setdefault(chunk_id, {})[method] = float(score)
            methods.setdefault(chunk_id, [])
            if method not in methods[chunk_id]:
                methods[chunk_id].append(method)

    fused = [
        RetrievedChunk(
            chunk=chunk_map[chunk_id],
            score=total,
            retrieval_methods=methods[chunk_id],
            ranks=ranks[chunk_id],
            method_scores=method_scores[chunk_id],
            fusion_score=total,
        )
        for chunk_id, total in totals.items()
        if chunk_id in chunk_map          # guards against a stale index
    ]
    fused.sort(key=lambda c: (-c.score, c.chunk.chunk_id))
    return fused


def _single_arm_results(method: str, results: list[tuple[str, float]],
                        chunk_map: dict[str, Chunk]) -> list[RetrievedChunk]:
    """Wrap one arm's output when there is nothing to fuse it with."""
    return [
        RetrievedChunk(
            chunk=chunk_map[chunk_id],
            score=float(score),
            retrieval_methods=[method],
            ranks={method: position + 1},
            method_scores={method: float(score)},
        )
        for position, (chunk_id, score) in enumerate(results)
        if chunk_id in chunk_map
    ]


def load_chunk_store(index_dir: Path) -> list[Chunk]:
    """Read `chunks.json`. Raises FileNotFoundError when ingest has never run."""
    path = Path(index_dir) / CHUNKS_FILE
    if not path.exists():
        raise FileNotFoundError(f"chunk store not found at {path}")
    return [Chunk.from_dict(item)
            for item in json.loads(path.read_text(encoding="utf-8"))]


def chunk_store_from_database(settings: Settings) -> list[Chunk]:
    """
    Rebuild the chunk store in memory straight from SQLite.

    The last rung of the fallback ladder: if the derived artifacts are missing
    entirely, the authoritative knowledge is still in `kb_articles`, so lexical
    search can be reconstructed on the spot. Slower and semantic-free, but the
    application keeps answering.
    """
    return chunk_articles(db.all_kb_articles(),
                          settings.chunk_size_words,
                          settings.chunk_overlap_words)


class HybridRetriever:
    """
    Owns the retrieval pipeline and the trace it produces.

    Everything above this class — the knowledge tool, the agent, the API —
    sees only `retrieve()` returning evidence plus a trace. None of them
    imports faiss, BM25, the embedding model or the reranker. That boundary is
    what lets the retrieval strategy change without touching agent code.

    Construction NEVER raises for a missing artifact. Each resource is probed
    and its failure recorded, so a request always gets the best retrieval the
    machine can currently support.
    """

    def __init__(self, settings: Settings | None = None,
                 chunks: list[Chunk] | None = None):
        self.settings = settings or get_settings()
        self.startup_fallbacks: list[str] = []

        # 1. the chunk store ------------------------------------------------
        if chunks is not None:
            self.chunks = list(chunks)
        else:
            try:
                self.chunks = load_chunk_store(self.settings.index_dir)
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
                logger.warning("chunk store unavailable (%s); chunking from SQLite", exc)
                self.startup_fallbacks.append("chunk_store_rebuilt_from_database")
                self.chunks = chunk_store_from_database(self.settings)

        self.chunk_map: dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}

        # 2. the lexical arm — always available, built in memory -------------
        self.bm25 = BM25Index(self.chunks)

        # 3. the semantic arm — may be unavailable --------------------------
        self.embedder: EmbeddingProvider | None = None
        self.vector_index: VectorIndex | None = None
        self._probe_semantic_arm()

    def _probe_semantic_arm(self) -> None:
        """Try to load the embedding model and vector index; record any failure."""
        try:
            self.embedder = get_embedding_provider(self.settings)
        except EmbeddingUnavailable as exc:
            logger.warning("semantic retrieval disabled: %s", exc)
            self.startup_fallbacks.append("embedding_provider_unavailable")
            return

        try:
            self.vector_index = load_index(self.settings.index_dir,
                                           self.settings.vector_backend)
        except VectorIndexUnavailable as exc:
            logger.warning("semantic retrieval disabled: %s", exc)
            self.startup_fallbacks.append("vector_index_unavailable")
            self.vector_index = None
            return

        # A stale index — built before the knowledge base changed — would map
        # vectors to chunk ids that no longer exist. Detect it now, loudly.
        known = set(self.chunk_map)
        unknown = [cid for cid in self.vector_index.chunk_ids if cid not in known]
        if unknown:
            logger.warning("vector index is stale: %d chunk ids are unknown "
                           "(rebuild with `python -m app.rag.ingest --rebuild`)",
                           len(unknown))
            self.startup_fallbacks.append("vector_index_stale")

    @property
    def semantic_available(self) -> bool:
        return self.embedder is not None and self.vector_index is not None

    def retrieve(self, query: str, top_k: int | None = None,
                 mode: str | None = None,
                 use_reranker: bool | None = None
                 ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        """
        Run retrieval for one query.

        `mode` and `use_reranker` default to configuration; the evaluation
        harness overrides them per run so it can compare lexical / semantic /
        hybrid / hybrid+rerank against the identical corpus and code path.

        Returns `(evidence, trace)`. The trace is the honest record of what ran.
        """
        settings = self.settings
        top_k = top_k or settings.top_k_final
        requested = (mode or settings.rag_mode).strip().lower()
        if requested not in VALID_MODES:
            logger.warning("unknown RAG_MODE '%s'; using hybrid", requested)
            requested = "hybrid"

        started = time.perf_counter()
        trace = RetrievalTrace(query=query, mode_requested=requested,
                               rrf_k=settings.rrf_k)
        trace.fallbacks.extend(self.startup_fallbacks)

        ranked_lists: dict[str, list[tuple[str, float]]] = {}

        # ── decide which arms to run ───────────────────────────────────
        # BM25 is the fallback arm: it needs no model and no derived index, so
        # it can always run. Whenever the semantic arm is unavailable or fails,
        # lexical is switched on even if the caller asked for semantic only —
        # a degraded answer beats no answer, provided the trace says so.
        run_lexical = requested in ("hybrid", "lexical")
        run_semantic = requested in ("hybrid", "semantic")

        if run_semantic and not self.semantic_available:
            trace.fallbacks.append("semantic_unavailable")
            run_semantic = False
            run_lexical = True

        # ── semantic arm ───────────────────────────────────────────────
        if run_semantic:
            stage = time.perf_counter()
            try:
                vector = self.embedder.embed_text(query)
                raw = self.vector_index.search(vector, settings.top_k_semantic)
                # Cosine is bounded in [-1, 1], so a floor here is a meaningful
                # "this is not actually about the query" filter — unlike BM25
                # or RRF scores, which have no absolute meaning.
                semantic = [(cid, score) for cid, score in raw
                            if score >= settings.min_semantic_score]
                trace.semantic_candidates = len(semantic)
                if semantic:
                    ranked_lists[SEMANTIC] = semantic
            except Exception as exc:               # model or index failed mid-query
                logger.warning("semantic retrieval failed: %s", exc)
                trace.fallbacks.append("semantic_search_failed")
                run_lexical = True
            finally:
                trace.latency_ms["semantic"] = (time.perf_counter() - stage) * 1000

        # ── lexical arm ────────────────────────────────────────────────
        if run_lexical:
            stage = time.perf_counter()
            lexical = self.bm25.search(query, settings.top_k_lexical)
            trace.latency_ms["lexical"] = (time.perf_counter() - stage) * 1000
            trace.lexical_candidates = len(lexical)
            if lexical:
                ranked_lists[LEXICAL] = lexical

        # ── fusion ─────────────────────────────────────────────────────
        stage = time.perf_counter()
        if len(ranked_lists) > 1:
            candidates = reciprocal_rank_fusion(
                ranked_lists, settings.rrf_k, self.chunk_map,
                weights={LEXICAL: settings.rrf_weight_lexical,
                         SEMANTIC: settings.rrf_weight_semantic})
            trace.fusion_method = "rrf"
        elif ranked_lists:
            method, results = next(iter(ranked_lists.items()))
            candidates = _single_arm_results(method, results, self.chunk_map)
            trace.fusion_method = None
        else:
            candidates = []
            trace.fusion_method = None
        trace.latency_ms["fusion"] = (time.perf_counter() - stage) * 1000

        # `mode_used` is derived from what ACTUALLY produced candidates, not
        # from what was asked for. This is the anti-lying property.
        methods_used = sorted(ranked_lists)
        if len(methods_used) > 1:
            trace.mode_used = "hybrid"
        elif methods_used:
            trace.mode_used = methods_used[0]
        else:
            trace.mode_used = "none"

        # ── reranking ──────────────────────────────────────────────────
        rerank_enabled = (settings.reranker.strip().lower() != "none"
                          if use_reranker is None else use_reranker)
        if candidates and rerank_enabled:
            stage = time.perf_counter()
            reranker = get_reranker(settings)
            shortlist = candidates[:max(settings.rerank_candidates, top_k)]
            candidates = reranker.rerank(query, shortlist, top_k)
            trace.reranker = reranker.name
            trace.latency_ms["rerank"] = (time.perf_counter() - stage) * 1000
        else:
            candidates = candidates[:top_k]
            trace.reranker = None

        trace.final_k = len(candidates)
        trace.top_chunk_ids = [c.chunk.chunk_id for c in candidates]
        trace.latency_ms["total"] = (time.perf_counter() - started) * 1000
        return candidates, trace


# ── process-wide cache ────────────────────────────────────────────────
# Building BM25 and loading FAISS costs real time; doing it per tool call
# would dominate latency. The cache key is every setting that changes what the
# retriever IS, so a test that repoints DATABASE_PATH or the index directory
# transparently gets a fresh instance instead of a stale one.
_CACHE: dict[tuple, HybridRetriever] = {}


def _cache_key(settings: Settings) -> tuple:
    return (
        str(settings.index_dir), str(settings.db_file),
        settings.embedding_provider, settings.embedding_model,
        settings.hashing_embedding_dim, settings.vector_backend,
        settings.chunk_size_words, settings.chunk_overlap_words,
    )


def get_retriever(settings: Settings | None = None) -> HybridRetriever:
    """Return the shared retriever for the current configuration."""
    settings = settings or get_settings()
    key = _cache_key(settings)
    if key not in _CACHE:
        _CACHE[key] = HybridRetriever(settings)
    return _CACHE[key]


def reset_retriever_cache() -> None:
    """Drop every cached retriever. Called by tests and after re-ingestion."""
    _CACHE.clear()
