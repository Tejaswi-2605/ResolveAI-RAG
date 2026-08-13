"""
models.py — THE DATA TYPES THE RETRIEVAL PIPELINE PASSES AROUND.

Three types, in the order the pipeline produces them:

    Chunk            a piece of a knowledge-base article, with its metadata
    RetrievedChunk   a Chunk plus WHY it was retrieved (score, method, rank)
    RetrievalTrace   the story of one retrieval call, for observability

Why dataclasses rather than dicts? A dict lets any typo become a silent
`None`. A dataclass names every field once, so the chain

    article  →  chunk  →  retrieved evidence  →  citation

is enforced by the type system instead of by hope. `Chunk` is FROZEN
(immutable) because a chunk is a fact about the corpus. `RetrievedChunk` and
`RetrievalTrace` are mutable because they are ASSEMBLED as the pipeline runs —
the reranker rewrites scores, and each stage appends to the trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The retrieval "arms". Named constants so a typo is an ImportError rather
# than a silently-missing provenance label.
LEXICAL = "lexical"
SEMANTIC = "semantic"


@dataclass(frozen=True)
class Chunk:
    """
    One retrievable unit of knowledge.

    The identity fields (`article_id`, `chunk_id`) are what make citation
    checking possible: a citation is only valid if it names a chunk that was
    actually retrieved, and every chunk knows which article it came from.
    """

    chunk_id: str            # "kb_003#02" — unique, stable, human-readable
    article_id: str          # "kb_003" — the authoritative row in kb_articles
    title: str               # the article title (repeated on every chunk)
    section: str             # the "## Section" heading this text sits under
    text: str                # the chunk body — what gets embedded and shown
    tags: list[str]          # keywords from the article
    url: str | None          # a link a human can follow to verify
    product_area: str | None
    ordinal: int             # 1-based position of this chunk within its article
    source: str              # where the knowledge came from, e.g. "kb_articles"

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form — used by the on-disk chunk store."""
        return {
            "chunk_id": self.chunk_id,
            "article_id": self.article_id,
            "title": self.title,
            "section": self.section,
            "text": self.text,
            "tags": list(self.tags),
            "url": self.url,
            "product_area": self.product_area,
            "ordinal": self.ordinal,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Rebuild a Chunk from `to_dict()` output."""
        return cls(
            chunk_id=data["chunk_id"],
            article_id=data["article_id"],
            title=data["title"],
            section=data.get("section", ""),
            text=data["text"],
            tags=list(data.get("tags", [])),
            url=data.get("url"),
            product_area=data.get("product_area"),
            ordinal=int(data.get("ordinal", 1)),
            source=data.get("source", "kb_articles"),
        )


@dataclass
class RetrievedChunk:
    """
    A Chunk that a retrieval run selected, carrying its PROVENANCE.

    `retrieval_methods` is the honest record of which arm(s) found it —
    ["lexical"], ["semantic"], or both. That list is what lets the UI say
    "BM25 and vector search agreed on this one", and it is why the system can
    call itself hybrid without hand-waving.

    `score` is the score of the LAST stage that ran (fusion, or reranking if a
    reranker is enabled). The per-stage numbers are kept alongside so nothing
    is lost: `method_scores` holds each arm's raw score, `ranks` each arm's
    1-based position, and `fusion_score` the RRF total before reranking.
    """

    chunk: Chunk
    score: float
    retrieval_methods: list[str]
    ranks: dict[str, int] = field(default_factory=dict)          # method -> rank
    method_scores: dict[str, float] = field(default_factory=dict)  # method -> raw score
    fusion_score: float | None = None                             # RRF total, if fused
    rerank_score: float | None = None                             # reranker output, if any

    def to_dict(self) -> dict[str, Any]:
        """
        The shape the knowledge TOOL hands to the agent.

        Deliberately flat and free of FAISS/BM25 internals — the agent sees
        evidence, not an index.
        """
        return {
            "chunk_id": self.chunk.chunk_id,
            "article_id": self.chunk.article_id,
            "title": self.chunk.title,
            "section": self.chunk.section,
            "text": self.chunk.text,
            "url": self.chunk.url,
            "tags": list(self.chunk.tags),
            "score": round(self.score, 6),
            "retrieval_methods": list(self.retrieval_methods),
            "ranks": dict(self.ranks),
            "method_scores": {k: round(v, 6) for k, v in self.method_scores.items()},
            "fusion_score": None if self.fusion_score is None else round(self.fusion_score, 6),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
        }


@dataclass
class RetrievalTrace:
    """
    What actually happened during one `HybridRetriever.retrieve()` call.

    This is the observability record, and its most important field is
    `fallbacks`. If the vector index is missing, the system must NOT quietly
    return lexical results while calling itself hybrid — `mode_used` changes
    and the reason is appended here. An honest trace is the difference between
    a system you can debug and one you can only guess at.

    It stores the query and chunk ids only. No customer PII goes in here.
    """

    query: str
    mode_requested: str                 # what config asked for: hybrid/lexical/semantic
    mode_used: str = ""                 # what actually ran (differs on fallback)
    lexical_candidates: int = 0
    semantic_candidates: int = 0
    fusion_method: str | None = None    # "rrf" when both arms ran, else None
    rrf_k: int = 0
    reranker: str | None = None
    final_k: int = 0
    latency_ms: dict[str, float] = field(default_factory=dict)  # per stage
    fallbacks: list[str] = field(default_factory=list)
    top_chunk_ids: list[str] = field(default_factory=list)

    @property
    def total_latency_ms(self) -> float:
        return round(self.latency_ms.get("total", 0.0), 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode_requested": self.mode_requested,
            "mode_used": self.mode_used,
            "lexical_candidates": self.lexical_candidates,
            "semantic_candidates": self.semantic_candidates,
            "fusion_method": self.fusion_method,
            "rrf_k": self.rrf_k,
            "reranker": self.reranker,
            "final_k": self.final_k,
            "latency_ms": {k: round(v, 3) for k, v in self.latency_ms.items()},
            "fallbacks": list(self.fallbacks),
            "top_chunk_ids": list(self.top_chunk_ids),
        }
