"""
app.rag — the Hybrid RAG engine.

Layering rule for this package (the separation the whole design rests on):

    agent  →  knowledge tool  →  HybridRetriever  →  BM25 / embeddings / FAISS / RRF / rerank

`app.core.agent` must never import anything from below `HybridRetriever`. It
asks the knowledge tool a question and receives evidence dicts; it has no idea
whether the answer came from BM25, a vector index, or a lexical fallback. The
one exception is `app.rag.citations`, which is pure functions over plain dicts
and carries no retrieval machinery with it.

Only the names below are intended for use outside this package.
"""

from app.rag.citations import CitationReport, sources_for, validate_citations
from app.rag.hybrid import (HybridRetriever, get_retriever,
                            reciprocal_rank_fusion, reset_retriever_cache)
from app.rag.models import Chunk, RetrievalTrace, RetrievedChunk

__all__ = [
    "Chunk",
    "RetrievedChunk",
    "RetrievalTrace",
    "HybridRetriever",
    "get_retriever",
    "reset_retriever_cache",
    "reciprocal_rank_fusion",
    "CitationReport",
    "validate_citations",
    "sources_for",
]
