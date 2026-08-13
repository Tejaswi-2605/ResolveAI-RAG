"""
vector_store.py — WHERE THE EMBEDDINGS LIVE AND HOW WE SEARCH THEM.

THE PROBLEM
Once every chunk is a vector, answering a query means finding the vectors
closest to the query's vector. Doing that with a Python loop over a million
chunks is far too slow. A VECTOR INDEX is a data structure built for exactly
this one operation.

WHAT FAISS IS
FAISS (Facebook AI Similarity Search) is a C++ library with Python bindings for
storing vectors and finding nearest neighbours fast. It runs in-process — no
server, no network, no account. `pip install faiss-cpu` and you have it.

WHY `IndexFlatIP` SPECIFICALLY
FAISS offers many index types. "Flat" means brute force: compare the query
against every stored vector. "IP" means inner product. Two consequences:

  * EXACT, not approximate. Approximate indexes (IVF, HNSW) trade recall for
    speed and only start paying off in the millions of vectors. At our scale
    exactness is free, and an evaluation is worth much more when the retriever
    has no randomness in it.
  * Because `embeddings.py` L2-normalises every vector, inner product IS
    cosine similarity. We get cosine ranking without an extra normalisation
    step at query time.

WHY A NUMPY FALLBACK EXISTS
`NumpyVectorIndex` does the same brute-force search with one matrix multiply.
It is a handful of lines and depends only on numpy, so the system still works
if faiss will not install (it can be awkward on some platforms) — and it makes
"FAISS is an optimisation, not a magic ingredient" something you can verify by
running the tests both ways. The two backends return identical rankings.

WHY NOT PINECONE / WEAVIATE / A CLOUD VECTOR DB?
They solve problems this project does not have: distributed scale, multi-tenant
serving, managed uptime. They add an account, a network hop, a bill and an
integration that cannot be run offline. Local and reproducible wins here.

WHAT IS ON DISK (all of it DERIVED and git-ignored)
    data/index/chunks.json    the chunk store — the text and metadata
    data/index/ids.json       chunk ids, in the same order as the vector rows
    data/index/vectors.npy    the raw normalised vectors
    data/index/index.faiss    the FAISS index (faiss backend only)
    data/index/manifest.json  model, dimension, counts, corpus fingerprint

`ids.json` is stored explicitly rather than inferred from `chunks.json`, so a
mismatch between the index and the chunk store is DETECTABLE instead of
silently returning the wrong chunk for a vector row.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

logger = logging.getLogger("resolveai.rag.vector_store")

IDS_FILE = "ids.json"
VECTORS_FILE = "vectors.npy"
FAISS_FILE = "index.faiss"


class VectorIndexUnavailable(RuntimeError):
    """No usable vector index could be loaded (missing, stale or corrupt)."""


class VectorIndex(ABC):
    """
    The interface `HybridRetriever` codes against.

    Note what it does NOT expose: no FAISS handles, no numpy arrays, no index
    types. Callers pass a query vector and receive `(chunk_id, score)` pairs.
    That is the seam which lets the backend be swapped without touching the
    retriever, the knowledge tool, or the agent.
    """

    def __init__(self, dimension: int):
        self.dimension = int(dimension)
        self.chunk_ids: list[str] = []

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    @abstractmethod
    def build(self, vectors: np.ndarray, chunk_ids: list[str]) -> None:
        """Populate the index from normalised vectors and their chunk ids."""

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return the k nearest `(chunk_id, cosine_similarity)`, best first."""

    @abstractmethod
    def save(self, index_dir: Path) -> None:
        """Write the index to disk so it survives a restart."""

    @property
    @abstractmethod
    def backend(self) -> str:
        """Name recorded in the manifest, e.g. "faiss"."""

    # -- shared helpers -------------------------------------------------
    def _validate(self, vectors: np.ndarray, chunk_ids: list[str]) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {array.shape}")
        if array.shape[1] != self.dimension:
            raise ValueError(
                f"vector dimension {array.shape[1]} != index dimension {self.dimension}")
        if array.shape[0] != len(chunk_ids):
            raise ValueError(
                f"{array.shape[0]} vectors but {len(chunk_ids)} chunk ids")
        return array

    def _save_shared(self, index_dir: Path, vectors: np.ndarray) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / VECTORS_FILE, vectors)
        (index_dir / IDS_FILE).write_text(
            json.dumps(self.chunk_ids, indent=2), encoding="utf-8")


class NumpyVectorIndex(VectorIndex):
    """
    Brute-force cosine search with one matrix multiply. Exact, ~20 lines.

    `vectors @ query` gives every similarity at once; `argsort` orders them.
    This is what FAISS's IndexFlatIP does, minus the C++ and the SIMD.
    """

    def __init__(self, dimension: int):
        super().__init__(dimension)
        self.vectors: np.ndarray | None = None

    @property
    def backend(self) -> str:
        return "numpy"

    def build(self, vectors: np.ndarray, chunk_ids: list[str]) -> None:
        self.vectors = self._validate(vectors, chunk_ids)
        self.chunk_ids = list(chunk_ids)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self.vectors is None or not self.size:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        similarities = self.vectors @ query
        # argsort ascending, so take the tail and reverse for descending order.
        top = np.argsort(similarities)[::-1][:max(0, k)]
        return [(self.chunk_ids[i], float(similarities[i])) for i in top]

    def save(self, index_dir: Path) -> None:
        if self.vectors is None:
            raise RuntimeError("cannot save an index that was never built")
        self._save_shared(index_dir, self.vectors)

    @classmethod
    def load(cls, index_dir: Path) -> "NumpyVectorIndex":
        vectors_path = index_dir / VECTORS_FILE
        ids_path = index_dir / IDS_FILE
        if not vectors_path.exists() or not ids_path.exists():
            raise VectorIndexUnavailable(f"no vector artifacts under {index_dir}")
        vectors = np.load(vectors_path).astype(np.float32)
        chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if vectors.shape[0] != len(chunk_ids):
            raise VectorIndexUnavailable(
                f"index/id mismatch: {vectors.shape[0]} vectors, {len(chunk_ids)} ids")
        index = cls(int(vectors.shape[1]))
        index.vectors = vectors
        index.chunk_ids = list(chunk_ids)
        return index


class FaissVectorIndex(VectorIndex):
    """
    Exact cosine search via `faiss.IndexFlatIP` over L2-normalised vectors.

    faiss is imported lazily so that a machine without it can still run
    everything else (the numpy backend produces identical rankings).
    """

    def __init__(self, dimension: int):
        super().__init__(dimension)
        self._faiss = self._import_faiss()
        self.index = None
        self._vectors: np.ndarray | None = None

    @staticmethod
    def _import_faiss():
        try:
            import faiss
        except ImportError as exc:
            raise VectorIndexUnavailable(f"faiss is not installed: {exc}") from exc
        return faiss

    @property
    def backend(self) -> str:
        return "faiss"

    def build(self, vectors: np.ndarray, chunk_ids: list[str]) -> None:
        array = self._validate(vectors, chunk_ids)
        self.index = self._faiss.IndexFlatIP(self.dimension)
        self.index.add(array)
        self.chunk_ids = list(chunk_ids)
        self._vectors = array

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self.index is None or not self.size:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        # FAISS pads with -1 when k exceeds the number of stored vectors.
        scores, positions = self.index.search(query, min(max(0, k), self.size))
        return [(self.chunk_ids[i], float(s))
                for i, s in zip(positions[0], scores[0]) if 0 <= i < self.size]

    def save(self, index_dir: Path) -> None:
        if self.index is None or self._vectors is None:
            raise RuntimeError("cannot save an index that was never built")
        self._save_shared(index_dir, self._vectors)
        self._faiss.write_index(self.index, str(index_dir / FAISS_FILE))

    @classmethod
    def load(cls, index_dir: Path) -> "FaissVectorIndex":
        faiss_path = index_dir / FAISS_FILE
        ids_path = index_dir / IDS_FILE
        if not faiss_path.exists() or not ids_path.exists():
            raise VectorIndexUnavailable(f"no faiss artifacts under {index_dir}")
        faiss = cls._import_faiss()
        raw = faiss.read_index(str(faiss_path))
        chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if raw.ntotal != len(chunk_ids):
            raise VectorIndexUnavailable(
                f"index/id mismatch: {raw.ntotal} vectors, {len(chunk_ids)} ids")
        index = cls(int(raw.d))
        index.index = raw
        index.chunk_ids = list(chunk_ids)
        return index


def create_index(backend: str, dimension: int) -> VectorIndex:
    """
    Build an empty index of the requested backend.

    Falls back to numpy with a logged warning if faiss is unavailable — safe
    here because the two backends produce identical results, so unlike the
    embedding fallback this one does not change what the system can do.
    """
    if backend.strip().lower() == "faiss":
        try:
            return FaissVectorIndex(dimension)
        except VectorIndexUnavailable as exc:
            logger.warning("faiss unavailable (%s); using the numpy backend", exc)
    return NumpyVectorIndex(dimension)


def load_index(index_dir: Path, backend: str) -> VectorIndex:
    """
    Load a saved index, or raise `VectorIndexUnavailable`.

    The retriever catches that exception and drops to lexical-only retrieval
    with the fallback recorded in the trace (Phase 21).
    """
    index_dir = Path(index_dir)
    if backend.strip().lower() == "faiss":
        try:
            return FaissVectorIndex.load(index_dir)
        except VectorIndexUnavailable as exc:
            logger.warning("faiss index unusable (%s); trying the numpy backend", exc)
    return NumpyVectorIndex.load(index_dir)
