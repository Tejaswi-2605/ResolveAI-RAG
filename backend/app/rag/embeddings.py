"""
embeddings.py — TURNING TEXT INTO NUMBERS THAT CARRY MEANING.

WHAT IS AN EMBEDDING?
A sentence is text; a computer cannot compare meanings directly. An embedding
model reads a piece of text and returns a fixed-length list of numbers — a
VECTOR — positioned so that texts with similar meaning land near each other.
"Two people left the company" and "deactivate a departed user's seat" share
almost no words, but a good embedding model puts their vectors close together.
That is the whole reason semantic search can answer a paraphrased question
that keyword search misses.

HOW "NEAR" IS MEASURED: COSINE SIMILARITY.
Cosine similarity is the cosine of the angle between two vectors: 1.0 means
the same direction, 0.0 unrelated, -1.0 opposite. We L2-NORMALISE every vector
(scale it to length 1) before storing it, and after that the dot product IS the
cosine. That matters practically: it lets the vector index use a plain
inner-product search and still be doing exact cosine similarity.

THE MODEL WE USE: sentence-transformers/all-MiniLM-L6-v2
  * 384 dimensions, ~90 MB, runs on CPU in milliseconds
  * downloaded once from Hugging Face, then cached locally — no API key, no
    network at query time, no per-token cost
  * trained on over a billion sentence pairs for exactly this job (semantic
    similarity), so it is a genuinely strong default rather than a toy
It is configurable via EMBEDDING_MODEL. Nothing in the codebase hard-codes it.

THE SECOND PROVIDER: HashingEmbeddings — AND AN HONEST WARNING.
`HashingEmbeddings` maps words and character n-grams into a fixed number of
buckets. It needs no PyTorch, no download and no network, so tests and CI stay
fast and hermetic. But be clear about what it is: it measures VOCABULARY
OVERLAP, not meaning. It cannot tell that "licences" and "seats" are related.
It exists to exercise the plumbing deterministically — never to claim semantic
quality. The evaluation numbers in this repo are produced with the real model.

WHY THE FACTORY RAISES INSTEAD OF SILENTLY SUBSTITUTING:
If sentence-transformers is unavailable, `get_embedding_provider()` raises
`EmbeddingUnavailable`. The retriever catches that and falls back to lexical
search WITH THE FALLBACK RECORDED IN THE TRACE. Quietly swapping in a
non-semantic model would let the system keep reporting "hybrid" while doing
nothing of the sort — the one thing Phase 21 forbids.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import re
from abc import ABC, abstractmethod

import numpy as np

from app.config import Settings, get_settings

logger = logging.getLogger("resolveai.rag.embeddings")


class EmbeddingUnavailable(RuntimeError):
    """The configured embedding provider cannot be used on this machine."""


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    Scale each row to unit length so that a dot product equals cosine similarity.

    Handles a single vector (1-D) and a batch (2-D). A zero vector is left
    alone rather than producing NaN.
    """
    array = np.asarray(vectors, dtype=np.float32)
    single = array.ndim == 1
    matrix = array.reshape(1, -1) if single else array

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized = (matrix / norms).astype(np.float32)

    return normalized[0] if single else normalized


class EmbeddingProvider(ABC):
    """
    The interface the rest of the system codes against.

    Two methods, because the two jobs have different cost profiles: documents
    are embedded once at ingest time in large batches, queries one at a time on
    the hot path. Every implementation must return L2-normalised vectors.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier recorded in the index manifest."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """The exact model this provider loads — pinned into the manifest."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of the vectors produced. Must match the index dimension."""

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed ONE string (a query). Returns shape (dimension,)."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed MANY strings (chunks). Returns shape (len(texts), dimension)."""


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """
    The real semantic model, loaded LAZILY.

    Lazy loading matters: importing sentence-transformers pulls in PyTorch,
    which costs seconds and hundreds of megabytes. A test that only touches
    lexical search should never pay that. The model is loaded on first use and
    then cached on the instance.
    """

    def __init__(self, model_id: str, batch_size: int = 32):
        self._model_id = model_id
        self._batch_size = max(1, batch_size)
        self._model = None

    @property
    def name(self) -> str:
        return "sentence-transformers"

    @property
    def model_id(self) -> str:
        return self._model_id

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - guarded by the factory
                raise EmbeddingUnavailable(
                    f"sentence-transformers is not installed: {exc}") from exc
            logger.info("loading embedding model %s", self._model_id)
            self._model = SentenceTransformer(self._model_id)
        return self._model

    @property
    def dimension(self) -> int:
        model = self._load()
        # sentence-transformers 5.x renamed this; support both so the project
        # works across versions without pinning users to one release.
        getter = (getattr(model, "get_embedding_dimension", None)
                  or model.get_sentence_embedding_dimension)
        return int(getter())

    def embed_text(self, text: str) -> np.ndarray:
        vector = self._load().encode(
            [text], convert_to_numpy=True, show_progress_bar=False, batch_size=1)[0]
        return l2_normalize(vector)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._load().encode(
            texts, convert_to_numpy=True, show_progress_bar=False,
            batch_size=self._batch_size)
        return l2_normalize(vectors)


class HashingEmbeddings(EmbeddingProvider):
    """
    A deterministic, dependency-free stand-in. NOT a semantic model.

    Each token — and each 3-to-5 character slice of it — is hashed with SHA-256
    into one of `dimension` buckets, and the bucket is incremented. Two texts
    sharing vocabulary land in the same buckets and score highly; two texts
    that mean the same thing in different words do not. Character n-grams give
    it a little robustness to suffixes ("export"/"exports"), nothing more.

    Use: unit tests, CI, and proving the fallback ladder works without
    downloading 90 MB of model weights.
    """

    _TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

    def __init__(self, dimension: int = 512):
        self._dimension = max(8, int(dimension))

    @property
    def name(self) -> str:
        return "hashing"

    @property
    def model_id(self) -> str:
        return f"hashing-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _bucket(self, value: str) -> int:
        # SHA-256 rather than Python's hash(): built-in string hashing is
        # randomised per process, which would make the index non-reproducible.
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self._dimension

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for token in self._TOKEN_RE.findall((text or "").lower()):
            vector[self._bucket(token)] += 1.0
            for size in (3, 4, 5):
                for i in range(len(token) - size + 1):
                    vector[self._bucket(token[i:i + size])] += 0.25
        return l2_normalize(vector)

    def embed_text(self, text: str) -> np.ndarray:
        return self._vector(text)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """
    Build the configured embedding provider, or raise `EmbeddingUnavailable`.

    Raising — rather than silently substituting a weaker model — is deliberate.
    The caller (`HybridRetriever`) catches it, drops to lexical-only retrieval
    and RECORDS the fallback in the trace, so the system can never claim
    semantic retrieval happened when it did not.
    """
    settings = settings or get_settings()
    provider = settings.embedding_provider.strip().lower()

    if provider == "hashing":
        return HashingEmbeddings(settings.hashing_embedding_dim)

    if provider == "sentence-transformers":
        # find_spec answers "is it installed?" without paying the import cost.
        if importlib.util.find_spec("sentence_transformers") is None:
            raise EmbeddingUnavailable(
                "sentence-transformers is not installed; set EMBEDDING_PROVIDER=hashing "
                "or `pip install sentence-transformers` to enable semantic retrieval")
        return SentenceTransformerEmbeddings(
            settings.embedding_model, settings.embedding_batch_size)

    raise EmbeddingUnavailable(f"unknown EMBEDDING_PROVIDER '{settings.embedding_provider}'")
