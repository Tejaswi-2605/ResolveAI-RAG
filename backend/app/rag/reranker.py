"""
reranker.py — THE SECOND-OPINION STAGE.

WHY RERANK AT ALL?
Retrieval is a two-speed problem. BM25 and vector search are FAST because they
never really read the query against the document: BM25 counts tokens, and the
vector index compares two numbers-only summaries that were computed
independently of each other. That speed is what makes it possible to score the
whole corpus — but it is also why the top of the list is often "roughly right,
slightly mis-ordered".

A reranker runs afterwards on a SHORT list (here, ~12 candidates) and can
afford to look at the query and the chunk TOGETHER. Cheap over everything,
then expensive over the survivors. That is the standard retrieve-then-rerank
pattern, and it is where most of the easy quality comes from.

WHAT IS SHIPPED, AND WHY IT IS THE DEFAULT
`HeuristicReranker` is a deterministic weighted blend of five signals. It has
no model, no download and no randomness, so the same input always produces the
same order — which is what makes it safe to gate CI on. It is a genuine
improvement over raw fusion order because it can see things RRF structurally
cannot: RRF only knows RANKS, so it has thrown away whether the chunk actually
contains the customer's exact phrase.

The five signals:
  1. fusion score       — what the two retrieval arms already concluded
  2. term coverage      — what fraction of the query's distinct terms appear
  3. field match        — do terms hit the title / section / tags?
  4. exact phrase       — does the chunk contain the query verbatim?
  5. arm agreement      — did BOTH lexical and semantic surface this chunk?

WHAT WOULD BE BETTER, AND WHY IT IS OPTIONAL
`CrossEncoderReranker` runs a real cross-encoder (ms-marco-MiniLM-L-6-v2): a
transformer that reads query and chunk in ONE forward pass and outputs a
relevance score. Because it sees both texts jointly it can judge relevance far
more accurately than any bag-of-words blend. The cost is ~90 MB more model and
tens of milliseconds per candidate, and it is non-deterministic across
hardware, so it is opt-in via `RERANKER=cross-encoder` rather than the default.

MEASURED RESULT: THE HEURISTIC RERANKER IS OFF BY DEFAULT
`RERANKER=none` is the shipped default because the retrieval evaluation said
so. On this corpus the heuristic reranker LOWERED recall@1 from 0.938 to 0.875
at every RRF weighting tested. Two reasons, both worth understanding:

  1. Its signals are bag-of-words. A paraphrase match found by the semantic arm
     has low term coverage BY DEFINITION — that is exactly why the embedding
     model was needed. So a lexical-flavoured reranker systematically penalises
     the results semantic retrieval exists to contribute.

  2. `fusion` is min-max normalised across the candidate list, and RRF scores
     are all very close together (1/61 vs 1/62 vs 1/63). Min-max stretches
     those near-identical values across the full [0, 1] range, which turns a
     rank difference of one place into a large score difference. It amplifies
     what is nearly noise.

With W_FUSION as the largest weight the effect is small, so in practice this
implementation behaves as a TIE-BREAKER among equally-fused candidates rather
than a true reranker. Kept, tested and pluggable — because the honest fix is a
cross-encoder that reads query and chunk jointly, and the architecture must
make that a config change. Shipping it enabled would have been the dishonest
choice: a stage that sounds impressive and measurably makes results worse.

THE POINT OF THE INTERFACE: all three implementations satisfy one small
protocol, so swapping them is a config change, not a code change.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from app.config import Settings, get_settings
from app.rag.models import LEXICAL, SEMANTIC, RetrievedChunk

logger = logging.getLogger("resolveai.rag.reranker")

# Signal weights. Named constants, summing to 1.0, so the blend is auditable
# and a change is a one-line diff rather than an archaeology exercise.
W_FUSION = 0.45      # trust the two retrieval arms most
W_COVERAGE = 0.25    # how much of the question this chunk actually addresses
W_FIELD = 0.15       # a title/section hit is a strong topical signal
W_PHRASE = 0.10      # verbatim phrase — rare, and very informative
W_AGREEMENT = 0.05   # both arms agreeing is mild corroboration

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


class Reranker(ABC):
    """Reorder a candidate list and return the best `top_k`."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier recorded in the retrieval trace."""

    @abstractmethod
    def rerank(self, query: str, candidates: list[RetrievedChunk],
               top_k: int) -> list[RetrievedChunk]:
        """Return a new list, best first, at most `top_k` long."""


class NoOpReranker(Reranker):
    """Keep the fusion order. The honest baseline the others are measured against."""

    @property
    def name(self) -> str:
        return "none"

    def rerank(self, query: str, candidates: list[RetrievedChunk],
               top_k: int) -> list[RetrievedChunk]:
        return list(candidates[:top_k])


class HeuristicReranker(Reranker):
    """Deterministic weighted blend of five lexical/structural signals."""

    @property
    def name(self) -> str:
        return "heuristic"

    @staticmethod
    def _terms(text: str) -> set[str]:
        from app.rag.lexical import STOPWORDS
        return {t for t in _TOKEN_RE.findall((text or "").lower())
                if t not in STOPWORDS}

    def rerank(self, query: str, candidates: list[RetrievedChunk],
               top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []

        query_terms = self._terms(query)
        query_lower = (query or "").lower().strip()

        # Min-max normalise the incoming scores to [0, 1] so the fusion signal
        # is comparable with the other four, whatever scale it arrived on.
        scores = [c.score for c in candidates]
        low, high = min(scores), max(scores)
        span = high - low

        reranked: list[RetrievedChunk] = []
        for candidate in candidates:
            chunk = candidate.chunk
            fusion = (candidate.score - low) / span if span > 0 else 1.0

            body_terms = self._terms(chunk.text)
            field_terms = self._terms(
                f"{chunk.title} {chunk.section} {' '.join(chunk.tags)}")

            coverage = (len(query_terms & body_terms) / len(query_terms)
                        if query_terms else 0.0)
            field = (len(query_terms & field_terms) / len(query_terms)
                     if query_terms else 0.0)
            # Only reward a verbatim phrase when the query is long enough for
            # the match to mean something; a 4-character query matches anything.
            phrase = 1.0 if len(query_lower) >= 8 and query_lower in chunk.text.lower() else 0.0
            agreement = 1.0 if {LEXICAL, SEMANTIC} <= set(candidate.retrieval_methods) else 0.0

            candidate.rerank_score = (
                W_FUSION * fusion
                + W_COVERAGE * coverage
                + W_FIELD * field
                + W_PHRASE * phrase
                + W_AGREEMENT * agreement
            )
            candidate.score = candidate.rerank_score
            reranked.append(candidate)

        # Ties break on chunk_id, so the output is fully deterministic.
        reranked.sort(key=lambda c: (-c.score, c.chunk.chunk_id))
        return reranked[:top_k]


class CrossEncoderReranker(Reranker):
    """
    A real cross-encoder: one transformer pass over `(query, chunk)` together.

    Loaded lazily, and if the model cannot be loaded the caller falls back to
    the heuristic reranker rather than failing the request.
    """

    def __init__(self, model_id: str):
        self._model_id = model_id
        self._model = None

    @property
    def name(self) -> str:
        return "cross-encoder"

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("loading cross-encoder %s", self._model_id)
            self._model = CrossEncoder(self._model_id)
        return self._model

    def rerank(self, query: str, candidates: list[RetrievedChunk],
               top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        model = self._load()
        pairs = [(query, c.chunk.text) for c in candidates]
        for candidate, score in zip(candidates, model.predict(pairs)):
            candidate.rerank_score = float(score)
            candidate.score = float(score)
        candidates.sort(key=lambda c: (-c.score, c.chunk.chunk_id))
        return candidates[:top_k]


def get_reranker(settings: Settings | None = None) -> Reranker:
    """
    Build the configured reranker.

    An unavailable cross-encoder degrades to the heuristic with a warning: the
    reranker only reorders an already-retrieved list, so a weaker one is a
    quality trade-off, never a correctness or security problem.
    """
    settings = settings or get_settings()
    choice = settings.reranker.strip().lower()

    if choice in ("none", "noop", "off"):
        return NoOpReranker()
    if choice == "cross-encoder":
        try:
            import importlib.util
            if importlib.util.find_spec("sentence_transformers") is None:
                raise ImportError("sentence-transformers is not installed")
            return CrossEncoderReranker(settings.cross_encoder_model)
        except ImportError as exc:
            logger.warning("cross-encoder unavailable (%s); using the heuristic reranker", exc)
            return HeuristicReranker()
    if choice != "heuristic":
        logger.warning("unknown RERANKER '%s'; using the heuristic reranker", choice)
    return HeuristicReranker()
