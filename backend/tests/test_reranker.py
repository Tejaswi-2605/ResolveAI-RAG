"""Tests for the reranking stage."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.rag.models import LEXICAL, SEMANTIC, Chunk, RetrievedChunk
from app.rag.reranker import (HeuristicReranker, NoOpReranker, get_reranker)


def _retrieved(chunk_id: str, text: str, score: float, methods: list[str],
               title: str = "Doc", tags: list[str] | None = None) -> RetrievedChunk:
    chunk = Chunk(chunk_id=chunk_id, article_id=chunk_id.split("#")[0], title=title,
                  section="Section", text=text, tags=tags or [], url=None,
                  product_area=None, ordinal=1, source="kb_articles")
    return RetrievedChunk(chunk=chunk, score=score, retrieval_methods=methods,
                          fusion_score=score)


def test_noop_preserves_the_incoming_order():
    candidates = [_retrieved("a#01", "one", 0.9, [LEXICAL]),
                  _retrieved("b#01", "two", 0.5, [SEMANTIC])]
    assert [c.chunk.chunk_id for c in NoOpReranker().rerank("q", candidates, 2)] \
        == ["a#01", "b#01"]


def test_top_k_is_enforced():
    candidates = [_retrieved(f"a#{i:02d}", f"text {i}", 1.0 - i / 10, [LEXICAL])
                  for i in range(1, 6)]
    assert len(HeuristicReranker().rerank("text", candidates, top_k=2)) == 2


def test_output_is_sorted_by_descending_score():
    candidates = [_retrieved("a#01", "refund policy details", 0.3, [LEXICAL]),
                  _retrieved("b#01", "unrelated content", 0.9, [SEMANTIC])]
    ranked = HeuristicReranker().rerank("refund policy", candidates, 2)
    scores = [c.score for c in ranked]
    assert scores == sorted(scores, reverse=True)


def test_relevance_breaks_a_tie_between_equally_fused_candidates():
    """
    What the heuristic reranker actually does: it is a TIE-BREAKER.

    RRF only knows ranks, so two candidates can fuse to the same score while
    only one of them contains the words the customer used. Here relevance
    decides, which is the reranker's genuine contribution.
    """
    candidates = [
        _retrieved("a#01", "completely unrelated filler about weather", 0.5,
                   [LEXICAL], title="Weather"),
        _retrieved("b#01", "refund policy eligibility and invoice rules", 0.5,
                   [LEXICAL], title="Refund policy", tags=["refund", "invoice"]),
    ]
    ranked = HeuristicReranker().rerank("refund policy eligibility invoice",
                                        candidates, top_k=2)
    assert ranked[0].chunk.chunk_id == "b#01"


def test_fusion_is_the_strongest_single_signal():
    """
    Coverage ALONE must not overturn fusion. The two retrieval arms searched
    the whole corpus; the reranker only saw twelve candidates, so it gets a
    vote rather than a veto.
    """
    candidates = [
        _retrieved("a#01", "unrelated filler", 0.9, [LEXICAL], title="Doc"),
        _retrieved("b#01", "refund policy eligibility invoice", 0.85, [LEXICAL],
                   title="Doc"),      # identical title → no field signal
    ]
    ranked = HeuristicReranker().rerank("refund policy eligibility invoice",
                                        candidates, top_k=2)
    assert ranked[0].chunk.chunk_id == "a#01"


def test_agreement_between_arms_is_rewarded():
    """Identical text and score; only the provenance differs."""
    both = _retrieved("a#01", "refund policy", 0.5, [LEXICAL, SEMANTIC])
    single = _retrieved("b#01", "refund policy", 0.5, [LEXICAL])
    ranked = HeuristicReranker().rerank("refund policy", [both, single], 2)
    assert ranked[0].chunk.chunk_id == "a#01"


def test_a_title_match_boosts_a_chunk():
    plain = _retrieved("a#01", "generic body text", 0.5, [LEXICAL], title="Other")
    titled = _retrieved("b#01", "generic body text", 0.5, [LEXICAL],
                        title="Rotating API keys")
    ranked = HeuristicReranker().rerank("rotating api keys", [plain, titled], 2)
    assert ranked[0].chunk.chunk_id == "b#01"


def test_an_exact_phrase_match_is_rewarded():
    exact = _retrieved("a#01", "the export limit is twenty per hour", 0.5, [LEXICAL])
    scattered = _retrieved("b#01", "limit hour export twenty the per is", 0.5, [LEXICAL])
    ranked = HeuristicReranker().rerank("the export limit is twenty per hour",
                                        [exact, scattered], 2)
    assert ranked[0].chunk.chunk_id == "a#01"


def test_reranking_is_deterministic():
    candidates = [_retrieved("a#01", "refund policy", 0.5, [LEXICAL]),
                  _retrieved("b#01", "refund policy", 0.5, [LEXICAL])]
    first = [c.chunk.chunk_id for c in HeuristicReranker().rerank("refund", candidates, 2)]
    second = [c.chunk.chunk_id for c in HeuristicReranker().rerank("refund", candidates, 2)]
    assert first == second == sorted(first)     # ties break on chunk_id


def test_scores_stay_in_a_bounded_range():
    """A [0,1] blend keeps the score comparable and readable in the UI."""
    candidates = [_retrieved("a#01", "refund policy eligibility", 0.5, [LEXICAL, SEMANTIC])]
    ranked = HeuristicReranker().rerank("refund policy eligibility", candidates, 1)
    assert 0.0 <= ranked[0].score <= 1.0


def test_the_rerank_score_is_recorded_separately():
    """The fusion score must survive, so the trace can show both stages."""
    candidates = [_retrieved("a#01", "refund policy", 0.5, [LEXICAL])]
    ranked = HeuristicReranker().rerank("refund", candidates, 1)
    assert ranked[0].rerank_score is not None
    assert ranked[0].fusion_score == 0.5


def test_empty_candidates_are_handled():
    assert HeuristicReranker().rerank("anything", [], 5) == []


def test_a_single_candidate_does_not_divide_by_zero():
    """Min-max normalisation over one item has a zero span."""
    ranked = HeuristicReranker().rerank("refund", [_retrieved("a#01", "refund", 0.5,
                                                              [LEXICAL])], 1)
    assert len(ranked) == 1


@pytest.mark.parametrize("configured,expected", [
    ("heuristic", "heuristic"),
    ("none", "none"),
    ("nonsense-value", "heuristic"),     # unknown values degrade safely
])
def test_factory_selects_the_configured_reranker(monkeypatch, configured, expected):
    monkeypatch.setenv("RERANKER", configured)
    assert get_reranker(get_settings()).name == expected


def test_cross_encoder_degrades_to_heuristic_when_unavailable(monkeypatch):
    """A weaker reranker only reorders — never a correctness or security problem."""
    monkeypatch.setenv("RERANKER", "cross-encoder")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert get_reranker(get_settings()).name == "heuristic"
