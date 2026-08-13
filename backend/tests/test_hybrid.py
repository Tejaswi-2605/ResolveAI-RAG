"""
Tests for RRF fusion and the hybrid retriever.

The fusion tests check the ARITHMETIC directly against hand-computed values,
because "the ranking looked reasonable" is not a test of a scoring function.
"""

from __future__ import annotations

import pytest

from app.rag.hybrid import (HybridRetriever, reciprocal_rank_fusion,
                            reset_retriever_cache)
from app.rag.models import LEXICAL, SEMANTIC, Chunk


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, article_id=chunk_id.split("#")[0], title="T",
                 section="S", text="text", tags=[], url=None, product_area=None,
                 ordinal=1, source="kb_articles")


@pytest.fixture
def chunk_map():
    return {cid: _chunk(cid) for cid in ["a#01", "a#02", "b#01", "b#02", "c#01"]}


# ── the RRF arithmetic ────────────────────────────────────────────
def test_rrf_score_matches_the_formula(chunk_map):
    """score = 1/(k + rank), with rank 1-based."""
    fused = reciprocal_rank_fusion({LEXICAL: [("a#01", 9.9)]}, rrf_k=60,
                                   chunk_map=chunk_map)
    assert fused[0].score == pytest.approx(1 / 61)


def test_a_chunk_found_by_both_arms_sums_both_contributions(chunk_map):
    fused = reciprocal_rank_fusion(
        {LEXICAL: [("a#01", 9.9)], SEMANTIC: [("a#01", 0.8)]},
        rrf_k=60, chunk_map=chunk_map)
    assert len(fused) == 1                                   # merged, not duplicated
    assert fused[0].score == pytest.approx(2 / 61)


def test_rank_position_lowers_the_contribution(chunk_map):
    fused = reciprocal_rank_fusion(
        {LEXICAL: [("a#01", 9.0), ("a#02", 8.0), ("b#01", 7.0)]},
        rrf_k=60, chunk_map=chunk_map)
    assert [c.score for c in fused] == pytest.approx([1 / 61, 1 / 62, 1 / 63])


def test_agreement_beats_a_single_arm_first_place(chunk_map):
    """
    The behaviour RRF exists to produce: a chunk both arms rank second beats a
    chunk only one arm ranks first. 2/62 > 1/61.
    """
    fused = reciprocal_rank_fusion(
        {LEXICAL: [("a#01", 9.0), ("b#01", 8.0)],
         SEMANTIC: [("c#01", 0.9), ("b#01", 0.8)]},
        rrf_k=60, chunk_map=chunk_map)
    assert fused[0].chunk.chunk_id == "b#01"


def test_raw_scores_never_influence_the_fused_ranking(chunk_map):
    """
    The core reason RRF is used: BM25's unbounded scores and cosine's [-1,1]
    are not comparable, so only rank may matter. Multiplying one arm's scores
    by a thousand must change nothing.
    """
    lists = {LEXICAL: [("a#01", 9.0), ("a#02", 8.0)],
             SEMANTIC: [("b#01", 0.9), ("a#01", 0.5)]}
    inflated = {LEXICAL: [("a#01", 9000.0), ("a#02", 8000.0)],
                SEMANTIC: [("b#01", 0.9), ("a#01", 0.5)]}
    assert ([c.chunk.chunk_id for c in reciprocal_rank_fusion(lists, 60, chunk_map)]
            == [c.chunk.chunk_id for c in reciprocal_rank_fusion(inflated, 60, chunk_map)])


def test_weights_default_to_plain_rrf(chunk_map):
    """Omitting weights must reproduce textbook RRF exactly."""
    lists = {LEXICAL: [("a#01", 9.0)], SEMANTIC: [("b#01", 0.9)]}
    assert ([c.score for c in reciprocal_rank_fusion(lists, 60, chunk_map)]
            == [c.score for c in reciprocal_rank_fusion(
                lists, 60, chunk_map, weights={LEXICAL: 1.0, SEMANTIC: 1.0})])


def test_a_lower_arm_weight_reduces_that_arm_s_vote(chunk_map):
    """
    Weighted RRF. Two chunks each ranked #1 by one arm: down-weighting lexical
    must let the semantic result win.
    """
    lists = {LEXICAL: [("a#01", 9.0)], SEMANTIC: [("b#01", 0.9)]}
    fused = reciprocal_rank_fusion(lists, 60, chunk_map,
                                   weights={LEXICAL: 0.8, SEMANTIC: 1.0})
    assert fused[0].chunk.chunk_id == "b#01"
    assert fused[0].score == pytest.approx(1.0 / 61)
    assert fused[1].score == pytest.approx(0.8 / 61)


def test_smaller_k_sharpens_the_top_of_the_ranking(chunk_map):
    """`k` damps the gap between adjacent ranks; a small k widens it."""
    lists = {LEXICAL: [("a#01", 9.0), ("a#02", 8.0)]}
    sharp = reciprocal_rank_fusion(lists, rrf_k=1, chunk_map=chunk_map)
    smooth = reciprocal_rank_fusion(lists, rrf_k=60, chunk_map=chunk_map)
    assert (sharp[0].score - sharp[1].score) > (smooth[0].score - smooth[1].score)


# ── provenance ────────────────────────────────────────────────────
def test_provenance_records_both_arms(chunk_map):
    fused = reciprocal_rank_fusion(
        {LEXICAL: [("a#01", 9.9)], SEMANTIC: [("a#01", 0.8)]}, 60, chunk_map)
    item = fused[0]
    assert sorted(item.retrieval_methods) == [LEXICAL, SEMANTIC]
    assert item.ranks == {LEXICAL: 1, SEMANTIC: 1}
    assert item.method_scores == {LEXICAL: 9.9, SEMANTIC: 0.8}


def test_provenance_records_a_single_arm(chunk_map):
    fused = reciprocal_rank_fusion({SEMANTIC: [("b#01", 0.7)]}, 60, chunk_map)
    assert fused[0].retrieval_methods == [SEMANTIC]


def test_fusion_is_deterministic_and_breaks_ties_on_chunk_id(chunk_map):
    lists = {LEXICAL: [("b#02", 1.0)], SEMANTIC: [("a#02", 1.0)]}
    first = [c.chunk.chunk_id for c in reciprocal_rank_fusion(lists, 60, chunk_map)]
    assert first == sorted(first)          # equal scores → ascending chunk_id
    assert first == [c.chunk.chunk_id for c in reciprocal_rank_fusion(lists, 60, chunk_map)]


def test_ids_missing_from_the_chunk_map_are_dropped(chunk_map):
    """Guards against a stale index pointing at chunks that no longer exist."""
    fused = reciprocal_rank_fusion({LEXICAL: [("gone#99", 5.0), ("a#01", 4.0)]},
                                   60, chunk_map)
    assert [c.chunk.chunk_id for c in fused] == ["a#01"]


def test_empty_input_produces_no_results(chunk_map):
    assert reciprocal_rank_fusion({}, 60, chunk_map) == []


# ── the retriever end to end ──────────────────────────────────────
def test_hybrid_mode_runs_both_arms_and_fuses(retriever):
    evidence, trace = retriever.retrieve("how do I schedule a recurring report")
    assert trace.mode_requested == "hybrid"
    assert trace.mode_used == "hybrid"
    assert trace.fusion_method == "rrf"
    assert trace.lexical_candidates > 0
    assert trace.semantic_candidates > 0
    assert trace.fallbacks == []
    assert evidence


def test_lexical_mode_uses_only_bm25(retriever):
    evidence, trace = retriever.retrieve("ERR-4029", mode="lexical")
    assert trace.mode_used == "lexical"
    assert trace.semantic_candidates == 0
    assert trace.fusion_method is None
    assert all(item.retrieval_methods == ["lexical"] for item in evidence)


def test_semantic_mode_uses_only_the_vector_index(retriever):
    evidence, trace = retriever.retrieve("scheduling a report", mode="semantic")
    assert trace.mode_used == "semantic"
    assert trace.lexical_candidates == 0
    assert all(item.retrieval_methods == ["semantic"] for item in evidence)


def test_top_k_is_respected(retriever):
    evidence, trace = retriever.retrieve("csv export", top_k=2)
    assert len(evidence) == 2
    assert trace.final_k == 2


def test_results_carry_full_citation_metadata(retriever):
    evidence, _ = retriever.retrieve("rotating an API key")
    for item in evidence:
        record = item.to_dict()
        assert record["chunk_id"] and record["article_id"] and record["title"]
        assert record["retrieval_methods"]


def test_no_duplicate_chunks_in_the_output(retriever):
    evidence, _ = retriever.retrieve("api key rotation and error codes", top_k=6)
    ids = [item.chunk.chunk_id for item in evidence]
    assert len(ids) == len(set(ids))


def test_the_reranker_is_off_by_default(retriever):
    """
    Shipped default, chosen from the retrieval evaluation: the heuristic
    reranker measurably lowered recall@1 on this corpus.
    """
    _, trace = retriever.retrieve("csv export limits")
    assert trace.reranker is None


def test_reranking_can_be_switched_on(built_index, monkeypatch):
    """The stage is wired up and pluggable, even though it is off by default."""
    monkeypatch.setenv("RERANKER", "heuristic")
    reset_retriever_cache()
    from app.config import get_settings

    evidence, trace = HybridRetriever(get_settings()).retrieve("csv export limits")
    assert trace.reranker == "heuristic"
    assert evidence
    assert all(item.rerank_score is not None for item in evidence)


def test_the_trace_records_every_stage(retriever):
    _, trace = retriever.retrieve("csv export limits")
    assert set(trace.latency_ms) >= {"lexical", "semantic", "fusion", "total"}
    assert trace.rrf_k == 60
    assert trace.top_chunk_ids


def test_retrieval_is_deterministic(retriever):
    first, _ = retriever.retrieve("saml single sign-on")
    second, _ = retriever.retrieve("saml single sign-on")
    assert ([c.chunk.chunk_id for c in first] == [c.chunk.chunk_id for c in second])


def test_an_unknown_mode_falls_back_to_hybrid(retriever):
    _, trace = retriever.retrieve("csv export", mode="telepathy")
    assert trace.mode_requested == "hybrid"


def test_a_query_matching_nothing_returns_no_evidence(built_index):
    """Empty evidence is a valid answer — it is what triggers escalation."""
    retriever = HybridRetriever(built_index)
    evidence, trace = retriever.retrieve("zzzzqqq nonexistent gibberish", mode="lexical")
    assert evidence == []
    assert trace.final_k == 0
