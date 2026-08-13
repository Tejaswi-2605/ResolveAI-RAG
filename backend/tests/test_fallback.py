"""
Tests for the fallback ladder (Phase 21).

THE RULE: retrieval degrades rather than failing, and the degradation is
ALWAYS visible in the trace. The system must never report "hybrid" for work it
did not do — a silent fallback is worse than an outage, because it produces
confident answers from half a pipeline while the dashboard stays green.

    vector index missing      → lexical only,  fallbacks: vector_index_unavailable
    embedding model missing   → lexical only,  fallbacks: embedding_provider_unavailable
    chunk store missing       → rebuilt from SQLite, fallbacks: chunk_store_rebuilt...
"""

from __future__ import annotations

import json

import pytest

from app.config import get_settings
from app.core.agent import run_triage
from app.core.tools import search_knowledge_base
from app.rag.hybrid import HybridRetriever, reset_retriever_cache


# ── missing vector index ──────────────────────────────────────────
def test_a_missing_index_degrades_to_lexical(seeded_db):
    """No ingest has run. Retrieval must still work."""
    retriever = HybridRetriever(get_settings())
    evidence, trace = retriever.retrieve("how do I export data to CSV")

    assert evidence, "lexical retrieval must still return results"
    assert trace.mode_used == "lexical"
    assert trace.semantic_candidates == 0


def test_a_missing_index_is_named_in_the_trace(seeded_db):
    """The honesty requirement: the reason is recorded, not swallowed."""
    _, trace = HybridRetriever(get_settings()).retrieve("csv export")
    assert "vector_index_unavailable" in trace.fallbacks
    assert trace.mode_requested == "hybrid"      # what we asked for
    assert trace.mode_used == "lexical"          # what actually ran


def test_a_missing_chunk_store_is_rebuilt_from_sqlite(seeded_db):
    """
    The last rung: the authoritative knowledge is still in kb_articles, so the
    chunk store can be reconstructed in memory on the spot.
    """
    retriever = HybridRetriever(get_settings())
    assert "chunk_store_rebuilt_from_database" in retriever.startup_fallbacks
    assert retriever.chunks
    evidence, _ = retriever.retrieve("api key rotation")
    assert evidence


def test_deleting_the_index_after_ingest_still_serves_results(built_index):
    """Someone wipes data/index/ in production. Answers must keep flowing."""
    for artifact in built_index.index_dir.iterdir():
        artifact.unlink()
    reset_retriever_cache()

    evidence, trace = HybridRetriever(built_index).retrieve("csv export limits")
    assert evidence
    assert trace.mode_used == "lexical"
    assert trace.fallbacks


# ── missing embedding model ───────────────────────────────────────
def test_an_unavailable_embedding_model_degrades_to_lexical(built_index, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence-transformers")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    reset_retriever_cache()

    retriever = HybridRetriever(get_settings())
    assert retriever.semantic_available is False
    assert "embedding_provider_unavailable" in retriever.startup_fallbacks

    evidence, trace = retriever.retrieve("how do I export data")
    assert evidence
    assert trace.mode_used == "lexical"


def test_a_model_failing_mid_query_is_caught(built_index):
    """A model that loads but then throws must not take the request down."""
    retriever = HybridRetriever(built_index)

    def explode(text):
        raise RuntimeError("model crashed")

    retriever.embedder.embed_text = explode

    evidence, trace = retriever.retrieve("csv export")
    assert evidence
    assert "semantic_search_failed" in trace.fallbacks
    assert trace.mode_used == "lexical"


def test_semantic_mode_falls_back_rather_than_returning_nothing(seeded_db):
    """
    Explicitly asking for semantic-only when it is unavailable must still
    answer the customer — degraded, and labelled as such.
    """
    evidence, trace = HybridRetriever(get_settings()).retrieve("csv export",
                                                              mode="semantic")
    assert trace.mode_requested == "semantic"
    assert trace.mode_used == "lexical"
    assert "semantic_unavailable" in trace.fallbacks
    assert evidence


# ── a stale index ─────────────────────────────────────────────────
def test_a_stale_index_is_detected(built_index):
    """
    The knowledge base changed but nobody re-ran ingest. The index now points
    at chunk ids that no longer exist.
    """
    ids_path = built_index.index_dir / "ids.json"
    ids_path.write_text(json.dumps(["ghost#01"] * len(
        json.loads(ids_path.read_text(encoding="utf-8")))), encoding="utf-8")
    reset_retriever_cache()

    retriever = HybridRetriever(built_index)
    assert "vector_index_stale" in retriever.startup_fallbacks


# ── the fallback surfaces all the way up ──────────────────────────
def test_the_tool_reports_the_fallback_to_the_agent(seeded_db):
    result = search_knowledge_base("how do I export data to CSV")
    assert result["retrieval"]["mode_used"] == "lexical"
    assert result["retrieval"]["fallbacks"]


def test_a_full_run_succeeds_with_no_vector_index(seeded_db, ticket_factory):
    """End to end: no index at all, and the agent still answers a customer."""
    ticket = ticket_factory("How do I export data?",
                            "I need to get our contacts out as a CSV file.")
    run = run_triage(ticket)

    assert run["status"] in ("completed", "escalated")
    assert run["rag_mode"] == "lexical"
    assert run["evidence"]


def test_the_fallback_is_persisted_for_later_analysis(seeded_db, ticket_factory):
    """"How often did semantic retrieval fall back last week?" must be answerable."""
    from app.database import db

    ticket = ticket_factory("How do I export data?", "Get our contacts as CSV.")
    run = run_triage(ticket)

    row = db.query_one("SELECT * FROM retrievals WHERE run_id=?", (run["run_id"],))
    assert row is not None
    assert row["mode_used"] == "lexical"
    assert "vector_index_unavailable" in json.loads(row["fallbacks"])


# ── retrieval disabled entirely ───────────────────────────────────
def test_the_agent_survives_rag_being_switched_off(built_index, ticket_factory,
                                                   monkeypatch):
    """RAG_ENABLED=false makes the tool error; the agent must still finish."""
    monkeypatch.setenv("RAG_ENABLED", "false")
    ticket = ticket_factory("How do I export data?", "Get our contacts as CSV.")
    run = run_triage(ticket, settings=get_settings())

    assert run["status"] in ("completed", "escalated")
    assert run["evidence"] == []


@pytest.mark.parametrize("missing", ["vectors.npy", "ids.json"])
def test_a_partially_deleted_index_is_treated_as_missing(built_index, missing):
    """Half an index is not an index — it must not be loaded optimistically."""
    (built_index.index_dir / missing).unlink()
    reset_retriever_cache()

    retriever = HybridRetriever(built_index)
    assert retriever.semantic_available is False
    assert retriever.startup_fallbacks
