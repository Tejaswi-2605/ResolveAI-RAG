"""
End-to-end integration tests for the retrieval pipeline over the REAL
knowledge base.

These run against the seeded 16-article corpus with the hashing embedding
provider, so they are fast and hermetic. They assert PIPELINE BEHAVIOUR — that
the stages connect correctly and provenance survives — rather than semantic
quality, which the hashing provider cannot deliver and which the retrieval
evaluation measures properly with the real model.
"""

from __future__ import annotations

import json

import pytest

from app.rag.chunking import chunk_articles
from app.rag.hybrid import HybridRetriever, reset_retriever_cache
from app.rag.ingest import (corpus_fingerprint, index_stats, ingest,
                            read_manifest)
from app.database import db


# ── ingestion ─────────────────────────────────────────────────────
def test_ingest_builds_every_artifact(settings):
    summary = ingest(settings)

    assert summary["articles"] == 16
    assert summary["chunks"] > 16          # chunking is doing real work
    for artifact in ("chunks.json", "ids.json", "vectors.npy", "manifest.json"):
        assert (settings.index_dir / artifact).exists(), artifact


def test_the_chunk_store_matches_the_index_row_order(built_index):
    """A mismatch here would map every vector to the wrong chunk."""
    chunks = json.loads((built_index.index_dir / "chunks.json").read_text(encoding="utf-8"))
    ids = json.loads((built_index.index_dir / "ids.json").read_text(encoding="utf-8"))
    assert [c["chunk_id"] for c in chunks] == ids


def test_the_manifest_pins_the_model_and_dimension(built_index):
    """
    An index built by a 384-dim model is meaningless to a 768-dim one.
    Recording it is what turns a silent garbage ranking into a caught error.
    """
    manifest = read_manifest(built_index)
    assert manifest["embedding_provider"] == "hashing"
    assert manifest["dimension"] == 256
    assert manifest["chunks"] > 0
    assert manifest["corpus_fingerprint"]


def test_ingestion_is_idempotent(settings):
    """Same knowledge base, same settings → byte-identical chunks."""
    first = ingest(settings)
    first_chunks = (settings.index_dir / "chunks.json").read_text(encoding="utf-8")
    second = ingest(settings)
    second_chunks = (settings.index_dir / "chunks.json").read_text(encoding="utf-8")

    assert first["corpus_fingerprint"] == second["corpus_fingerprint"]
    assert first_chunks == second_chunks


def test_a_dry_run_writes_nothing(settings):
    summary = ingest(settings, dry_run=True)
    assert summary["dry_run"] is True
    assert summary["chunks"] > 0
    assert not (settings.index_dir / "manifest.json").exists()


def test_the_fingerprint_changes_when_an_article_changes(built_index):
    before = corpus_fingerprint(db.all_kb_articles())
    db.execute("UPDATE kb_articles SET body=body || ' A new sentence.' WHERE id='kb_001'")
    assert corpus_fingerprint(db.all_kb_articles()) != before


def test_stats_report_a_stale_index(built_index):
    assert index_stats(built_index)["stale"] is False
    db.execute("UPDATE kb_articles SET body=body || ' Changed.' WHERE id='kb_002'")
    assert index_stats(built_index)["stale"] is True


def test_stats_report_a_missing_index(settings):
    assert index_stats(settings)["exists"] is False


def test_the_index_is_rebuildable_from_scratch(built_index):
    """
    The claim that justifies git-ignoring data/index/: deleting it loses
    nothing, because kb_articles is the source of truth.
    """
    before = json.loads((built_index.index_dir / "chunks.json").read_text(encoding="utf-8"))
    for artifact in built_index.index_dir.iterdir():
        artifact.unlink()
    reset_retriever_cache()

    ingest(built_index)
    after = json.loads((built_index.index_dir / "chunks.json").read_text(encoding="utf-8"))
    assert before == after


# ── the full pipeline ─────────────────────────────────────────────
def test_every_chunk_traces_back_to_a_real_article(built_index):
    retriever = HybridRetriever(built_index)
    article_ids = {a["id"] for a in db.all_kb_articles()}
    for chunk in retriever.chunks:
        assert chunk.article_id in article_ids


def test_the_chain_from_article_to_citation_is_unbroken(built_index):
    """
    article → chunk → retrieved evidence → citation.

    Every link must survive, or a citation cannot be verified against a source
    a human can open.
    """
    evidence, _ = HybridRetriever(built_index).retrieve("how do I rotate an API key")
    assert evidence

    for item in evidence:
        record = item.to_dict()
        article = db.query_one("SELECT * FROM kb_articles WHERE id=?",
                               (record["article_id"],))
        assert article is not None
        assert record["chunk_id"].startswith(record["article_id"] + "#")
        assert record["title"] == article["title"]
        assert record["url"] == article["url"]


def test_an_exact_error_code_retrieves_the_right_article(built_index):
    """Lexical's home turf: a rare exact identifier."""
    evidence, trace = HybridRetriever(built_index).retrieve("ERR-4029")
    assert evidence
    assert "kb_011" in {item.chunk.article_id for item in evidence}
    assert "lexical" in evidence[0].retrieval_methods


def test_both_arms_run_and_agree_over_the_real_corpus(built_index):
    """
    The pipeline-level check: with an index present, both arms run and their
    results fuse. This uses the hashing provider, so it proves the WIRING —
    not semantic quality.
    """
    retriever = HybridRetriever(built_index)
    for query in ["how do I schedule a recurring report", "rotating an api key safely",
                  "csv export encoding", "saml single sign-on"]:
        evidence, trace = retriever.retrieve(query)
        assert trace.mode_used == "hybrid"
        assert trace.fusion_method == "rrf"
        assert trace.lexical_candidates > 0
        assert trace.semantic_candidates > 0
        assert evidence


@pytest.mark.slow
def test_the_semantic_arm_contributes_unique_results(settings, monkeypatch):
    """
    Proof the system is genuinely hybrid, and it needs the REAL model.

    Over a batch of realistic queries, at least one result must be found by the
    semantic arm ALONE. If every result also came from BM25, the vector index
    would be decoration and "hybrid" would be a label rather than a mechanism.

    This cannot pass with the hashing provider, which scores vocabulary overlap
    and therefore mostly agrees with BM25 by construction — exactly the
    limitation documented in embeddings.py.
    """
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence-transformers")
    monkeypatch.setenv("VECTOR_BACKEND", "faiss")
    pytest.importorskip("sentence_transformers")

    from app.config import get_settings
    live = get_settings()
    ingest(live)
    reset_retriever_cache()
    retriever = HybridRetriever(live)

    queries = ["someone left the company and we still pay for them",
               "log in with our company account instead of a password",
               "our bill is overdue and the card was declined",
               "stop the report from being emailed every week"]

    semantic_only = 0
    for query in queries:
        evidence, trace = retriever.retrieve(query)
        assert trace.mode_used == "hybrid"
        semantic_only += sum(1 for item in evidence
                             if item.retrieval_methods == ["semantic"])

    assert semantic_only > 0, "the semantic arm never contributed a unique result"


def test_chunking_settings_change_the_index(settings, monkeypatch):
    small = chunk_articles(db.all_kb_articles(), 40, 5)
    large = chunk_articles(db.all_kb_articles(), 400, 20)
    assert len(small) > len(large)


def test_retrieval_latency_is_recorded_per_stage(built_index):
    _, trace = HybridRetriever(built_index).retrieve("csv export limits")
    assert trace.latency_ms["total"] > 0
    assert trace.latency_ms["total"] >= trace.latency_ms["lexical"]


def test_the_retriever_cache_returns_one_instance(built_index):
    from app.rag.hybrid import get_retriever
    assert get_retriever(built_index) is get_retriever(built_index)


def test_resetting_the_cache_rebuilds_the_retriever(built_index):
    from app.rag.hybrid import get_retriever
    first = get_retriever(built_index)
    reset_retriever_cache()
    assert get_retriever(built_index) is not first
