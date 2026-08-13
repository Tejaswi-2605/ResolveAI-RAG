"""Tests for the database layer and the seed dataset."""

from __future__ import annotations

import json
import sqlite3

import pytest

from app.config import get_settings
from app.database import db


def test_the_schema_creates_every_table(seeded_db):
    tables = {r["name"] for r in db.query(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"accounts", "invoices", "kb_articles", "service_status", "tickets",
            "agent_runs", "tool_calls", "retrievals", "approvals"} <= tables


def test_init_db_is_safe_to_run_twice(seeded_db):
    db.init_db()
    assert db.query_one("SELECT COUNT(*) c FROM kb_articles")["c"] == 16


def test_the_seed_counts_are_as_documented(seeded_db):
    assert seeded_db == {"accounts": 6, "invoices": 7, "kb_articles": 16,
                         "service_status": 5, "tickets": 12}


def test_foreign_keys_are_enforced(seeded_db):
    """SQLite disables FK enforcement by default — the pragma must be set."""
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO invoices VALUES ('inv_x','acct_nope',100,'paid','t',NULL)")


def test_check_constraints_reject_invalid_enums(seeded_db):
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO tickets VALUES ('t_x',NULL,'a@b.example','s','b','email','bogus','t')")


def test_a_failed_transaction_rolls_back(seeded_db):
    """Commit on success, roll back on error: all the changes, or none."""
    before = db.query_one("SELECT COUNT(*) c FROM accounts")["c"]
    with pytest.raises(sqlite3.IntegrityError):
        with db.connect() as conn:
            conn.execute("INSERT INTO tickets VALUES "
                         "('t_ok',NULL,'a@b.example','s','b','email','new','t')")
            conn.execute("INSERT INTO accounts VALUES "
                         "('acct_001','dup','dup@x.example','trial',1,0,'active',1,'t')")
    assert db.query_one("SELECT COUNT(*) c FROM accounts")["c"] == before
    assert db.query_one("SELECT * FROM tickets WHERE id='t_ok'") is None


def test_queries_are_parameterised(seeded_db):
    """
    A classic injection string must be treated as DATA. If it were
    concatenated into SQL, the tickets table would be gone.
    """
    assert db.query_one("SELECT * FROM accounts WHERE contact_email=?",
                        ("x'; DROP TABLE tickets; --",)) is None
    assert db.query_one("SELECT COUNT(*) c FROM tickets")["c"] == 12


def test_ids_are_prefixed_and_unique():
    ids = {db.new_id("run") for _ in range(200)}
    assert len(ids) == 200
    assert all(i.startswith("run_") for i in ids)


def test_the_database_path_is_read_at_call_time(monkeypatch, tmp_path):
    """
    Lazy resolution is what gives each test its own database — the setting is
    read on every call, not captured at import.
    """
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "other.db"))
    assert str(get_settings().db_file).endswith("other.db")


# ── knowledge base access ─────────────────────────────────────────
def test_articles_are_returned_in_a_deterministic_order(seeded_db):
    """Ingestion determinism starts here: same database, same chunk order."""
    assert [a["id"] for a in db.all_kb_articles()] == sorted(
        a["id"] for a in db.all_kb_articles())


def test_every_article_has_the_fields_the_chunker_needs(seeded_db):
    for article in db.all_kb_articles():
        assert article["id"] and article["title"] and article["body"]
        assert "## " in article["body"], f"{article['id']} has no section headings"


def test_the_corpus_is_large_enough_for_retrieval_to_be_a_real_problem(seeded_db):
    articles = db.all_kb_articles()
    word_counts = [len(a["body"].split()) for a in articles]
    assert min(word_counts) > 150      # not two-sentence stubs
    assert len(articles) == 16


# ── run lifecycle ─────────────────────────────────────────────────
def test_a_run_can_be_started_and_finalised(seeded_db):
    run_id = db.new_id("run")
    db.start_run(run_id, "tkt_001", "v2", "mock", "mock-rules-v2", "hybrid", False)
    assert db.query_one("SELECT status FROM agent_runs WHERE id=?",
                        (run_id,))["status"] == "running"

    db.finalize_run(run_id, "completed", '{"ok":true}', None, 3, 120, 100, 50, True, "hybrid")
    row = db.query_one("SELECT * FROM agent_runs WHERE id=?", (run_id,))
    assert row["status"] == "completed"
    assert row["citations_valid"] == 1
    assert row["rag_mode"] == "hybrid"


def test_a_tool_call_needs_its_parent_run_to_exist(seeded_db):
    """
    Why start_run is called BEFORE the loop: tool_calls.run_id is a foreign key,
    so logging the first tool call would otherwise fail.
    """
    with pytest.raises(sqlite3.IntegrityError):
        db.log_tool_call("run_does_not_exist", 1, "lookup_account", "{}", True,
                         "{}", None, 5)


def test_a_retrieval_trace_round_trips(seeded_db):
    run_id = db.new_id("run")
    db.start_run(run_id, "tkt_001", "v2", "mock", "m", "hybrid", False)
    db.log_retrieval(run_id, {
        "query": "csv export", "mode_requested": "hybrid", "mode_used": "lexical",
        "lexical_candidates": 8, "semantic_candidates": 0, "fusion_method": None,
        "reranker": "heuristic", "final_k": 4,
        "fallbacks": ["vector_index_unavailable"],
        "top_chunk_ids": ["kb_002#01"], "latency_ms": 12,
    })
    row = db.query_one("SELECT * FROM retrievals WHERE run_id=?", (run_id,))
    assert row["mode_used"] == "lexical"
    assert json.loads(row["fallbacks"]) == ["vector_index_unavailable"]
    assert json.loads(row["top_chunk_ids"]) == ["kb_002#01"]


def test_an_approval_starts_pending(seeded_db):
    run_id = db.new_id("run")
    db.start_run(run_id, "tkt_002", "v2", "mock", "m", "hybrid", False)
    approval_id = db.new_id("apr")
    db.create_approval(approval_id, run_id, "tkt_002", "issue_refund", "{}", "why")
    assert db.query_one("SELECT state FROM approvals WHERE id=?",
                        (approval_id,))["state"] == "pending"
