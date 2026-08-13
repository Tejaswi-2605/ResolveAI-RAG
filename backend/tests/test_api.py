"""Tests for the REST API surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(built_index):
    return TestClient(app)


# ── health and discovery ──────────────────────────────────────────
def test_health_reports_the_configuration(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["provider"] == "mock"
    assert body["prompt_versions"] == ["v1", "v2"]
    assert body["rag_enabled"] is True


def test_tools_are_discoverable_with_their_privilege_flag(client):
    tools = client.get("/api/tools").json()
    privileged = [t["name"] for t in tools if t["privileged"]]
    assert privileged == ["issue_refund"]


# ── tickets ───────────────────────────────────────────────────────
def test_tickets_are_listable(client):
    assert len(client.get("/api/tickets").json()) == 12


def test_tickets_can_be_filtered_by_status(client):
    for ticket in client.get("/api/tickets?status=new").json():
        assert ticket["status"] == "new"


def test_a_missing_ticket_returns_404(client):
    assert client.get("/api/tickets/tkt_nope").status_code == 404


def test_a_ticket_can_be_created(client):
    response = client.post("/api/tickets", json={
        "sender_email": "priya@northwind.example",
        "subject": "New question", "body": "How do I export data?"})
    assert response.status_code == 201
    assert response.json()["account_id"] == "acct_001"    # auto-linked


def test_a_malformed_email_is_rejected(client):
    assert client.post("/api/tickets", json={
        "sender_email": "not-an-email", "subject": "s",
        "body": "b"}).status_code == 422


def test_an_empty_subject_is_rejected(client):
    assert client.post("/api/tickets", json={
        "sender_email": "a@b.example", "subject": "", "body": "b"}).status_code == 422


# ── triage ────────────────────────────────────────────────────────
def test_triage_returns_a_run_with_a_trace(client):
    body = client.post("/api/tickets/tkt_001/triage",
                       json={"prompt_version": "v2"}).json()
    assert body["status"] in ("completed", "escalated")
    assert body["trace"]
    assert body["stages"]


def test_triage_of_a_missing_ticket_returns_404(client):
    assert client.post("/api/tickets/tkt_nope/triage", json={}).status_code == 404


def test_an_unknown_prompt_version_returns_422(client):
    assert client.post("/api/tickets/tkt_001/triage",
                       json={"prompt_version": "v99"}).status_code == 422


def test_runs_are_retrievable_after_triage(client):
    run_id = client.post("/api/tickets/tkt_001/triage", json={}).json()["run_id"]

    run = client.get(f"/api/runs/{run_id}").json()
    assert run["id"] == run_id
    assert run["prompt_version"] == "v2"
    assert run["retrieval"]           # the RAG record is exposed


def test_a_missing_run_returns_404(client):
    assert client.get("/api/runs/run_nope").status_code == 404


def test_runs_for_a_ticket_are_listed(client):
    client.post("/api/tickets/tkt_001/triage", json={})
    assert client.get("/api/tickets/tkt_001/runs").json()


def test_the_run_record_exposes_the_retrieval_trace(client):
    run_id = client.post("/api/tickets/tkt_001/triage", json={}).json()["run_id"]
    retrieval = client.get(f"/api/runs/{run_id}").json()["retrieval"][0]
    assert retrieval["mode_used"] == "hybrid"
    assert retrieval["fusion_method"] == "rrf"
    assert retrieval["top_chunk_ids"]


# ── knowledge search ──────────────────────────────────────────────
def test_kb_search_returns_evidence_and_a_trace(client):
    body = client.get("/api/kb/search", params={"q": "csv export", "limit": 3}).json()
    assert 0 < len(body["evidence"]) <= 3
    assert body["retrieval"]["mode_used"] == "hybrid"


def test_kb_search_evidence_is_citable(client):
    for item in client.get("/api/kb/search", params={"q": "api keys"}).json()["evidence"]:
        assert item["chunk_id"] and item["article_id"] and item["retrieval_methods"]


def test_an_empty_kb_query_is_rejected(client):
    assert client.get("/api/kb/search", params={"q": "   "}).status_code == 400


def test_the_limit_is_bounded(client):
    assert client.get("/api/kb/search", params={"q": "csv", "limit": 99}).status_code == 422


# ── RAG status ────────────────────────────────────────────────────
def test_rag_status_reports_a_healthy_index(client):
    body = client.get("/api/rag/status").json()
    assert body["exists"] is True
    assert body["stale"] is False
    assert body["semantic_available"] is True
    assert body["startup_fallbacks"] == []


def test_rag_status_reports_a_missing_index(seeded_db):
    """The honest answer to "is this really running hybrid retrieval?"."""
    body = TestClient(app).get("/api/rag/status").json()
    assert body["exists"] is False
    assert body["startup_fallbacks"]


# ── stats ─────────────────────────────────────────────────────────
def test_stats_include_retrieval_health(client):
    client.post("/api/tickets/tkt_001/triage", json={})
    body = client.get("/api/stats").json()

    assert body["run_count"] >= 1
    assert body["retrieval_count"] >= 1
    assert body["hybrid_retrieval_rate"] == 1.0
    assert body["fallback_retrieval_count"] == 0


def test_stats_work_on_an_empty_database(client):
    body = client.get("/api/stats").json()
    assert body["run_count"] == 0
    assert body["escalation_rate"] == 0.0


# ── eval results ──────────────────────────────────────────────────
def test_eval_endpoint_404s_before_any_eval_has_run(client, monkeypatch, tmp_path):
    monkeypatch.setattr("app.main._EVAL_RESULTS", tmp_path / "nothing")
    assert client.get("/api/eval/latest").status_code == 404
