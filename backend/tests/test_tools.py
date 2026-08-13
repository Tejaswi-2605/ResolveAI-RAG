"""Tests for the tool layer: argument validation, business rules, and the RAG adapter."""

from __future__ import annotations

import pytest

from app.core.tools import (REGISTRY, Tool, ToolError, call_tool,
                            check_service_status, escalate_to_human,
                            get_billing_history, issue_refund, lookup_account,
                            search_knowledge_base, tool_schemas, validate_args)
from app.database import db


# ── argument validation: the trust boundary on model output ───────
@pytest.fixture
def sample_tool():
    return Tool(
        name="sample", description="d", fn=lambda **kw: kw,
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "minLength": 2, "maxLength": 10},
                "count": {"type": "integer", "minimum": 1, "maximum": 5},
                "mode": {"type": "string", "enum": ["a", "b"]},
                "email": {"type": "string", "pattern": r"^[^@\s]+@[^@\s]+\.[^@\s]+$"},
            },
            "required": ["text"],
        })


def test_valid_arguments_pass(sample_tool):
    assert validate_args(sample_tool, {"text": "hello", "count": 3}) == {
        "text": "hello", "count": 3}


def test_unknown_arguments_are_rejected(sample_tool):
    """An argument we never declared is a hallucination or a probe."""
    with pytest.raises(ToolError, match="unknown argument"):
        validate_args(sample_tool, {"text": "hi", "admin": True})


def test_missing_required_arguments_are_rejected(sample_tool):
    with pytest.raises(ToolError, match="missing required"):
        validate_args(sample_tool, {"count": 2})


def test_wrong_types_are_rejected(sample_tool):
    with pytest.raises(ToolError, match="must be of type"):
        validate_args(sample_tool, {"text": 123})


def test_a_boolean_is_not_an_integer(sample_tool):
    """In Python bool subclasses int, so this needs an explicit guard."""
    with pytest.raises(ToolError, match="not a boolean"):
        validate_args(sample_tool, {"text": "hi", "count": True})


def test_numeric_bounds_are_enforced(sample_tool):
    with pytest.raises(ToolError, match=">= 1"):
        validate_args(sample_tool, {"text": "hi", "count": 0})
    with pytest.raises(ToolError, match="<= 5"):
        validate_args(sample_tool, {"text": "hi", "count": 99})


def test_string_length_bounds_are_enforced(sample_tool):
    with pytest.raises(ToolError, match="maxLength"):
        validate_args(sample_tool, {"text": "x" * 50})
    with pytest.raises(ToolError, match="minLength"):
        validate_args(sample_tool, {"text": "x"})


def test_enum_values_are_enforced(sample_tool):
    with pytest.raises(ToolError, match="must be one of"):
        validate_args(sample_tool, {"text": "hi", "mode": "c"})


def test_patterns_are_enforced(sample_tool):
    with pytest.raises(ToolError, match="required format"):
        validate_args(sample_tool, {"text": "hi", "email": "not-an-email"})


def test_non_dict_arguments_are_rejected(sample_tool):
    with pytest.raises(ToolError, match="must be an object"):
        validate_args(sample_tool, ["not", "a", "dict"])


# ── the registry ──────────────────────────────────────────────────
def test_all_six_tools_are_registered():
    assert set(REGISTRY) == {
        "search_knowledge_base", "lookup_account", "get_billing_history",
        "check_service_status", "issue_refund", "escalate_to_human"}


def test_only_issue_refund_is_privileged():
    """Exactly one tool touches money, so exactly one may be privileged."""
    assert [n for n, t in REGISTRY.items() if t.privileged] == ["issue_refund"]


def test_schemas_are_model_ready():
    for schema in tool_schemas():
        assert {"name", "description", "input_schema"} <= set(schema)


def test_calling_an_unknown_tool_raises():
    with pytest.raises(ToolError, match="unknown tool"):
        call_tool("definitely_not_a_tool", {})


def test_call_tool_validates_before_executing(seeded_db):
    """Validation cannot be skipped — that is why call_tool is the only door."""
    with pytest.raises(ToolError, match="unknown argument"):
        call_tool("lookup_account", {"email": "priya@northwind.example", "x": 1})


def test_call_tool_reports_latency(seeded_db):
    _, latency_ms = call_tool("lookup_account", {"email": "priya@northwind.example"})
    assert latency_ms >= 0


# ── database-backed tools ─────────────────────────────────────────
def test_lookup_account_finds_a_known_account(seeded_db):
    assert lookup_account("priya@northwind.example")["company"] == "Northwind Trading"


def test_lookup_account_rejects_a_malformed_email(seeded_db):
    with pytest.raises(ToolError, match="invalid email"):
        lookup_account("not-an-email")


def test_lookup_account_reports_an_unknown_account(seeded_db):
    """The message must be readable by a MODEL so it can self-correct."""
    with pytest.raises(ToolError, match="no account found"):
        lookup_account("nobody@nowhere.example")


def test_billing_history_returns_invoices(seeded_db):
    result = get_billing_history("acct_001")
    assert result["account_id"] == "acct_001"
    assert result["invoices"]


def test_billing_history_rejects_an_unknown_account(seeded_db):
    with pytest.raises(ToolError, match="unknown account"):
        get_billing_history("acct_does_not_exist")


def test_service_status_returns_a_component(seeded_db):
    assert check_service_status("api")["state"] in ("operational", "degraded", "outage")


def test_escalation_returns_a_structured_record():
    assert escalate_to_human("needs a person", "high") == {
        "action_type": "escalate_to_human", "reason": "needs a person", "priority": "high"}


# ── issue_refund: the privileged tool ─────────────────────────────
def test_refund_returns_a_proposal_and_writes_nothing(seeded_db):
    """
    THE most important test in the project. The tool must not move money.
    """
    before = db.query_one("SELECT status FROM invoices WHERE id='inv_001'")["status"]
    proposal = issue_refund("acct_001", "inv_001", 49900, "duplicate charge")
    after = db.query_one("SELECT status FROM invoices WHERE id='inv_001'")["status"]

    assert proposal["action_type"] == "issue_refund"
    assert proposal["requires_human_approval"] is True
    assert after == before == "paid"       # nothing changed


def test_refund_blocked_for_an_ineligible_account(seeded_db):
    """
    Business rules live in CODE. A knowledge-base article describing a generous
    refund policy cannot authorise this — only the database can.
    """
    with pytest.raises(ToolError, match="not refund-eligible"):
        issue_refund("acct_005", "inv_005", 100, "customer asked")


def test_refund_blocked_when_the_invoice_belongs_to_another_account(seeded_db):
    with pytest.raises(ToolError, match="does not belong"):
        issue_refund("acct_001", "inv_002", 100, "wrong owner")


def test_double_refund_is_blocked(seeded_db):
    with pytest.raises(ToolError, match="already refunded"):
        issue_refund("acct_001", "inv_007", 100, "again please")


def test_refund_cannot_exceed_the_invoice_total(seeded_db):
    with pytest.raises(ToolError, match="exceeds the invoice"):
        issue_refund("acct_001", "inv_001", 999_999, "too much")


def test_refund_rejects_an_unknown_invoice(seeded_db):
    with pytest.raises(ToolError, match="unknown invoice"):
        issue_refund("acct_001", "inv_nope", 100, "missing")


# ── the RAG adapter ───────────────────────────────────────────────
def test_knowledge_tool_returns_evidence_and_a_trace(built_index):
    result = search_knowledge_base("how do I schedule a report", limit=3)
    assert result["query"]
    assert 0 < len(result["evidence"]) <= 3
    assert result["retrieval"]["mode_used"] in ("hybrid", "lexical", "semantic")


def test_evidence_items_carry_citable_identity(built_index):
    for item in search_knowledge_base("csv export", limit=3)["evidence"]:
        assert item["chunk_id"] and item["article_id"] and item["title"]
        assert item["retrieval_methods"]


def test_the_tool_leaks_no_retrieval_internals(built_index):
    """
    The layering boundary, asserted. The agent must never be handed a FAISS
    handle, a raw vector, or a BM25 object.
    """
    item = search_knowledge_base("api keys", limit=1)["evidence"][0]
    assert not {"vector", "embedding", "faiss", "index"} & set(item)


def test_an_empty_query_is_rejected(built_index):
    with pytest.raises(ToolError, match="must not be empty"):
        search_knowledge_base("   ")


def test_the_tool_refuses_when_rag_is_disabled(built_index, monkeypatch):
    monkeypatch.setenv("RAG_ENABLED", "false")
    with pytest.raises(ToolError, match="retrieval is disabled"):
        search_knowledge_base("anything")
