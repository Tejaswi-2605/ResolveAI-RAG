"""
api/models.py — THE API'S REQUEST AND RESPONSE SHAPES (Pydantic v2).

Pydantic models do two jobs for a FastAPI application:
  1. VALIDATE incoming JSON — wrong type or missing field becomes an automatic
     422 before a single line of our code runs.
  2. DOCUMENT the API — FastAPI turns these into the OpenAPI/Swagger schema.

Keeping them separate from the database rows means the public API shape is
decoupled from internal storage: we choose exactly what to expose, and a
column rename does not become a breaking API change.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# Pydantic's `EmailStr` would pull in the `email-validator` package for one
# field. The same pattern already guards `lookup_account` in tools.py, so we
# reuse it here and keep the dependency list to things the project genuinely
# needs.
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


# ── requests ──────────────────────────────────────────────────────
class TicketCreate(BaseModel):
    """Body for POST /api/tickets."""

    sender_email: str = Field(pattern=EMAIL_PATTERN, max_length=254)
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=8000)
    channel: Literal["email", "chat", "web"] = "email"


class TriageRequest(BaseModel):
    """Body for POST /api/tickets/{id}/triage."""

    prompt_version: str = "v2"
    provider: Optional[str] = None


class ApprovalDecision(BaseModel):
    """Body for POST /api/approvals/{id}/decision."""

    decision: Literal["approved", "rejected"]
    decided_by: str = Field(min_length=1, max_length=120)
    note: Optional[str] = None


# ── responses ─────────────────────────────────────────────────────
class TicketOut(BaseModel):
    id: str
    account_id: Optional[str] = None
    sender_email: str
    subject: str
    body: str
    channel: Optional[str] = None
    status: str
    created_at: str


class ToolCallOut(BaseModel):
    """One entry in the agent trace rail."""

    step: Optional[int] = None
    tool_name: str
    args: dict = {}
    ok: Optional[bool] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    latency_ms: Optional[int] = None


class RetrievalOut(BaseModel):
    """
    One knowledge-tool call, as recorded in the `retrievals` table.

    This is what makes the RAG pipeline inspectable from the UI: which arms
    ran, how many candidates each produced, whether a fallback fired, and which
    chunks won.
    """

    query: str
    mode_requested: Optional[str] = None
    mode_used: Optional[str] = None
    lexical_candidates: Optional[int] = None
    semantic_candidates: Optional[int] = None
    fusion_method: Optional[str] = None
    reranker: Optional[str] = None
    final_k: Optional[int] = None
    fallbacks: list[str] = []
    top_chunk_ids: list[str] = []
    latency_ms: Optional[int] = None
    created_at: Optional[str] = None


class RunOut(BaseModel):
    id: str
    ticket_id: str
    prompt_version: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    rag_mode: Optional[str] = None
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None
    steps_used: Optional[int] = None
    latency_ms: Optional[int] = None
    injection_flagged: Optional[bool] = None
    citations_valid: Optional[bool] = None
    approval_id: Optional[str] = None
    trace: list[ToolCallOut] = []
    retrieval: list[RetrievalOut] = []
    created_at: Optional[str] = None


class ApprovalOut(BaseModel):
    id: str
    run_id: str
    ticket_id: str
    action_type: str
    payload: Optional[dict] = None
    rationale: Optional[str] = None
    state: str
    decided_by: Optional[str] = None
    decided_at: Optional[str] = None
    created_at: str


class Stats(BaseModel):
    """Payload for GET /api/stats — the dashboard numbers."""

    tickets_by_status: dict
    run_count: int
    escalation_rate: float
    avg_latency_ms: float
    pending_approvals: int
    injection_flagged_runs: int
    retrieval_count: int
    hybrid_retrieval_rate: float
    fallback_retrieval_count: int


class IndexStatus(BaseModel):
    """Payload for GET /api/rag/status — is the retrieval index healthy?"""

    exists: bool
    stale: Optional[bool] = None
    index_dir: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    dimension: Optional[int] = None
    vector_backend: Optional[str] = None
    articles: Optional[int] = None
    chunks: Optional[int] = None
    built_at: Optional[str] = None
    rag_mode: Optional[str] = None
    semantic_available: Optional[bool] = None
    startup_fallbacks: list[str] = []


class EvidenceOut(BaseModel):
    """One retrieved chunk, as returned by GET /api/kb/search."""

    chunk_id: str
    article_id: str
    title: str
    section: Optional[str] = None
    text: str
    url: Optional[str] = None
    score: float
    retrieval_methods: list[str] = []
    ranks: dict[str, int] = {}
    method_scores: dict[str, float] = {}


class SearchOut(BaseModel):
    query: str
    evidence: list[EvidenceOut] = []
    retrieval: dict[str, Any] = {}
