"""
main.py — THE REST API around the agent core.

Design rule: routes stay THIN. Each one parses the request, calls into
`app.core` / `app.rag` / `app.database`, and shapes the response. No business
logic lives here — it lives where it can be unit-tested without a web server.

THE ONE PLACE MONEY MOVES: `POST /api/approvals/{id}/decision`. It is the only
code in the entire project that sets an invoice to 'refunded'. The
authorisation boundary sits at the HUMAN DECISION, not inside the agent — that
is the whole architecture of the approval system in one sentence.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.api.models import (ApprovalDecision, ApprovalOut, IndexStatus, RunOut,
                            SearchOut, Stats, TicketCreate, TicketOut,
                            RetrievalOut, ToolCallOut, TriageRequest)
from app.config import get_settings
from app.core import prompts
from app.core.agent import run_triage
from app.core.tools import REGISTRY, ToolError, search_knowledge_base
from app.database import db
from app.providers import get_provider
from app.rag.hybrid import get_retriever
from app.rag.ingest import index_stats

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("resolveai.api")

_EVAL_RESULTS = Path(__file__).resolve().parents[2] / "evaluation" / "results"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure the schema exists before the first request is served."""
    db.init_db()
    yield


app = FastAPI(
    title="ResolveAI-RAG",
    version="1.0.0",
    description="Secure, agentic, retrieval-augmented customer support.",
    lifespan=lifespan,
)

# CORS: the browser rule that a page served from origin A may only call API B
# if B says so. The React dev server runs on :5173.
_origins = [o.strip() for o in get_settings().cors_origins.split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins,
                   allow_methods=["*"], allow_headers=["*"])


# ── helpers ───────────────────────────────────────────────────────
def _load_json(text, default=None):
    try:
        return json.loads(text) if text else default
    except (json.JSONDecodeError, TypeError):
        return default


def _run_out(row) -> RunOut:
    """Assemble a run plus its tool-call trace and retrieval records."""
    run = dict(row)

    trace = [ToolCallOut(
        step=t["step"], tool_name=t["tool_name"],
        args=_load_json(t["args_json"], {}) or {},
        ok=bool(t["ok"]) if t["ok"] is not None else None,
        result=_load_json(t["result_json"]),
        error=t["error"], latency_ms=t["latency_ms"],
    ) for t in db.query("SELECT * FROM tool_calls WHERE run_id=? ORDER BY id", (run["id"],))]

    retrieval = [RetrievalOut(
        query=r["query"], mode_requested=r["mode_requested"], mode_used=r["mode_used"],
        lexical_candidates=r["lexical_candidates"], semantic_candidates=r["semantic_candidates"],
        fusion_method=r["fusion_method"], reranker=r["reranker"], final_k=r["final_k"],
        fallbacks=_load_json(r["fallbacks"], []) or [],
        top_chunk_ids=_load_json(r["top_chunk_ids"], []) or [],
        latency_ms=r["latency_ms"], created_at=r["created_at"],
    ) for r in db.query("SELECT * FROM retrievals WHERE run_id=? ORDER BY id", (run["id"],))]

    approval = db.query_one("SELECT id FROM approvals WHERE run_id=?", (run["id"],))

    return RunOut(
        id=run["id"], ticket_id=run["ticket_id"], prompt_version=run["prompt_version"],
        provider=run["provider"], model=run["model"], rag_mode=run["rag_mode"],
        status=run["status"], result=_load_json(run["result_json"]), error=run["error"],
        steps_used=run["steps_used"], latency_ms=run["latency_ms"],
        injection_flagged=bool(run["injection_flagged"]) if run["injection_flagged"] is not None else None,
        citations_valid=bool(run["citations_valid"]) if run["citations_valid"] is not None else None,
        approval_id=approval["id"] if approval else None,
        trace=trace, retrieval=retrieval, created_at=run["created_at"],
    )


def _approval_out(row) -> ApprovalOut:
    approval = dict(row)
    return ApprovalOut(
        id=approval["id"], run_id=approval["run_id"], ticket_id=approval["ticket_id"],
        action_type=approval["action_type"], payload=_load_json(approval["payload_json"]),
        rationale=approval["rationale"], state=approval["state"],
        decided_by=approval["decided_by"], decided_at=approval["decided_at"],
        created_at=approval["created_at"],
    )


# ── health & discovery ────────────────────────────────────────────
@app.get("/health")
def health():
    """Liveness, plus how this instance is configured."""
    settings = get_settings()
    provider = get_provider(settings=settings)
    return {
        "status": "ok",
        "provider": provider.name,
        "model": provider.model,
        "prompt_versions": list(prompts.PROMPTS.keys()),
        "tools": list(REGISTRY.keys()),
        "rag_enabled": settings.rag_enabled,
        "rag_mode": settings.rag_mode,
    }


@app.get("/api/tools")
def list_tools():
    """Every tool's schema plus whether it is privileged."""
    return [{**t.schema(), "privileged": t.privileged} for t in REGISTRY.values()]


# ── tickets ───────────────────────────────────────────────────────
@app.get("/api/tickets", response_model=list[TicketOut])
def list_tickets(status: str | None = None, limit: int = Query(50, ge=1, le=200)):
    if status:
        rows = db.query(
            "SELECT * FROM tickets WHERE status=? ORDER BY created_at DESC, id LIMIT ?",
            (status, limit))
    else:
        rows = db.query("SELECT * FROM tickets ORDER BY created_at DESC, id LIMIT ?", (limit,))
    return [TicketOut(**dict(r)) for r in rows]


@app.get("/api/tickets/{ticket_id}", response_model=TicketOut)
def get_ticket(ticket_id: str):
    row = db.query_one("SELECT * FROM tickets WHERE id=?", (ticket_id,))
    if row is None:
        raise HTTPException(status_code=404, detail=f"ticket {ticket_id} not found")
    return TicketOut(**dict(row))


@app.post("/api/tickets", response_model=TicketOut, status_code=201)
def create_ticket(payload: TicketCreate):
    """Create a ticket, auto-linking the account by sender email when known."""
    account = db.query_one("SELECT id FROM accounts WHERE contact_email=?",
                           (payload.sender_email,))
    ticket_id = db.new_id("tkt")
    db.execute(
        """INSERT INTO tickets
           (id, account_id, sender_email, subject, body, channel, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, 'new', ?)""",
        (ticket_id, account["id"] if account else None, payload.sender_email,
         payload.subject, payload.body, payload.channel, db.now_iso()))
    return get_ticket(ticket_id)


@app.post("/api/tickets/{ticket_id}/triage")
def triage_ticket(ticket_id: str, request: TriageRequest):
    """Run the agent on a ticket. 422 if the prompt version is unknown."""
    ticket = db.query_one("SELECT * FROM tickets WHERE id=?", (ticket_id,))
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"ticket {ticket_id} not found")
    if request.prompt_version not in prompts.PROMPTS:
        raise HTTPException(status_code=422,
                            detail=f"unknown prompt_version '{request.prompt_version}'")

    provider = get_provider(request.provider) if request.provider else None
    return run_triage(dict(ticket), provider=provider,
                      prompt_version=request.prompt_version)


@app.get("/api/tickets/{ticket_id}/runs", response_model=list[RunOut])
def runs_for_ticket(ticket_id: str):
    rows = db.query(
        "SELECT * FROM agent_runs WHERE ticket_id=? ORDER BY created_at DESC, id",
        (ticket_id,))
    return [_run_out(r) for r in rows]


@app.get("/api/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str):
    row = db.query_one("SELECT * FROM agent_runs WHERE id=?", (run_id,))
    if row is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return _run_out(row)


# ── approvals: the human-in-the-loop gate ─────────────────────────
@app.get("/api/approvals", response_model=list[ApprovalOut])
def list_approvals(state: str = "pending"):
    rows = db.query("SELECT * FROM approvals WHERE state=? ORDER BY created_at DESC, id",
                    (state,))
    return [_approval_out(a) for a in rows]


@app.post("/api/approvals/{approval_id}/decision")
def decide_approval(approval_id: str, decision: ApprovalDecision):
    """
    Approve or reject a proposed action. THE ONLY ENDPOINT THAT MOVES MONEY.

    Returns 409 if the approval was already decided. That is the idempotency
    guard: a replayed request — a double-click, a retried webhook, a network
    retry — must not produce a second refund.
    """
    approval = db.query_one("SELECT * FROM approvals WHERE id=?", (approval_id,))
    if approval is None:
        raise HTTPException(status_code=404, detail=f"approval {approval_id} not found")
    if approval["state"] != "pending":
        raise HTTPException(status_code=409,
                            detail=f"approval already {approval['state']}")

    now = db.now_iso()

    if decision.decision == "approved":
        payload = _load_json(approval["payload_json"], {}) or {}
        # *** THE ONLY DATABASE WRITE IN THE PROJECT THAT REFUNDS AN INVOICE ***
        if approval["action_type"] == "issue_refund":
            db.execute("UPDATE invoices SET status='refunded' WHERE id=?",
                       (payload.get("invoice_id"),))
        db.execute(
            "UPDATE approvals SET state='executed', decided_by=?, decided_at=? WHERE id=?",
            (decision.decided_by, now, approval_id))
        db.execute("UPDATE tickets SET status='closed' WHERE id=?", (approval["ticket_id"],))
        state = "executed"
    else:
        db.execute(
            "UPDATE approvals SET state='rejected', decided_by=?, decided_at=? WHERE id=?",
            (decision.decided_by, now, approval_id))
        db.execute("UPDATE tickets SET status='escalated' WHERE id=?", (approval["ticket_id"],))
        state = "rejected"

    logger.info("approval %s decided=%s by=%s", approval_id, state, decision.decided_by)
    return {"approval_id": approval_id, "state": state}


# ── knowledge base / RAG ──────────────────────────────────────────
@app.get("/api/kb/search", response_model=SearchOut)
def kb_search(q: str, limit: int = Query(4, ge=1, le=10)):
    """
    Run the hybrid retriever directly. Useful for demos and for debugging why
    a particular chunk did or did not surface.
    """
    try:
        return search_knowledge_base(q, limit)
    except ToolError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/rag/status", response_model=IndexStatus)
def rag_status():
    """
    Is the retrieval index present, current, and is semantic search actually
    available right now? `startup_fallbacks` is the honest answer to "is this
    really running hybrid retrieval?".
    """
    settings = get_settings()
    stats = index_stats(settings)
    retriever = get_retriever(settings)
    return IndexStatus(
        **{k: v for k, v in stats.items() if k in IndexStatus.model_fields},
        rag_mode=settings.rag_mode,
        semantic_available=retriever.semantic_available,
        startup_fallbacks=retriever.startup_fallbacks,
    )


# ── stats ─────────────────────────────────────────────────────────
@app.get("/api/stats", response_model=Stats)
def stats():
    by_status = {r["status"]: r["c"] for r in
                 db.query("SELECT status, COUNT(*) c FROM tickets GROUP BY status")}
    runs = db.query_one("SELECT COUNT(*) c FROM agent_runs")["c"]
    escalated = db.query_one("SELECT COUNT(*) c FROM agent_runs WHERE status='escalated'")["c"]
    avg_latency = db.query_one("SELECT AVG(latency_ms) a FROM agent_runs")["a"] or 0
    pending = db.query_one("SELECT COUNT(*) c FROM approvals WHERE state='pending'")["c"]
    injection = db.query_one("SELECT COUNT(*) c FROM agent_runs WHERE injection_flagged=1")["c"]

    retrievals = db.query_one("SELECT COUNT(*) c FROM retrievals")["c"]
    hybrid = db.query_one("SELECT COUNT(*) c FROM retrievals WHERE mode_used='hybrid'")["c"]
    fallbacks = db.query_one(
        "SELECT COUNT(*) c FROM retrievals WHERE fallbacks IS NOT NULL AND fallbacks != '[]'")["c"]

    return Stats(
        tickets_by_status=by_status,
        run_count=runs,
        escalation_rate=round(escalated / runs, 3) if runs else 0.0,
        avg_latency_ms=round(avg_latency, 1),
        pending_approvals=pending,
        injection_flagged_runs=injection,
        retrieval_count=retrievals,
        hybrid_retrieval_rate=round(hybrid / retrievals, 3) if retrievals else 0.0,
        fallback_retrieval_count=fallbacks,
    )


# ── evaluation results ────────────────────────────────────────────
@app.get("/api/eval/latest")
def eval_latest():
    """Serve whatever the evaluation suite last wrote."""
    if not _EVAL_RESULTS.exists() or not list(_EVAL_RESULTS.glob("*.json")):
        raise HTTPException(
            status_code=404,
            detail="no eval results yet — run: python evaluation/run_eval.py --compare v1 v2")
    return {path.stem: _load_json(path.read_text(encoding="utf-8"))
            for path in sorted(_EVAL_RESULTS.glob("*.json"))}
