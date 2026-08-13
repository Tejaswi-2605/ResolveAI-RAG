"""
tools.py — THE ONLY ACTIONS THE AGENT IS ALLOWED TO TAKE.

WHAT A TOOL IS
An LLM by itself can only produce text. A "tool" is a function we expose so it
can *do* something — look up an account, search the knowledge base, propose a
refund. The model never runs these functions. It emits a NAME and ARGUMENTS,
and OUR code decides whether to run them. That gap is where all the safety
lives, and it is the single most important thing to understand about agents.

THREE SAFETY IDEAS IN THIS FILE

1. EVERY ARGUMENT IS VALIDATED by our code (`validate_args`) before a tool
   runs. Arguments chosen by a model are untrusted input, exactly like a form
   field submitted by a browser.

2. THE ONE TOOL THAT COULD MOVE MONEY NEVER WRITES. `issue_refund` is marked
   `privileged` and returns a PROPOSAL. There is no INSERT or UPDATE anywhere
   in it. The agent proposes; a human executes.

3. BUSINESS RULES LIVE IN CODE, NOT IN THE PROMPT. `issue_refund` re-checks
   eligibility, ownership, double-refunds and amounts against the database
   every time. The knowledge base may *describe* the refund policy, and the
   model may have read that description — but the description is not what
   enforces it. This is the RAG-specific trap worth naming out loud: retrieved
   policy text is documentation, never authorisation.

THE KNOWLEDGE TOOL AND THE LAYERING RULE
`search_knowledge_base` is the agent's ONLY door to the RAG engine:

    agent  →  search_knowledge_base  →  HybridRetriever  →  BM25 / FAISS / RRF / rerank

The agent never imports faiss, an embedding model, BM25, RRF or the reranker.
It asks a question and receives evidence dicts. That means the entire retrieval
strategy can be replaced without touching a line of agent code — which is the
main architectural claim this project makes.
"""

from __future__ import annotations

import re
import time
from typing import Callable

from app.config import get_settings
from app.database import db
from app.rag.hybrid import get_retriever


class ToolError(Exception):
    """
    A tool could not run, or its arguments were invalid.

    Messages are written so a MODEL can read one and self-correct on the next
    turn ("no account found for X" → try a different email), which is why they
    name the tool and the specific problem.
    """


class Tool:
    """Everything we know about one tool: how to describe it, and how to run it."""

    def __init__(self, name: str, description: str, parameters: dict,
                 fn: Callable, privileged: bool = False, tags: list[str] | None = None):
        self.name = name                  # what the model calls it
        self.description = description    # tells the model when to use it
        self.parameters = parameters      # JSON Schema for the arguments
        self.fn = fn                      # the Python function we actually run
        self.privileged = privileged      # True = can propose money-moving actions
        self.tags = tags or []

    def schema(self) -> dict:
        """The shape a model API expects — passable straight to Anthropic/OpenAI."""
        return {"name": self.name, "description": self.description,
                "input_schema": self.parameters}


# ── hand-written argument validation ──────────────────────────────
# Deliberately NOT the `jsonschema` library. This is a trust boundary; writing
# the checks by hand keeps the dependency list small and forces us to know
# exactly what we accept. It is fifty lines.

_TYPE_MAP = {
    "string": str, "integer": int, "number": (int, float),
    "boolean": bool, "array": list, "object": dict,
}


def validate_args(tool: Tool, args: dict) -> dict:
    """
    Check `args` against `tool.parameters`. Raises ToolError on the first problem.

    Rejects unknown arguments, missing required ones, wrong types, out-of-range
    numbers, over-long strings, values outside an enum, and strings failing a
    regex pattern.
    """
    schema = tool.parameters or {}
    properties = schema.get("properties", {})

    if not isinstance(args, dict):
        raise ToolError(f"{tool.name}: arguments must be an object")

    # The model may only pass arguments we declared. An unknown key is either
    # a hallucination or an attempt to reach a parameter we did not expose.
    for key in args:
        if key not in properties:
            raise ToolError(f"{tool.name}: unknown argument '{key}'")

    for key in schema.get("required", []):
        if key not in args:
            raise ToolError(f"{tool.name}: missing required argument '{key}'")

    for key, spec in properties.items():
        if key not in args:
            continue
        value = args[key]
        expected = spec.get("type")

        if expected in _TYPE_MAP:
            # In Python bool is a subclass of int, so True would pass as an
            # integer without this guard.
            if expected == "integer" and isinstance(value, bool):
                raise ToolError(f"{tool.name}: '{key}' must be an integer, not a boolean")
            if not isinstance(value, _TYPE_MAP[expected]):
                raise ToolError(f"{tool.name}: '{key}' must be of type {expected}")

        if "enum" in spec and value not in spec["enum"]:
            raise ToolError(
                f"{tool.name}: '{key}' must be one of {spec['enum']}, got '{value}'")

        if expected in ("integer", "number") and not isinstance(value, bool):
            if "minimum" in spec and value < spec["minimum"]:
                raise ToolError(f"{tool.name}: '{key}' must be >= {spec['minimum']}")
            if "maximum" in spec and value > spec["maximum"]:
                raise ToolError(f"{tool.name}: '{key}' must be <= {spec['maximum']}")

        if expected == "string" and isinstance(value, str):
            if "maxLength" in spec and len(value) > spec["maxLength"]:
                raise ToolError(f"{tool.name}: '{key}' exceeds maxLength {spec['maxLength']}")
            if "minLength" in spec and len(value) < spec["minLength"]:
                raise ToolError(
                    f"{tool.name}: '{key}' is shorter than minLength {spec['minLength']}")
            if "pattern" in spec and not re.search(spec["pattern"], value):
                raise ToolError(f"{tool.name}: '{key}' does not match required format")

    return args


# ═══════════════════════════════════════════════════════════════════
#  THE SIX TOOLS
# ═══════════════════════════════════════════════════════════════════

_EMAIL_RE = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


def _row_to_dict(row):
    return dict(row) if row is not None else None


# ── 1. search_knowledge_base — the RAG entry point ────────────────
def search_knowledge_base(query: str, limit: int | None = None) -> dict:
    """
    Retrieve knowledge-base evidence for a query using Hybrid RAG.

    This function is a thin ADAPTER, and that is the whole point. It delegates
    to `HybridRetriever` and reshapes the result into evidence dicts. It
    contains no ranking logic of its own, so there is exactly one
    implementation of retrieval in this repository.

    Returns:
        {
          "query": "...",
          "evidence": [ {chunk_id, article_id, title, section, text, url,
                         score, retrieval_methods}, ... ],
          "retrieval": { the full RetrievalTrace as a dict }
        }

    Note what is absent: no FAISS handle, no vector, no BM25 object. The agent
    could not reach the index internals even if it tried.
    """
    settings = get_settings()

    if not settings.rag_enabled:
        raise ToolError("search_knowledge_base: retrieval is disabled (RAG_ENABLED=false)")
    if not (query or "").strip():
        raise ToolError("search_knowledge_base: query must not be empty")

    top_k = limit or settings.top_k_final
    evidence, trace = get_retriever(settings).retrieve(query, top_k=top_k)

    return {
        "query": query,
        "evidence": [item.to_dict() for item in evidence],
        "retrieval": trace.to_dict(),
    }


# ── 2. lookup_account ─────────────────────────────────────────────
def lookup_account(email: str) -> dict:
    """Return the account for an email, or raise if there is none."""
    if not re.search(_EMAIL_RE, email):
        raise ToolError("lookup_account: invalid email format")
    row = db.query_one("SELECT * FROM accounts WHERE contact_email=?", (email,))
    if row is None:
        raise ToolError(f"lookup_account: no account found for {email}")
    return _row_to_dict(row)


# ── 3. get_billing_history ────────────────────────────────────────
def get_billing_history(account_id: str, limit: int = 5) -> dict:
    """Return recent invoices for an account."""
    if db.query_one("SELECT id FROM accounts WHERE id=?", (account_id,)) is None:
        raise ToolError(f"get_billing_history: unknown account {account_id}")
    rows = db.query(
        "SELECT * FROM invoices WHERE account_id=? ORDER BY issued_at DESC, id LIMIT ?",
        (account_id, limit))
    return {"account_id": account_id, "invoices": [_row_to_dict(r) for r in rows]}


# ── 4. check_service_status ───────────────────────────────────────
def check_service_status(component: str) -> dict:
    """Return the current health of one product component."""
    row = db.query_one("SELECT * FROM service_status WHERE component=?", (component,))
    if row is None:
        raise ToolError(f"check_service_status: unknown component {component}")
    return _row_to_dict(row)


# ── 5. issue_refund  (PRIVILEGED — NEVER WRITES) ──────────────────
def issue_refund(account_id: str, invoice_id: str,
                 amount_cents: int, reason: str) -> dict:
    """
    Validate a refund and return a PROPOSAL. Never touches the database.

    This is the most important safety property in the project. A tool that can
    move money is a tool an attacker will aim at your customers, so the agent
    is given no way to move money at all — only to ask.

    The checks below are CODE-LEVEL AUTHORISATION. A retrieved policy document
    saying "growth-plan customers get automatic refunds" does not make one
    legal; these four checks against the live database do:
      - the account exists and is refund-eligible
      - the invoice belongs to that account
      - the invoice is not already refunded
      - the amount does not exceed the invoice total
    """
    account = db.query_one("SELECT * FROM accounts WHERE id=?", (account_id,))
    if account is None:
        raise ToolError(f"issue_refund: unknown account {account_id}")
    if not account["refund_eligible"]:
        raise ToolError(f"issue_refund: account {account_id} is not refund-eligible")

    invoice = db.query_one("SELECT * FROM invoices WHERE id=?", (invoice_id,))
    if invoice is None:
        raise ToolError(f"issue_refund: unknown invoice {invoice_id}")
    if invoice["account_id"] != account_id:
        raise ToolError("issue_refund: invoice does not belong to that account")
    if invoice["status"] == "refunded":
        raise ToolError("issue_refund: invoice is already refunded")
    if amount_cents > invoice["amount_cents"]:
        raise ToolError("issue_refund: amount exceeds the invoice total")

    # No INSERT. No UPDATE. Only a proposal object.
    return {
        "action_type": "issue_refund",
        "account_id": account_id,
        "invoice_id": invoice_id,
        "amount_cents": amount_cents,
        "reason": reason,
        "requires_human_approval": True,
    }


# ── 6. escalate_to_human ──────────────────────────────────────────
def escalate_to_human(reason: str, priority: str = "normal") -> dict:
    """Hand the ticket to a person, with a structured reason."""
    return {"action_type": "escalate_to_human", "reason": reason, "priority": priority}


# ═══════════════════════════════════════════════════════════════════
#  REGISTRY
# ═══════════════════════════════════════════════════════════════════
REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> None:
    REGISTRY[tool.name] = tool


register(Tool(
    name="search_knowledge_base",
    description=("Search the help knowledge base for passages relevant to a query, "
                 "using hybrid keyword + semantic retrieval. Call this before "
                 "explaining how the product works, and cite the chunk ids it returns."),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 300},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    },
    fn=search_knowledge_base,
    tags=["retrieval"],
))

register(Tool(
    name="lookup_account",
    description=("Look up a customer account by its contact email address. "
                 "Call this before stating any fact about an account."),
    parameters={
        "type": "object",
        "properties": {"email": {"type": "string", "pattern": _EMAIL_RE, "maxLength": 254}},
        "required": ["email"],
    },
    fn=lookup_account,
    tags=["account"],
))

register(Tool(
    name="get_billing_history",
    description=("Get recent invoices for an account. Call before discussing a "
                 "specific charge or proposing a refund."),
    parameters={
        "type": "object",
        "properties": {
            "account_id": {"type": "string", "maxLength": 40},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": ["account_id"],
    },
    fn=get_billing_history,
    tags=["billing"],
))

register(Tool(
    name="check_service_status",
    description=("Check whether a product component is operational, degraded, or in "
                 "an outage. Call before blaming the customer's setup."),
    parameters={
        "type": "object",
        "properties": {
            "component": {"type": "string",
                          "enum": ["api", "dashboard", "webhooks", "exports", "auth"]},
        },
        "required": ["component"],
    },
    fn=check_service_status,
    tags=["status"],
))

register(Tool(
    name="issue_refund",
    description=("Propose a refund for an invoice. This does NOT execute the refund; "
                 "it returns a proposal that a human must approve."),
    parameters={
        "type": "object",
        "properties": {
            "account_id": {"type": "string", "maxLength": 40},
            "invoice_id": {"type": "string", "maxLength": 40},
            "amount_cents": {"type": "integer", "minimum": 1, "maximum": 100000000},
            "reason": {"type": "string", "minLength": 1, "maxLength": 500},
        },
        "required": ["account_id", "invoice_id", "amount_cents", "reason"],
    },
    fn=issue_refund,
    privileged=True,          # ← the human-approval gate keys off this flag
    tags=["billing", "privileged"],
))

register(Tool(
    name="escalate_to_human",
    description="Escalate the ticket to a human support agent with a reason and priority.",
    parameters={
        "type": "object",
        "properties": {
            "reason": {"type": "string", "minLength": 1, "maxLength": 500},
            "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
        },
        "required": ["reason"],
    },
    fn=escalate_to_human,
    tags=["escalation"],
))


def tool_schemas() -> list[dict]:
    """Every tool's schema — this is what we hand to the model."""
    return [tool.schema() for tool in REGISTRY.values()]


def call_tool(name: str, args: dict) -> tuple[dict, int]:
    """
    Validate arguments, then execute the named tool. Returns `(result, latency_ms)`.

    The SINGLE entry point the agent uses. Centralising it is what makes
    "validation can never be skipped" true by construction rather than by
    convention.
    """
    tool = REGISTRY.get(name)
    if tool is None:
        raise ToolError(f"unknown tool: {name}")

    validate_args(tool, args)

    started = time.perf_counter()
    result = tool.fn(**args)
    return result, int((time.perf_counter() - started) * 1000)
