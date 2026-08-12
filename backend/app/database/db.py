"""
db.py — THE ONLY MODULE THAT TALKS TO SQLITE.

Nothing else in the project imports `sqlite3`. Funnelling every query through
one file gives three concrete benefits:

  1. Every statement is PARAMETERISED (`?` placeholders). SQL injection is
     impossible by construction because no query is ever built by string
     concatenation with user data.
  2. Transaction handling is written once — commit on success, roll back on
     error, always close.
  3. Swapping SQLite for Postgres later means editing one file.

Why raw `sqlite3` and not an ORM like SQLAlchemy? The data model is nine
small tables with explicit, hand-written SQL. An ORM earns its complexity
when you have many tables, migrations and a large team; here it would hide
the one thing a reviewer most wants to read — the actual queries.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def _db_path() -> str:
    """
    Resolve the database file path AT CALL TIME, never at import time.

    This matters: pytest sets DATABASE_PATH to a throwaway file *after* this
    module has already been imported. Reading the setting lazily is exactly
    what gives each test its own isolated database.
    """
    return str(get_settings().db_file)


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string — the format every *_at column uses."""
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    """Generate a short readable id such as `run_a1b2c3d4`."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@contextmanager
def connect():
    """
    Yield a configured SQLite connection, then clean up correctly.

    On the way in:
      * `row_factory = sqlite3.Row` so rows behave like dicts (`row["title"]`).
      * `PRAGMA foreign_keys = ON` — SQLite disables FK enforcement by default,
        and the pragma is per-connection, so it must be set every time.

    On the way out:
      * COMMIT if the block finished cleanly, ROLLBACK if it raised, and
        CLOSE either way. That is a transaction: all of the changes, or none.
    """
    path = _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create every table and index by running schema.sql. Safe to re-run."""
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    with connect() as conn:
        conn.executescript(sql)


# ── generic query helpers ─────────────────────────────────────────

def query(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    """Run a SELECT and return every matching row (possibly an empty list)."""
    with connect() as conn:
        return conn.execute(sql, params).fetchall()


def query_one(sql: str, params: tuple = ()) -> sqlite3.Row | None:
    """Run a SELECT and return the first row, or None if nothing matched."""
    with connect() as conn:
        return conn.execute(sql, params).fetchone()


def execute(sql: str, params: tuple = ()) -> None:
    """Run an INSERT/UPDATE/DELETE inside a transaction."""
    with connect() as conn:
        conn.execute(sql, params)


# ── knowledge base ────────────────────────────────────────────────

def all_kb_articles() -> list[dict]:
    """
    Return every knowledge-base article as a plain dict, ordered by id.

    This is the RAG pipeline's ONLY entry point into the knowledge base. The
    ordering is fixed so that ingestion is deterministic: the same database
    always produces the same chunks in the same order.
    """
    rows = query(
        "SELECT id, title, body, tags, url, product_area, updated_at "
        "FROM kb_articles ORDER BY id"
    )
    return [dict(r) for r in rows]


# ── agent run lifecycle ───────────────────────────────────────────

def start_run(run_id: str, ticket_id: str, prompt_version: str, provider: str,
              model: str, rag_mode: str, injection_flagged: bool) -> None:
    """
    Insert the `agent_runs` row with status 'running' BEFORE the agent loop starts.

    Why before? `tool_calls.run_id` and `retrievals.run_id` are foreign keys
    pointing at `agent_runs.id`. A child row cannot be written until its parent
    exists, so logging the very first tool call would fail with
    `sqlite3.IntegrityError: FOREIGN KEY constraint failed`. Creating the run
    first also means a crash mid-loop leaves a visible 'running' record rather
    than no evidence at all.
    """
    execute(
        """INSERT INTO agent_runs
           (id, ticket_id, prompt_version, provider, model, rag_mode, status,
            injection_flagged, created_at)
           VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?)""",
        (run_id, ticket_id, prompt_version, provider, model, rag_mode,
         1 if injection_flagged else 0, now_iso()),
    )


def finalize_run(run_id: str, status: str, result_json: str | None,
                 error: str | None, steps_used: int, latency_ms: int,
                 input_tokens: int, output_tokens: int,
                 citations_valid: bool | None, rag_mode: str | None) -> None:
    """Write the final outcome and metrics onto the run row."""
    execute(
        """UPDATE agent_runs
           SET status=?, result_json=?, error=?, steps_used=?, latency_ms=?,
               input_tokens=?, output_tokens=?, citations_valid=?, rag_mode=?
           WHERE id=?""",
        (status, result_json, error, steps_used, latency_ms, input_tokens,
         output_tokens,
         None if citations_valid is None else (1 if citations_valid else 0),
         rag_mode, run_id),
    )


def log_tool_call(run_id: str, step: int, tool_name: str, args_json: str,
                  ok: bool, result_json: str | None, error: str | None,
                  latency_ms: int) -> None:
    """Record one tool invocation so the trace rail can replay the run."""
    execute(
        """INSERT INTO tool_calls
           (run_id, step, tool_name, args_json, ok, result_json, error,
            latency_ms, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, step, tool_name, args_json, 1 if ok else 0, result_json,
         error, latency_ms, now_iso()),
    )


def log_retrieval(run_id: str | None, trace: dict) -> None:
    """
    Persist one retrieval trace (see `app.rag.models.RetrievalTrace`).

    Stored separately from `tool_calls` because a retrieval has its own
    vocabulary — candidate counts per arm, fusion method, fallbacks — and we
    want to query it independently (e.g. "how often did semantic search fall
    back last week?").
    """
    import json

    execute(
        """INSERT INTO retrievals
           (run_id, query, mode_requested, mode_used, lexical_candidates,
            semantic_candidates, fusion_method, reranker, final_k, fallbacks,
            top_chunk_ids, latency_ms, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, trace.get("query", ""), trace.get("mode_requested"),
         trace.get("mode_used"), trace.get("lexical_candidates"),
         trace.get("semantic_candidates"), trace.get("fusion_method"),
         trace.get("reranker"), trace.get("final_k"),
         json.dumps(trace.get("fallbacks", [])),
         json.dumps(trace.get("top_chunk_ids", [])),
         trace.get("latency_ms"), now_iso()),
    )


def create_approval(approval_id: str, run_id: str, ticket_id: str,
                    action_type: str, payload_json: str, rationale: str) -> None:
    """
    Create a 'pending' approval — a REQUEST for a human decision.

    Nothing is executed here. This row is how a proposed refund waits for a
    person; the money only moves in the approval endpoint.
    """
    execute(
        """INSERT INTO approvals
           (id, run_id, ticket_id, action_type, payload_json, rationale,
            state, created_at)
           VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
        (approval_id, run_id, ticket_id, action_type, payload_json,
         rationale, now_iso()),
    )
