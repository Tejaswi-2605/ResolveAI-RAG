-- ═══════════════════════════════════════════════════════════════════
-- schema.sql — the structure of the ResolveAI-RAG SQLite database.
--
-- THE KEY IDEA: this database is the SOURCE OF TRUTH.
-- `kb_articles` holds the authoritative knowledge. Chunks, embeddings and
-- the vector index are DERIVED artifacts built from these rows — they live
-- on disk under data/index/, are git-ignored, and can be rebuilt at any
-- time with `python -m app.rag.ingest`. Deleting the index loses nothing.
--
-- Run once at startup by database.db.init_db(). Every statement uses
-- "IF NOT EXISTS", so running it repeatedly is safe.
-- ═══════════════════════════════════════════════════════════════════

-- SQLite ships with foreign-key enforcement OFF. Turn it on, or a child row
-- could happily point at a parent that does not exist.
PRAGMA foreign_keys = ON;


-- ── accounts ──────────────────────────────────────────────────────
-- The customer companies who file support tickets.
CREATE TABLE IF NOT EXISTS accounts (
    id              TEXT PRIMARY KEY,              -- e.g. "acct_001"
    company         TEXT NOT NULL,
    contact_email   TEXT UNIQUE NOT NULL,          -- one account per email
    -- CHECK = the database itself rejects any value outside this set.
    plan            TEXT NOT NULL CHECK (plan IN ('trial','starter','growth','enterprise')),
    seats           INTEGER NOT NULL,
    mrr_cents       INTEGER NOT NULL,              -- monthly recurring revenue, in cents
    status          TEXT NOT NULL CHECK (status IN ('active','past_due','churned')),
    refund_eligible INTEGER NOT NULL,              -- 0 or 1 (SQLite has no real boolean)
    created_at      TEXT NOT NULL                  -- ISO-8601 timestamp
);


-- ── invoices ──────────────────────────────────────────────────────
-- Bills sent to accounts. This is the ONLY money-bearing table, which is
-- what makes "only one code path may write to it" an auditable claim.
CREATE TABLE IF NOT EXISTS invoices (
    id           TEXT PRIMARY KEY,
    account_id   TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    amount_cents INTEGER NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('paid','open','failed','refunded')),
    issued_at    TEXT NOT NULL,
    description  TEXT
);


-- ── kb_articles ───────────────────────────────────────────────────
-- THE KNOWLEDGE BASE — the authoritative source the RAG pipeline indexes.
--
-- `body` is markdown-ish: "## Section heading" lines separate sections, and
-- blank lines separate paragraphs. The chunker understands both, which is
-- how a retrieved chunk can say WHICH SECTION of an article it came from.
CREATE TABLE IF NOT EXISTS kb_articles (
    id           TEXT PRIMARY KEY,                 -- e.g. "kb_001" — cited by the agent
    title        TEXT NOT NULL,
    body         TEXT NOT NULL,
    tags         TEXT,                             -- space-separated keywords
    url          TEXT,                             -- link a human can follow
    product_area TEXT,                             -- e.g. "billing", "api"
    updated_at   TEXT                              -- feeds the index staleness check
);


-- ── service_status ────────────────────────────────────────────────
-- Current health of each product component (for outage questions).
CREATE TABLE IF NOT EXISTS service_status (
    component  TEXT PRIMARY KEY,
    state      TEXT NOT NULL CHECK (state IN ('operational','degraded','outage')),
    detail     TEXT,
    updated_at TEXT NOT NULL
);


-- ── tickets ───────────────────────────────────────────────────────
-- The incoming support requests. `body` is UNTRUSTED text: an attacker
-- controls it, so it is fenced before it ever reaches the model.
CREATE TABLE IF NOT EXISTS tickets (
    id           TEXT PRIMARY KEY,
    account_id   TEXT REFERENCES accounts(id),     -- nullable: sender may be unknown
    sender_email TEXT NOT NULL,
    subject      TEXT NOT NULL,
    body         TEXT NOT NULL,
    channel      TEXT,
    status       TEXT NOT NULL CHECK (status IN ('new','triaged','awaiting_approval','escalated','closed')),
    created_at   TEXT NOT NULL
);


-- ── agent_runs ────────────────────────────────────────────────────
-- One row per agent execution: the audit record. Which prompt, which model,
-- which retrieval mode, what it decided, and whether the citations checked out.
CREATE TABLE IF NOT EXISTS agent_runs (
    id                TEXT PRIMARY KEY,
    ticket_id         TEXT NOT NULL REFERENCES tickets(id),
    prompt_version    TEXT,                        -- "v1" or "v2"
    provider          TEXT,                        -- "mock" or "anthropic"
    model             TEXT,
    rag_mode          TEXT,                        -- retrieval mode ACTUALLY used
    status            TEXT NOT NULL CHECK (status IN ('running','completed','failed','escalated')),
    result_json       TEXT,                        -- the validated TriageResult
    error             TEXT,
    steps_used        INTEGER,
    latency_ms        INTEGER,
    input_tokens      INTEGER,
    output_tokens     INTEGER,
    injection_flagged INTEGER,                     -- 1 if injection was detected
    citations_valid   INTEGER,                     -- 1 if every citation matched real evidence
    created_at        TEXT NOT NULL
);


-- ── tool_calls ────────────────────────────────────────────────────
-- One row per tool the agent invoked. Powers the UI trace rail and makes
-- every action the agent took reviewable after the fact.
CREATE TABLE IF NOT EXISTS tool_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES agent_runs(id),
    step        INTEGER,
    tool_name   TEXT NOT NULL,
    args_json   TEXT,
    ok          INTEGER,                           -- 1 = succeeded, 0 = tool error
    result_json TEXT,
    error       TEXT,
    latency_ms  INTEGER,
    created_at  TEXT NOT NULL
);


-- ── retrievals ────────────────────────────────────────────────────
-- OBSERVABILITY FOR THE RAG PIPELINE. One row per knowledge-tool call,
-- recording how the evidence was actually produced: how many candidates each
-- retrieval arm returned, whether a fallback fired, and which chunks won.
--
-- This is what turns "the agent used RAG" into an inspectable claim. Note it
-- stores the QUERY and CHUNK IDS only — never customer PII.
CREATE TABLE IF NOT EXISTS retrievals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT REFERENCES agent_runs(id),
    query               TEXT NOT NULL,
    mode_requested      TEXT,                      -- what config asked for
    mode_used           TEXT,                      -- what actually ran (may differ: fallback)
    lexical_candidates  INTEGER,
    semantic_candidates INTEGER,
    fusion_method       TEXT,                      -- "rrf" or "none"
    reranker            TEXT,
    final_k             INTEGER,
    fallbacks           TEXT,                      -- JSON array of fallback reasons
    top_chunk_ids       TEXT,                      -- JSON array of the winning chunk ids
    latency_ms          INTEGER,
    created_at          TEXT NOT NULL
);


-- ── approvals ─────────────────────────────────────────────────────
-- The human-in-the-loop gate. When the agent PROPOSES a money-moving action
-- it creates one of these instead of acting. A human decides later.
CREATE TABLE IF NOT EXISTS approvals (
    id           TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES agent_runs(id),
    ticket_id    TEXT NOT NULL REFERENCES tickets(id),
    action_type  TEXT NOT NULL,                    -- e.g. "issue_refund"
    payload_json TEXT,                             -- the proposed action's details
    rationale    TEXT,
    state        TEXT NOT NULL CHECK (state IN ('pending','approved','rejected','executed')),
    decided_by   TEXT,
    decided_at   TEXT,
    created_at   TEXT NOT NULL
);


-- ── indexes ───────────────────────────────────────────────────────
-- Sorted shortcuts on the columns we filter and join on most often.
CREATE INDEX IF NOT EXISTS idx_tickets_status     ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_tool_calls_run     ON tool_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_ticket  ON agent_runs(ticket_id);
CREATE INDEX IF NOT EXISTS idx_approvals_state    ON approvals(state);
CREATE INDEX IF NOT EXISTS idx_retrievals_run     ON retrievals(run_id);
