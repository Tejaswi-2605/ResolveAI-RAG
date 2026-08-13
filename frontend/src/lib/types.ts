// types.ts — the TypeScript shapes mirroring the backend's JSON.
//
// TypeScript checks these at compile time, so if the UI reads a field the
// backend does not send, the BUILD fails instead of the page breaking in front
// of a support agent.

export interface Ticket {
  id: string;
  account_id: string | null;
  sender_email: string;
  subject: string;
  body: string;
  channel: string | null;
  status: string;
  created_at: string;
}

// One source card, built strictly from evidence the run actually retrieved.
export interface Source {
  article_id: string;
  title: string;
  chunk_id: string;
  section: string | null;
  url: string | null;
}

export interface TriageResult {
  intent: string;
  priority: string;
  sentiment: string;
  summary: string;
  suggested_reply: string;
  citations: string[];
  sources?: Source[];
  proposed_actions: Array<Record<string, unknown>>;
  requires_human: boolean;
  confidence: number;
  escalation_reason?: string;
}

// One entry in the agent trace rail.
export interface ToolCall {
  step: number | null;
  tool: string; // normalised name (api.ts maps tool_name -> tool)
  args: Record<string, unknown>;
  ok: boolean | null;
  result: Record<string, unknown> | null;
  error: string | null;
  latency_ms: number | null;
}

// The retrieval record: how the evidence was actually produced.
export interface RetrievalTrace {
  query: string;
  mode_requested: string | null;
  mode_used: string | null;
  lexical_candidates: number | null;
  semantic_candidates: number | null;
  fusion_method: string | null;
  reranker: string | null;
  final_k: number | null;
  fallbacks: string[];
  top_chunk_ids: string[];
  latency_ms?: Record<string, number> | number | null;
}

// The citation verdict for a run.
export interface CitationReport {
  valid: string[];
  invalid: string[];
  grounded: boolean;
  coverage: number;
  has_fabrication: boolean;
}

export interface Run {
  run_id?: string;
  id?: string;
  ticket_id: string;
  status: string;
  result: TriageResult;
  trace: ToolCall[];
  stages?: string[];
  retrieval?: RetrievalTrace[];
  citations?: CitationReport;
  evidence?: Array<Record<string, unknown>>;
  rag_mode?: string | null;
  approval_id: string | null;
  tools_used?: string[];
  injection?: { flagged: boolean; categories: string[] };
  evidence_injection?: { flagged: boolean; chunk_ids: string[] };
  injection_flagged?: boolean;
  citations_valid?: boolean | null;
  prompt_version?: string;
  provider?: string;
  model?: string;
  steps_used?: number;
  latency_ms?: number;
  error?: string | null;
}

export interface Approval {
  id: string;
  run_id: string;
  ticket_id: string;
  action_type: string;
  payload: Record<string, unknown> | null;
  rationale: string | null;
  state: string;
  created_at: string;
}

export interface Stats {
  tickets_by_status: Record<string, number>;
  run_count: number;
  escalation_rate: number;
  avg_latency_ms: number;
  pending_approvals: number;
  injection_flagged_runs: number;
  retrieval_count: number;
  hybrid_retrieval_rate: number;
  fallback_retrieval_count: number;
}

export interface IndexStatus {
  exists: boolean;
  stale: boolean | null;
  embedding_model: string | null;
  dimension: number | null;
  vector_backend: string | null;
  articles: number | null;
  chunks: number | null;
  rag_mode: string | null;
  semantic_available: boolean | null;
  startup_fallbacks: string[];
}

// One retrieved chunk from GET /api/kb/search.
export interface Evidence {
  chunk_id: string;
  article_id: string;
  title: string;
  section: string | null;
  text: string;
  url: string | null;
  score: number;
  retrieval_methods: string[];
  ranks: Record<string, number>;
  method_scores: Record<string, number>;
}

export interface SearchResult {
  query: string;
  evidence: Evidence[];
  retrieval: RetrievalTrace;
}

// The eval JSON served by /api/eval/latest, keyed by file stem.
export interface EvalVersion {
  label?: string;
  version?: string;
  metrics: Record<string, number>;
  failures: Array<{ id: string; reasons: string[] }>;
  n_cases: number;
}
export type EvalReport = Record<string, EvalVersion>;
