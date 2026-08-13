// api.ts — the typed HTTP client. Every network call goes through request<T>.
//
// It turns any non-2xx response into an ApiError carrying FastAPI's "detail"
// message, so the console can show a real reason ("approval already executed")
// instead of a generic failure.

import type {
  Approval, EvalReport, IndexStatus, Run, SearchResult, Stats, Ticket, ToolCall,
} from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText; // FastAPI errors look like { "detail": "..." }
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* no JSON body — keep the status text */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// The triage endpoint returns `tool`, the stored-run endpoint returns
// `tool_name`. Normalise once here so no component has to know the difference.
function normaliseRun(run: any): Run {
  const trace: ToolCall[] = (run.trace ?? []).map((t: any) => ({
    step: t.step ?? null,
    tool: t.tool ?? t.tool_name,
    args: t.args ?? {},
    ok: t.ok ?? null,
    result: t.result ?? null,
    error: t.error ?? null,
    latency_ms: t.latency_ms ?? null,
  }));
  return { ...run, trace };
}

export const api = {
  tickets: (status?: string) =>
    request<Ticket[]>(`/api/tickets${status ? `?status=${status}` : ""}`),

  ticket: (id: string) => request<Ticket>(`/api/tickets/${id}`),

  createTicket: (payload: {
    sender_email: string; subject: string; body: string; channel?: string;
  }) => request<Ticket>("/api/tickets", {
    method: "POST", body: JSON.stringify(payload),
  }),

  triage: async (id: string, prompt_version: string) =>
    normaliseRun(await request<any>(`/api/tickets/${id}/triage`, {
      method: "POST", body: JSON.stringify({ prompt_version }),
    })),

  runsForTicket: async (id: string) =>
    (await request<any[]>(`/api/tickets/${id}/runs`)).map(normaliseRun),

  approvals: (state = "pending") =>
    request<Approval[]>(`/api/approvals?state=${state}`),

  decide: (id: string, decision: "approved" | "rejected", decided_by: string) =>
    request<{ approval_id: string; state: string }>(
      `/api/approvals/${id}/decision`,
      { method: "POST", body: JSON.stringify({ decision, decided_by }) }),

  search: (q: string, limit = 4) =>
    request<SearchResult>(`/api/kb/search?q=${encodeURIComponent(q)}&limit=${limit}`),

  ragStatus: () => request<IndexStatus>("/api/rag/status"),

  stats: () => request<Stats>("/api/stats"),

  evals: () => request<EvalReport>("/api/eval/latest"),
};
