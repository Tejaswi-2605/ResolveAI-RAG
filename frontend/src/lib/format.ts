// format.ts — display helpers plus three PURE decision functions.
//
// Keeping the decision logic pure means it can be unit-tested (format.test.ts)
// without rendering any React. A reviewer-facing verdict like "do not send" is
// worth testing properly.

import type { Run, RetrievalTrace, TriageResult } from "./types";

// cents (integer) → "$49.90"
export function money(cents: number): string {
  return `$${(cents / 100).toFixed(2)}`;
}

// milliseconds → "32 ms" or "1.2 s"
export function ms(value: number | null | undefined): string {
  if (value == null) return "—";
  return value < 1000 ? `${value} ms` : `${(value / 1000).toFixed(1)} s`;
}

// "billing_refund" → "Billing refund"
export function titleise(text: string): string {
  const s = (text ?? "").replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export type Tone = "block" | "review" | "send";

// reviewAction — the coloured verdict card for a run.
//   block  = failed, injection flagged, or a fabricated citation → DO NOT SEND
//   review = a human must look (pending approval or requires_human)
//   send   = safe to send as-is
export function reviewAction(run: Run): { label: string; tone: Tone } {
  const injected =
    run.injection?.flagged === true ||
    run.injection_flagged === true ||
    run.evidence_injection?.flagged === true;

  if (run.status === "failed" || injected) {
    return { label: "Blocked — do not send", tone: "block" };
  }
  // A fabricated citation means the answer's provenance is untrustworthy, even
  // if everything else about the run looks fine.
  if (run.citations?.has_fabrication) {
    return { label: "Blocked — citation not in evidence", tone: "block" };
  }
  if (run.approval_id != null || run.result?.requires_human === true) {
    return { label: "Needs human review", tone: "review" };
  }
  return { label: "Ready to send", tone: "send" };
}

// confidenceBand — bucket a result's confidence for display.
export function confidenceBand(result: TriageResult): { label: string; tone: Tone } {
  const c = result?.confidence ?? 0;
  if (c >= 0.75) return { label: "High", tone: "send" };
  if (c >= 0.5) return { label: "Medium", tone: "review" };
  return { label: "Low", tone: "block" };
}

// retrievalHealth — did retrieval do what it claimed?
//   send   = ran in the requested mode
//   review = degraded to a fallback (still answered, but with one arm)
//   block  = retrieval produced nothing at all
export function retrievalHealth(trace: RetrievalTrace): { label: string; tone: Tone } {
  if (!trace.mode_used || trace.mode_used === "none") {
    return { label: "No results", tone: "block" };
  }
  if (trace.fallbacks?.length || trace.mode_used !== trace.mode_requested) {
    return { label: `Degraded → ${trace.mode_used}`, tone: "review" };
  }
  return { label: trace.mode_used, tone: "send" };
}
