// format.test.ts — unit tests for the pure display-decision functions.
// Run with: npm test
import { describe, expect, it } from "vitest";
import {
  confidenceBand, money, ms, retrievalHealth, reviewAction, titleise,
} from "./format";
import type { RetrievalTrace, Run, TriageResult } from "./types";

const result = (over: Partial<TriageResult> = {}): TriageResult => ({
  intent: "how_to", priority: "normal", sentiment: "neutral",
  summary: "s", suggested_reply: "a helpful reply here",
  citations: [], proposed_actions: [], requires_human: false, confidence: 0.9,
  ...over,
});

const run = (over: Partial<Run> = {}): Run => ({
  ticket_id: "tkt_1", status: "completed", result: result(),
  trace: [], approval_id: null, ...over,
});

const trace = (over: Partial<RetrievalTrace> = {}): RetrievalTrace => ({
  query: "q", mode_requested: "hybrid", mode_used: "hybrid",
  lexical_candidates: 10, semantic_candidates: 10, fusion_method: "rrf",
  reranker: null, final_k: 4, fallbacks: [], top_chunk_ids: [], ...over,
});

describe("formatting", () => {
  it("formats money from cents", () => expect(money(49900)).toBe("$499.00"));
  it("formats short durations in ms", () => expect(ms(32)).toBe("32 ms"));
  it("formats long durations in seconds", () => expect(ms(1200)).toBe("1.2 s"));
  it("shows a dash for a missing duration", () => expect(ms(null)).toBe("—"));
  it("titleises snake_case", () => expect(titleise("billing_refund")).toBe("Billing refund"));
});

describe("reviewAction", () => {
  it("allows sending a clean completed run", () => {
    expect(reviewAction(run()).tone).toBe("send");
  });

  it("blocks a failed run", () => {
    expect(reviewAction(run({ status: "failed" })).tone).toBe("block");
  });

  it("blocks a run with a direct injection", () => {
    expect(reviewAction(run({ injection: { flagged: true, categories: [] } })).tone)
      .toBe("block");
  });

  it("blocks a run whose retrieved evidence was poisoned", () => {
    // Indirect injection — the RAG-specific risk must be visible in the UI too.
    expect(reviewAction(run({
      evidence_injection: { flagged: true, chunk_ids: ["kb_evil#01"] },
    })).tone).toBe("block");
  });

  it("blocks a run that cited evidence it never retrieved", () => {
    expect(reviewAction(run({
      citations: { valid: [], invalid: ["kb_999#07"], grounded: false,
                   coverage: 0, has_fabrication: true },
    })).label).toContain("citation");
  });

  it("routes a pending approval to a human", () => {
    expect(reviewAction(run({ approval_id: "apr_1" })).tone).toBe("review");
  });

  it("routes requires_human to a human", () => {
    expect(reviewAction(run({ result: result({ requires_human: true }) })).tone)
      .toBe("review");
  });
});

describe("confidenceBand", () => {
  it("bands high confidence", () =>
    expect(confidenceBand(result({ confidence: 0.9 })).label).toBe("High"));
  it("bands medium confidence", () =>
    expect(confidenceBand(result({ confidence: 0.6 })).label).toBe("Medium"));
  it("bands low confidence", () =>
    expect(confidenceBand(result({ confidence: 0.2 })).label).toBe("Low"));
});

describe("retrievalHealth", () => {
  it("reports a healthy hybrid run", () => {
    expect(retrievalHealth(trace()).tone).toBe("send");
  });

  it("flags a fallback so a degraded pipeline is never invisible", () => {
    const degraded = trace({
      mode_used: "lexical", fallbacks: ["vector_index_unavailable"],
    });
    expect(retrievalHealth(degraded).tone).toBe("review");
    expect(retrievalHealth(degraded).label).toContain("lexical");
  });

  it("flags retrieval that returned nothing", () => {
    expect(retrievalHealth(trace({ mode_used: "none" })).tone).toBe("block");
  });
});
