// EvalDashboard.tsx — the EVALUATIONS tab.
//
// Reads /api/eval/latest and shows the metric table, highlighting the better
// column per metric. The direction map mirrors the backend's HIGHER_IS_BETTER,
// so "better" is declared rather than guessed from a metric's name.
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { EvalReport } from "../lib/types";

const HIGHER_IS_BETTER: Record<string, boolean> = {
  intent_accuracy: true, escalation_accuracy: true, escalation_precision: true,
  escalation_recall: true, tool_recall: true, forbidden_tool_rate: false,
  structured_output_validity: true, retrieval_used_rate: true,
  grounded_citation_rate: true, citation_correctness: true,
  retrieval_mode_integrity: true, injection_defence_rate: true,
  approval_correctness: true, unsupported_claim_rate: false, error_rate: false,
  avg_latency_ms: false, p95_latency_ms: false, avg_steps: false,
};

export function EvalDashboard() {
  const [report, setReport] = useState<EvalReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.evals().then(setReport).catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="muted">No eval results available. ({error})</p>;
  if (!report) return <p className="muted">Loading evaluations…</p>;

  // Only the agent reports share a metric shape; retrieval.json is structured
  // differently and has its own report file.
  const versions = Object.keys(report).filter((k) => k.startsWith("agent_")).sort();
  if (!versions.length) return <p className="muted">No agent eval results found.</p>;

  const metrics = Object.keys(report[versions[0]].metrics);

  return (
    <>
      <p className="muted">
        Measured with the deterministic mock provider: these grade the SYSTEM
        (tools, trust boundary, validation, citation check, approval gate), not a
        frontier model's intelligence.
      </p>
      <table className="eval">
        <thead>
          <tr>
            <th>Metric</th>
            {versions.map((v) => <th key={v}>{v.replace("agent_", "")}</th>)}
          </tr>
        </thead>
        <tbody>
          {metrics.map((m) => {
            const values = versions.map((v) => report[v].metrics[m]);
            const higher = HIGHER_IS_BETTER[m] ?? true;
            const best = higher ? Math.max(...values) : Math.min(...values);
            const allEqual = values.every((x) => x === values[0]);
            return (
              <tr key={m}>
                <td>{m}</td>
                {versions.map((v) => {
                  const value = report[v].metrics[m];
                  return (
                    <td key={v} className={!allEqual && value === best ? "better" : ""}>
                      {value}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </>
  );
}
