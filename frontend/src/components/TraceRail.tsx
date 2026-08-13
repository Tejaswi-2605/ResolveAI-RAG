// TraceRail.tsx — a NUMBERED timeline of every tool call the agent made.
//
// Numbering matters because the ORDER is what you debug: "it proposed a refund
// before checking the invoice" is only visible if the sequence is visible. A
// support agent will not trust a draft they cannot audit; this is that audit
// trail.
import type { Run, ToolCall } from "../lib/types";
import { ms } from "../lib/format";

export function TraceRail({ run }: { run: Run | null }) {
  const trace: ToolCall[] = run?.trace ?? [];

  if (!trace.length) {
    return <p className="muted">No tool calls yet. Run triage to see the trace.</p>;
  }

  return (
    <div>
      {run?.stages && (
        <div className="stages">
          {run.stages.map((s) => <span className="chip" key={s}>{s}</span>)}
        </div>
      )}
      {trace.map((step, i) => (
        <div className="trace-step" key={i}>
          <div className={`trace-num ${step.ok === false ? "err" : ""}`}>{i + 1}</div>
          <div className="trace-body">
            <span className="lat">{ms(step.latency_ms)}</span>
            <span className="tool">{step.tool}</span>
            <pre>{JSON.stringify(step.args, null, 2)}</pre>
            {step.ok === false
              ? <pre className="error">{step.error}</pre>
              : step.result && <pre>{JSON.stringify(step.result, null, 2)}</pre>}
          </div>
        </div>
      ))}
    </div>
  );
}
