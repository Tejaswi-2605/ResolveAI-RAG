// TicketDetail.tsx — the centre pane: the ticket, the run controls, the
// coloured reviewer verdict, the approval card, the agent draft, and the
// SOURCES card that makes every claim checkable.
import type { Run, Ticket } from "../lib/types";
import { confidenceBand, money, reviewAction, titleise } from "../lib/format";

export function TicketDetail({
  ticket, run, promptVersion, onVersionChange, onTriage, onDecide, busy,
}: {
  ticket: Ticket;
  run: Run | null;
  promptVersion: string;
  onVersionChange: (v: string) => void;
  onTriage: () => void;
  onDecide: (approvalId: string, decision: "approved" | "rejected") => void;
  busy: boolean;
}) {
  return (
    <div className="pane">
      <div className="card">
        <h3>Ticket · {ticket.id}</h3>
        <div style={{ fontWeight: 600, marginBottom: 4 }}>{ticket.subject}</div>
        <div className="from muted">{ticket.sender_email} · {ticket.channel}</div>
        <p style={{ whiteSpace: "pre-wrap" }}>{ticket.body}</p>
      </div>

      <div className="card" style={{ display: "flex", gap: 10, alignItems: "center" }}>
        <label>
          Prompt&nbsp;
          <select value={promptVersion} onChange={(e) => onVersionChange(e.target.value)}>
            <option value="v1">v1 (baseline)</option>
            <option value="v2">v2 (hardened)</option>
          </select>
        </label>
        <button className="primary" onClick={onTriage} disabled={busy}>
          {busy ? "Running…" : "Run triage"}
        </button>
      </div>

      {run && <RunView run={run} onDecide={onDecide} busy={busy} />}
    </div>
  );
}

function RunView({
  run, onDecide, busy,
}: {
  run: Run;
  onDecide: (id: string, d: "approved" | "rejected") => void;
  busy: boolean;
}) {
  const action = reviewAction(run);
  const band = confidenceBand(run.result);
  const r = run.result;

  return (
    <>
      <div className={`card action-card ${action.tone}`}>
        <div className="label">{action.label}</div>
        {r.escalation_reason && <div className="muted">{r.escalation_reason}</div>}
        {run.evidence_injection?.flagged && (
          <div className="muted">
            Suspicious instructions found in retrieved knowledge:{" "}
            {run.evidence_injection.chunk_ids.join(", ")}
          </div>
        )}
      </div>

      {/* Only shown when a money-moving action is waiting on a person. */}
      {run.approval_id && (
        <div className="card action-card review">
          <h3>Approval required</h3>
          <p className="muted">
            The agent proposed a refund. It has NOT been executed — money moves
            only when you approve.
          </p>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="primary" disabled={busy}
              onClick={() => onDecide(run.approval_id!, "approved")}>Approve</button>
            <button disabled={busy}
              onClick={() => onDecide(run.approval_id!, "rejected")}>Reject</button>
          </div>
        </div>
      )}

      <div className="card">
        <h3>Agent draft</h3>
        <div className="chips">
          <span className="chip">{titleise(r.intent)}</span>
          <span className="chip">priority: {r.priority}</span>
          <span className="chip">{r.sentiment}</span>
          <span className={`chip ${band.tone}`}>
            confidence: {r.confidence} ({band.label})
          </span>
          {run.rag_mode && <span className="chip">rag: {run.rag_mode}</span>}
        </div>

        {r.suggested_reply
          ? <div className="reply">{r.suggested_reply}</div>
          : <p className="muted">No draft — routed to a human.</p>}

        {r.proposed_actions.length > 0 && (
          <p className="muted">
            Proposed: {r.proposed_actions.map((a: any) =>
              a.action_type === "issue_refund"
                ? `refund ${money(Number(a.amount_cents))}`
                : a.action_type).join(", ")}
          </p>
        )}
      </div>

      <SourcesCard run={run} />
    </>
  );
}

// SourcesCard — where the answer came from.
//
// Every entry is built from evidence the run actually retrieved, so a source
// card can never describe an article the agent was not shown. When the
// citation check rejected something, that is stated plainly rather than hidden.
function SourcesCard({ run }: { run: Run }) {
  const sources = run.result.sources ?? [];
  const invalid = run.citations?.invalid ?? [];

  if (!sources.length && !invalid.length) {
    return (
      <div className="card">
        <h3>Sources</h3>
        <p className="muted">
          No citations. The agent either answered from account data or escalated
          without making a product claim.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <h3>Sources</h3>
      {sources.map((s) => (
        <div className="source" key={s.chunk_id}>
          <code className="chunk">{s.chunk_id}</code>
          <span className="source-title">{s.title}</span>
          {s.section && <span className="muted"> — {s.section}</span>}
          {s.url && (
            <a href={s.url} target="_blank" rel="noreferrer" className="source-link">
              open
            </a>
          )}
        </div>
      ))}

      {invalid.length > 0 && (
        <div className="fallback">
          <b>Rejected citations:</b> {invalid.join(", ")}
          <div className="muted">
            These were not retrieved during this run, so they were stripped and
            the ticket was escalated.
          </div>
        </div>
      )}
    </div>
  );
}
