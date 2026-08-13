// Inbox.tsx — the left pane: the ticket list with status badges.
import type { Ticket } from "../lib/types";
import { titleise } from "../lib/format";

export function Inbox({
  tickets, selectedId, onSelect,
}: {
  tickets: Ticket[];
  selectedId: string | null;
  onSelect: (t: Ticket) => void;
}) {
  return (
    <div className="pane">
      <h3 className="muted" style={{ marginTop: 0 }}>Inbox ({tickets.length})</h3>
      {tickets.map((t) => (
        <button
          key={t.id}
          className={`inbox-item ${t.id === selectedId ? "active" : ""}`}
          onClick={() => onSelect(t)}
        >
          <div className="subj">{t.subject}</div>
          <div className="from">{t.sender_email}</div>
          <span className={`badge ${t.status}`}>{titleise(t.status)}</span>
        </button>
      ))}
    </div>
  );
}
