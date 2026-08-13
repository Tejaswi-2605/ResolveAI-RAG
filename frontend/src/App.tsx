// App.tsx — the three-pane operations console.
//
// State lives here and flows DOWN to the panes as props; the panes call
// callbacks that flow events back UP. That one-way data flow is the core React
// mental model, and it keeps every pane a pure function of the current run.
//
// Three tabs on the right pane, one per question a reviewer actually asks:
//   Trace     — what did the agent DO?
//   Retrieval — where did the evidence come from?
//   Evals     — is the system getting better or worse?
import { useEffect, useState } from "react";
import { api, ApiError } from "./lib/api";
import type { IndexStatus, Run, Stats, Ticket } from "./lib/types";
import { TopBar } from "./components/TopBar";
import { Inbox } from "./components/Inbox";
import { TicketDetail } from "./components/TicketDetail";
import { TraceRail } from "./components/TraceRail";
import { RetrievalPanel } from "./components/RetrievalPanel";
import { EvalDashboard } from "./components/EvalDashboard";

type Tab = "trace" | "retrieval" | "eval";

export default function App() {
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [selected, setSelected] = useState<Ticket | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [index, setIndex] = useState<IndexStatus | null>(null);
  const [promptVersion, setPromptVersion] = useState("v2");
  const [tab, setTab] = useState<Tab>("trace");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    try {
      setTickets(await api.tickets());
      setStats(await api.stats());
      setIndex(await api.ragStatus());
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }
  useEffect(() => { refresh(); }, []);

  function selectTicket(t: Ticket) {
    setSelected(t);
    setRun(null);
    setError(null);
  }

  async function runTriage() {
    if (!selected) return;
    setBusy(true);
    setError(null);
    try {
      setRun(await api.triage(selected.id, promptVersion));
      await refresh();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function decide(approvalId: string, decision: "approved" | "rejected") {
    setBusy(true);
    try {
      await api.decide(approvalId, decision, "operator");
      // Re-read the stored run so the UI reflects the decision without
      // re-running the agent (which would create a second approval).
      if (selected) {
        const runs = await api.runsForTicket(selected.id);
        setRun(runs[0] ?? null);
      }
      await refresh();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <TopBar stats={stats} index={index} />
      <div className="console">
        <Inbox tickets={tickets} selectedId={selected?.id ?? null} onSelect={selectTicket} />

        <div className="pane">
          {error && <div className="error-banner">{error}</div>}
          {selected ? (
            <TicketDetail
              ticket={selected}
              run={run}
              promptVersion={promptVersion}
              onVersionChange={setPromptVersion}
              onTriage={runTriage}
              onDecide={decide}
              busy={busy}
            />
          ) : (
            <p className="muted">Select a ticket from the inbox to begin.</p>
          )}
        </div>

        <div className="pane">
          <div className="tabs">
            <button className={tab === "trace" ? "active" : ""}
              onClick={() => setTab("trace")}>Agent trace</button>
            <button className={tab === "retrieval" ? "active" : ""}
              onClick={() => setTab("retrieval")}>Retrieval</button>
            <button className={tab === "eval" ? "active" : ""}
              onClick={() => setTab("eval")}>Evaluations</button>
          </div>

          {tab === "trace" && <TraceRail run={run} />}
          {tab === "retrieval" && <RetrievalPanel traces={run?.retrieval ?? []} />}
          {tab === "eval" && <EvalDashboard />}
        </div>
      </div>
    </>
  );
}
