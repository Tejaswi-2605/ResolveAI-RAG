// TopBar.tsx — the title bar with live stats, including RAG health.
import type { IndexStatus, Stats } from "../lib/types";
import { ms } from "../lib/format";

export function TopBar({ stats, index }: { stats: Stats | null; index: IndexStatus | null }) {
  return (
    <div className="topbar">
      <h1>ResolveAI-RAG</h1>
      {stats && (
        <>
          <span className="stat">runs <b>{stats.run_count}</b></span>
          <span className="stat">escalation <b>{(stats.escalation_rate * 100).toFixed(0)}%</b></span>
          <span className="stat">avg latency <b>{ms(stats.avg_latency_ms)}</b></span>
          <span className="stat">pending approvals <b>{stats.pending_approvals}</b></span>
          <span className="stat">injection flagged <b>{stats.injection_flagged_runs}</b></span>
          <span className="stat">hybrid <b>{(stats.hybrid_retrieval_rate * 100).toFixed(0)}%</b></span>
        </>
      )}
      {index && (
        // The honest indicator: green only when semantic retrieval is genuinely
        // available and the index matches the current knowledge base.
        <span className={`stat pill ${index.semantic_available && !index.stale ? "ok" : "warn"}`}>
          {!index.exists
            ? "no index — run ingest"
            : index.stale
              ? "index stale"
              : index.semantic_available
                ? `${index.chunks} chunks · ${index.dimension}d`
                : "lexical only"}
        </span>
      )}
    </div>
  );
}
