// RetrievalPanel.tsx — MAKES THE HYBRID RAG PIPELINE VISIBLE.
//
// This is the component the whole project exists to justify. Without it a
// reviewer has to take "we use hybrid retrieval" on faith. With it they can
// see, for every single answer: how many candidates each arm returned, whether
// the two were fused, whether a fallback fired, and which chunks won.
//
// The fallback row is the important one. If the vector index is missing, this
// panel says so in orange rather than quietly showing a green "hybrid" badge.
import type { RetrievalTrace } from "../lib/types";
import { retrievalHealth } from "../lib/format";

export function RetrievalPanel({ traces }: { traces: RetrievalTrace[] }) {
  if (!traces.length) {
    return (
      <p className="muted">
        No knowledge-base search in this run. The agent answered from the
        database alone, or escalated before retrieving.
      </p>
    );
  }

  return (
    <div>
      {traces.map((trace, i) => {
        const health = retrievalHealth(trace);
        return (
          <div className="card" key={i}>
            <div className="chips">
              <span className={`chip ${health.tone}`}>{health.label}</span>
              {trace.fusion_method && <span className="chip">fusion: {trace.fusion_method}</span>}
              <span className="chip">
                rerank: {trace.reranker ?? "off"}
              </span>
            </div>

            <div className="query">“{trace.query}”</div>

            {/* The pipeline, drawn as the two arms converging. */}
            <div className="pipeline">
              <div className="arm">
                <div className="arm-label">BM25 lexical</div>
                <div className="arm-count">{trace.lexical_candidates ?? 0}</div>
                <div className="arm-sub">candidates</div>
              </div>
              <div className="arm">
                <div className="arm-label">Vector semantic</div>
                <div className="arm-count">{trace.semantic_candidates ?? 0}</div>
                <div className="arm-sub">candidates</div>
              </div>
              <div className="arm merge">
                <div className="arm-label">{trace.fusion_method?.toUpperCase() ?? "SINGLE ARM"}</div>
                <div className="arm-count">{trace.final_k ?? 0}</div>
                <div className="arm-sub">evidence</div>
              </div>
            </div>

            {trace.fallbacks?.length > 0 && (
              <div className="fallback">
                <b>Fallback:</b> {trace.fallbacks.join(", ")}
                <div className="muted">
                  Retrieval degraded rather than failing. The answer used only the
                  arm that was available.
                </div>
              </div>
            )}

            {trace.top_chunk_ids?.length > 0 && (
              <p className="muted" style={{ marginBottom: 0 }}>
                Top chunks: {trace.top_chunk_ids.map((id) => (
                  <code key={id} className="chunk">{id}</code>
                ))}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
