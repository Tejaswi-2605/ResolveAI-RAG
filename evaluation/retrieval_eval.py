"""
retrieval_eval.py — DOES HYBRID RETRIEVAL ACTUALLY HELP?

A system that calls itself "Hybrid RAG" is making a measurable claim. This
harness tests it by running the SAME queries against the SAME corpus through
four configurations and comparing them:

    lexical          BM25 only
    semantic         embeddings + vector search only
    hybrid           both arms, fused with RRF
    hybrid+rerank    hybrid, then the heuristic reranker

Run it:
    python evaluation/retrieval_eval.py
    python evaluation/retrieval_eval.py --embedding-provider hashing

THE METRICS, AND WHAT EACH ONE ANSWERS

  Recall@k   "Of the queries, how many had a correct article somewhere in the
             top k?" This is the metric that matters most for RAG, because the
             LLM reads all k chunks. A correct chunk at position 3 is just as
             usable as one at position 1 — but a correct chunk at position 9
             never reaches the model at all.

  MRR        Mean Reciprocal Rank: 1/(rank of the first correct result),
             averaged. Unlike recall it cares about ORDER, so it rewards
             putting the right answer first. Useful because the first chunk
             gets the most attention in a prompt.

  Latency    Average and p95 per query. p95 rather than max, because one cold
             start should not define the number a user experiences.

THE BREAKDOWN BY QUERY KIND IS THE POINT
An overall average can hide the whole story. The dataset labels each query
`exact` (rare identifiers — lexical's home turf), `paraphrase` (different words
for the same idea — semantic's home turf) or `mixed`. If hybrid only ties the
overall average but wins BOTH subsets, that is still the right architecture,
and the breakdown is what shows it.

HONESTY NOTE: gold labels are at ARTICLE level, not chunk level. A retrieval
counts as correct when it returns any chunk of a labelled article. Chunk-level
labels would be stricter, but article-level matches how the agent actually
uses the evidence — and inventing chunk-level labels by hand would encode my
guesses rather than measure the retriever.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone

import _bootstrap  # noqa: F401  (must be imported before any `app` import)
from _bootstrap import DATASETS_DIR, RESULTS_DIR

from app.config import get_settings
from app.database.seed import seed
from app.rag.hybrid import HybridRetriever, reset_retriever_cache
from app.rag.ingest import ingest

DATASET_PATH = DATASETS_DIR / "retrieval_queries.json"

# (label, retrieval mode, reranker name or None)
# The reranker is named EXPLICITLY rather than read from configuration: the
# shipped default is `none`, so relying on the setting would silently turn the
# "hybrid+rerank" row into a duplicate of "hybrid" and quietly invalidate the
# comparison this file exists to make.
CONFIGURATIONS = [
    ("lexical", "lexical", None),
    ("semantic", "semantic", None),
    ("hybrid", "hybrid", None),
    ("hybrid+rerank", "hybrid", "heuristic"),
]

RECALL_DEPTHS = (1, 3, 5)
MRR_DEPTH = 10


def load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _first_relevant_rank(article_ids: list[str], gold: set[str]) -> int | None:
    """1-based position of the first correct article, or None if absent."""
    for position, article_id in enumerate(article_ids, start=1):
        if article_id in gold:
            return position
    return None


def evaluate_configuration(retriever: HybridRetriever, dataset: list[dict],
                           label: str, mode: str, reranker: str | None,
                           depth: int) -> dict:
    """Run every query through one configuration and score the results."""
    # The retriever reads the reranker from settings, so name it here for the
    # duration of this configuration only.
    previous = os.environ.get("RERANKER")
    os.environ["RERANKER"] = reranker or "none"
    use_reranker = reranker is not None

    per_query: list[dict] = []

    for case in dataset:
        gold = set(case["gold_article_ids"])

        started = time.perf_counter()
        evidence, trace = retriever.retrieve(case["query"], top_k=depth, mode=mode,
                                             use_reranker=use_reranker)
        elapsed_ms = (time.perf_counter() - started) * 1000

        # De-duplicate to article level while PRESERVING rank order: several
        # chunks of one article collapse to that article's best position.
        article_order: list[str] = []
        for item in evidence:
            if item.chunk.article_id not in article_order:
                article_order.append(item.chunk.article_id)

        rank = _first_relevant_rank(article_order, gold)
        per_query.append({
            "id": case["id"],
            "kind": case["kind"],
            "query": case["query"],
            "gold": sorted(gold),
            "retrieved_articles": article_order[:MRR_DEPTH],
            "top_chunk_ids": trace.top_chunk_ids,
            "first_relevant_rank": rank,
            "reciprocal_rank": (1.0 / rank) if rank and rank <= MRR_DEPTH else 0.0,
            "mode_used": trace.mode_used,
            "fallbacks": trace.fallbacks,
            "latency_ms": elapsed_ms,
        })

    if previous is None:
        os.environ.pop("RERANKER", None)
    else:
        os.environ["RERANKER"] = previous

    return {
        "label": label,
        "mode": mode,
        "reranker": reranker or "none",
        "metrics": aggregate(per_query),
        "by_kind": {kind: aggregate([q for q in per_query if q["kind"] == kind])
                    for kind in sorted({q["kind"] for q in per_query})},
        "per_query": per_query,
    }


def aggregate(per_query: list[dict]) -> dict:
    """Turn per-query results into headline metrics."""
    if not per_query:
        return {}

    total = len(per_query)
    metrics: dict[str, float] = {"n_queries": total}

    for k in RECALL_DEPTHS:
        hits = sum(1 for q in per_query
                   if q["first_relevant_rank"] is not None and q["first_relevant_rank"] <= k)
        metrics[f"recall@{k}"] = round(hits / total, 3)

    metrics[f"mrr@{MRR_DEPTH}"] = round(
        statistics.mean(q["reciprocal_rank"] for q in per_query), 3)

    latencies = sorted(q["latency_ms"] for q in per_query)
    metrics["avg_latency_ms"] = round(statistics.mean(latencies), 2)
    index = min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))
    metrics["p95_latency_ms"] = round(latencies[index], 2)

    return metrics


def run(depth: int = 5, rebuild: bool = True) -> dict:
    """Evaluate every configuration against the current index."""
    settings = get_settings()

    # Seed first so the evaluation is reproducible from a clean checkout: the
    # numbers must describe a known corpus, not whatever happens to be in the
    # developer's database.
    seed()

    if rebuild:
        summary = ingest(settings)
        reset_retriever_cache()
        print(f"index built: {summary['chunks']} chunks from {summary['articles']} "
              f"articles using {summary['embedding_model']} "
              f"({summary['dimension']}d, {summary['vector_backend']})\n")

    retriever = HybridRetriever(settings)
    if not retriever.semantic_available:
        raise RuntimeError(
            "semantic retrieval is unavailable, so lexical/semantic/hybrid cannot be "
            f"compared honestly. Fallbacks: {retriever.startup_fallbacks}")

    # Warm up before timing anything. The first semantic query pays the model
    # load (hundreds of ms), which would otherwise be charged to whichever
    # configuration happens to run first and make the latency table nonsense.
    retriever.retrieve("warm up the embedding model", top_k=depth, mode="hybrid",
                       use_reranker=False)

    dataset = load_dataset()
    reports = [evaluate_configuration(retriever, dataset, label, mode, rerank, depth)
               for label, mode, rerank in CONFIGURATIONS]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_queries": len(dataset),
        "top_k": depth,
        "embedding_model": retriever.embedder.model_id,
        "embedding_provider": retriever.embedder.name,
        "vector_backend": settings.vector_backend,
        "rrf_k": settings.rrf_k,
        "reranker": settings.reranker,
        "configurations": reports,
    }


# ── reporting ─────────────────────────────────────────────────────
def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(headers) + " |",
            "|" + "---|" * len(headers)] + \
           ["| " + " | ".join(row) + " |" for row in rows]


def write_report(report: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "retrieval.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    configurations = report["configurations"]
    metric_names = [f"recall@{k}" for k in RECALL_DEPTHS] + \
                   [f"mrr@{MRR_DEPTH}", "avg_latency_ms", "p95_latency_ms"]

    lines = [
        "# ResolveAI-RAG — Retrieval Evaluation", "",
        f"Generated: {report['generated_at']}", "",
        f"- Queries: **{report['n_queries']}**, top-K = {report['top_k']}",
        f"- Embedding model: `{report['embedding_model']}` ({report['embedding_provider']})",
        f"- Vector backend: `{report['vector_backend']}`, RRF k = {report['rrf_k']}",
        "",
        "Gold labels are at ARTICLE level: a query counts as answered when any "
        "chunk of a labelled article is retrieved.", "",
        "## Overall", "",
    ]
    lines += _table(
        ["Metric"] + [c["label"] for c in configurations],
        [[metric] + [str(c["metrics"][metric]) for c in configurations]
         for metric in metric_names])

    lines += ["", "## By query kind", "",
              "`exact` = rare identifiers (lexical's home turf). "
              "`paraphrase` = different words for the same idea (semantic's home "
              "turf). `mixed` = natural how-to questions.", ""]

    for kind in sorted(configurations[0]["by_kind"]):
        count = configurations[0]["by_kind"][kind]["n_queries"]
        lines += [f"### {kind} ({count} queries)", ""]
        lines += _table(
            ["Metric"] + [c["label"] for c in configurations],
            [[metric] + [str(c["by_kind"][kind][metric]) for c in configurations]
             for metric in [f"recall@{k}" for k in RECALL_DEPTHS] + [f"mrr@{MRR_DEPTH}"]])
        lines.append("")

    # Failures are kept visible on purpose — an evaluation that only reports
    # wins is marketing, not measurement.
    lines += ["## Queries no configuration answered in the top-K", ""]
    never_found = [q["id"] for q in configurations[0]["per_query"]
                   if all(any(pq["id"] == q["id"] and pq["first_relevant_rank"] is None
                              for pq in c["per_query"]) for c in configurations)]
    lines.append("_None._" if not never_found else
                 "\n".join(f"- `{qid}`" for qid in never_found))

    lines += ["", "## Per-configuration misses", ""]
    for configuration in configurations:
        misses = [q for q in configuration["per_query"] if q["first_relevant_rank"] is None]
        lines.append(f"**{configuration['label']}** — {len(misses)} miss(es)")
        for miss in misses:
            lines.append(f"  - `{miss['id']}` ({miss['kind']}): \"{miss['query']}\" "
                         f"— wanted {miss['gold']}, got {miss['retrieved_articles'][:3]}")
        lines.append("")

    (RESULTS_DIR / "retrieval_report.md").write_text("\n".join(lines) + "\n",
                                                     encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate ResolveAI-RAG retrieval.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="how many results each configuration returns (default 5)")
    parser.add_argument("--no-rebuild", action="store_true",
                        help="use the existing index instead of rebuilding it")
    args = parser.parse_args(argv)

    report = run(depth=args.top_k, rebuild=not args.no_rebuild)
    write_report(report)

    metric_names = [f"recall@{k}" for k in RECALL_DEPTHS] + \
                   [f"mrr@{MRR_DEPTH}", "avg_latency_ms"]
    width = max(len(c["label"]) for c in report["configurations"]) + 2

    print(f"{'config':<{width}}" + "".join(f"{m:>16}" for m in metric_names))
    for configuration in report["configurations"]:
        print(f"{configuration['label']:<{width}}" +
              "".join(f"{configuration['metrics'][m]:>16}" for m in metric_names))

    print(f"\nWrote {RESULTS_DIR / 'retrieval_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
