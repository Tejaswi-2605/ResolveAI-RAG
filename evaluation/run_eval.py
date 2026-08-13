"""
run_eval.py — THE END-TO-END AGENT EVALUATION.

It runs the REAL agent over a labelled dataset and scores every run with the
deterministic graders in `evaluators.py`. It can compare two prompt versions,
or two retrieval modes, on identical inputs.

Run it:
    python evaluation/run_eval.py --compare-prompts v1 v2
    python evaluation/run_eval.py --compare-rag hybrid lexical
    python evaluation/run_eval.py --prompt-version v2

WHY COMPARE RATHER THAN REPORT ONE NUMBER
"Intent accuracy 0.9" means nothing on its own — 0.9 against what? Every table
this writes has at least two columns, and `HIGHER_IS_BETTER` decides which
direction counts as an improvement EXPLICITLY, so nothing is inferred from a
metric's name.

HONESTY RULES, applied without exception:
  * Every number comes from a real agent run. Nothing is estimated.
  * Failing cases are listed in the report by name and reason. An evaluation
    that only reports wins is marketing.
  * With the mock provider these measure the SYSTEM, not a frontier model, and
    the labels share an author with the mock's rules — so intent accuracy is
    partly circular. That caveat is printed in the report itself, not buried.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone

import _bootstrap  # noqa: F401  (must precede any `app` import)
from _bootstrap import DATASETS_DIR, RESULTS_DIR

import evaluators as ev
from app.config import get_settings
from app.core.agent import run_triage
from app.database import db
from app.database.seed import seed
from app.providers import get_provider
from app.rag.hybrid import reset_retriever_cache
from app.rag.ingest import ingest

DATASET_PATH = DATASETS_DIR / "agent_cases.json"

# Which direction is better for each metric. Explicit, never guessed.
HIGHER_IS_BETTER = {
    "intent_accuracy": True,
    "escalation_accuracy": True,
    "escalation_precision": True,
    "escalation_recall": True,
    "tool_recall": True,
    "forbidden_tool_rate": False,
    "structured_output_validity": True,
    "retrieval_used_rate": True,
    "grounded_citation_rate": True,
    "citation_correctness": True,
    "retrieval_mode_integrity": True,
    "injection_defence_rate": True,
    "approval_correctness": True,
    "unsupported_claim_rate": False,
    "error_rate": False,
    "avg_latency_ms": False,
    "p95_latency_ms": False,
    "avg_steps": False,
}


def load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _insert_case(case: dict, label: str) -> dict:
    """Insert (once) a ticket row for this case + run label."""
    ticket_id = f"eval_{label}_{case['id']}"
    account = db.query_one("SELECT id FROM accounts WHERE contact_email=?",
                           (case["sender_email"],))
    db.execute(
        """INSERT OR REPLACE INTO tickets
           (id, account_id, sender_email, subject, body, channel, status, created_at)
           VALUES (?, ?, ?, ?, ?, 'email', 'new', ?)""",
        (ticket_id, account["id"] if account else None, case["sender_email"],
         case["subject"], case["body"], db.now_iso()))
    return dict(db.query_one("SELECT * FROM tickets WHERE id=?", (ticket_id,)))


def score_case(case: dict, run: dict) -> dict:
    """Apply every grader to one run and collect the reasons it failed."""
    expected = case["expected"]
    tools = ev.check_expected_tools(run, expected)

    scored = {
        "id": case["id"],
        "intent_correct": ev.check_intent(run, expected),
        "escalation_correct": ev.check_escalation(run, expected),
        "predicted_human": bool(run["result"].get("requires_human")),
        "expected_human": bool(expected["requires_human"]),
        "tools_found": tools["found"],
        "tools_expected": tools["expected"],
        "forbidden_clean": ev.check_forbidden_tools(run, expected),
        "structured_valid": ev.check_structured_output(run, expected),
        "retrieval_used": ev.check_retrieval_used(run, expected),
        "grounded_citation": ev.check_grounded_citation(run, expected),
        "citation_correct": ev.check_citation_correctness(run, expected),
        "retrieval_mode_ok": ev.check_retrieval_mode(run, expected),
        "injection_defended": ev.check_injection_defended(run, expected),
        "approval_correct": ev.check_approval(run, expected),
        "unsupported_claim": ev.check_unsupported_claim(run, expected),
        "no_error": ev.check_no_error(run, expected),
        "has_forbidden": bool(expected.get("forbidden_tools")),
        "latency_ms": run["latency_ms"],
        "steps": run["steps_used"],
        "tokens": run["input_tokens"] + run["output_tokens"],
        "rag_mode": run.get("rag_mode"),
    }

    reasons: list[str] = []
    if not scored["intent_correct"]:
        reasons.append(f"intent {run['result'].get('intent')} != {expected['intent']}")
    if not scored["escalation_correct"]:
        reasons.append(f"requires_human {scored['predicted_human']} "
                       f"!= {scored['expected_human']}")
    if scored["tools_expected"] and scored["tools_found"] < scored["tools_expected"]:
        reasons.append(f"missing expected tools "
                       f"({scored['tools_found']}/{scored['tools_expected']})")
    if not scored["forbidden_clean"]:
        reasons.append("used a forbidden tool")
    if scored["grounded_citation"] is False:
        reasons.append("no citation where one was required")
    if not scored["citation_correct"]:
        reasons.append("fabricated a citation")
    if scored["injection_defended"] is False:
        reasons.append("INJECTION NOT DEFENDED")
    if scored["approval_correct"] is False:
        reasons.append("approval created/omitted incorrectly")
    if not scored["no_error"]:
        reasons.append("run failed")
    scored["reasons"] = reasons

    return scored


def _rate(values: list) -> float:
    """Mean of a list of booleans, EXCLUDING None (not applicable)."""
    applicable = [1 if v else 0 for v in values if v is not None]
    return round(sum(applicable) / len(applicable), 3) if applicable else 0.0


def aggregate(scored: list[dict]) -> dict:
    """Turn per-case scores into the headline metrics."""
    total = len(scored)

    # Escalation precision/recall on the "requires_human = True" class.
    true_positive = sum(1 for s in scored if s["predicted_human"] and s["expected_human"])
    false_positive = sum(1 for s in scored if s["predicted_human"] and not s["expected_human"])
    false_negative = sum(1 for s in scored if not s["predicted_human"] and s["expected_human"])

    metrics = {
        "intent_accuracy": _rate([s["intent_correct"] for s in scored]),
        "escalation_accuracy": _rate([s["escalation_correct"] for s in scored]),
        "escalation_precision": round(
            true_positive / (true_positive + false_positive), 3)
        if (true_positive + false_positive) else 0.0,
        "escalation_recall": round(
            true_positive / (true_positive + false_negative), 3)
        if (true_positive + false_negative) else 0.0,
        "tool_recall": round(sum(s["tools_found"] for s in scored) /
                             max(1, sum(s["tools_expected"] for s in scored)), 3),
        "forbidden_tool_rate": round(
            sum(1 for s in scored if s["has_forbidden"] and not s["forbidden_clean"])
            / total, 3),
        "structured_output_validity": _rate([s["structured_valid"] for s in scored]),
        "retrieval_used_rate": _rate([s["retrieval_used"] for s in scored]),
        "grounded_citation_rate": _rate([s["grounded_citation"] for s in scored]),
        "citation_correctness": _rate([s["citation_correct"] for s in scored]),
        "retrieval_mode_integrity": _rate([s["retrieval_mode_ok"] for s in scored]),
        "injection_defence_rate": _rate([s["injection_defended"] for s in scored]),
        "approval_correctness": _rate([s["approval_correct"] for s in scored]),
        "unsupported_claim_rate": round(
            sum(1 for s in scored if s["unsupported_claim"]) / total, 3),
        "error_rate": round(sum(1 for s in scored if not s["no_error"]) / total, 3),
        "avg_steps": round(statistics.mean(s["steps"] for s in scored), 2),
        "total_tokens": sum(s["tokens"] for s in scored),
    }

    latencies = sorted(s["latency_ms"] for s in scored)
    metrics["avg_latency_ms"] = round(statistics.mean(latencies), 1)
    index = min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))
    metrics["p95_latency_ms"] = latencies[index]

    return metrics


def evaluate(label: str, prompt_version: str, rag_mode: str,
             provider_name: str | None) -> dict:
    """Run every case under one configuration."""
    import os
    os.environ["RAG_MODE"] = rag_mode
    reset_retriever_cache()

    dataset = load_dataset()
    scored = []
    for case in dataset:
        ticket = _insert_case(case, label)
        provider = get_provider(provider_name) if provider_name else None
        run = run_triage(ticket, provider=provider, prompt_version=prompt_version,
                         settings=get_settings())
        scored.append(score_case(case, run))

    return {
        "label": label,
        "prompt_version": prompt_version,
        "rag_mode": rag_mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": aggregate(scored),
        "failures": [{"id": s["id"], "reasons": s["reasons"]}
                     for s in scored if s["reasons"]],
        "n_cases": len(scored),
    }


# ── reporting ─────────────────────────────────────────────────────
def write_results(reports: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for report in reports:
        (RESULTS_DIR / f"agent_{report['label']}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")

    labels = [r["label"] for r in reports]
    metric_names = list(reports[0]["metrics"])

    lines = [
        "# ResolveAI-RAG — Agent Evaluation", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
        "> **How to read this.** With the `mock` provider these numbers measure the "
        "SYSTEM — tool wiring, the trust boundary, output validation, the citation "
        "check and the approval gate — not the intelligence of a frontier model. "
        "The mock's rules and these labels share an author, so intent accuracy is "
        "partly circular. The retrieval evaluation "
        "(`retrieval_report.md`) is the one that measures real quality.", "",
        "## Metrics", "",
        "| Metric | " + " | ".join(labels) + " |",
        "|" + "---|" * (len(labels) + 1),
    ]
    for metric in metric_names:
        lines.append("| " + " | ".join([metric] +
                                       [str(r["metrics"][metric]) for r in reports]) + " |")

    if len(reports) == 2:
        base, new = reports
        lines += ["", f"## {new['label']} vs {base['label']}", ""]
        for metric in metric_names:
            if metric not in HIGHER_IS_BETTER:
                continue
            before, after = base["metrics"][metric], new["metrics"][metric]
            if after == before:
                verdict = "same"
            else:
                verdict = ("IMPROVED" if (after > before) == HIGHER_IS_BETTER[metric]
                           else "REGRESSED")
            lines.append(f"- **{metric}**: {base['label']}={before} → "
                         f"{new['label']}={after} ({verdict})")

    for report in reports:
        lines += ["", f"## Failing cases — {report['label']} "
                      f"({len(report['failures'])} of {report['n_cases']})", ""]
        lines.append("_None._" if not report["failures"] else "")
        for failure in report["failures"]:
            lines.append(f"- **{failure['id']}**: {'; '.join(failure['reasons'])}")

    (RESULTS_DIR / "agent_report.md").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ResolveAI-RAG agent evaluation.")
    parser.add_argument("--prompt-version", default="v2")
    parser.add_argument("--rag-mode", default="hybrid")
    parser.add_argument("--compare-prompts", nargs=2, metavar=("A", "B"))
    parser.add_argument("--compare-rag", nargs=2, metavar=("A", "B"))
    parser.add_argument("--provider", default=None)
    args = parser.parse_args(argv)

    seed()
    settings = get_settings()
    summary = ingest(settings)
    reset_retriever_cache()
    print(f"index: {summary['chunks']} chunks, {summary['embedding_model']}\n")

    if args.compare_prompts:
        configurations = [(v, v, args.rag_mode) for v in args.compare_prompts]
    elif args.compare_rag:
        configurations = [(f"rag-{m}", args.prompt_version, m) for m in args.compare_rag]
    else:
        configurations = [(args.prompt_version, args.prompt_version, args.rag_mode)]

    reports = []
    for label, prompt_version, rag_mode in configurations:
        print(f"=== {label} (prompt={prompt_version}, rag={rag_mode}) ===")
        report = evaluate(label, prompt_version, rag_mode, args.provider)
        reports.append(report)
        for name, value in report["metrics"].items():
            print(f"  {name}: {value}")
        print(f"  failing cases: {len(report['failures'])}/{report['n_cases']}\n")

    write_results(reports)
    print(f"Wrote {RESULTS_DIR / 'agent_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
