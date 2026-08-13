"""
agent.py — THE ORCHESTRATOR. The heart of the project.

`run_triage()` takes one support ticket and drives the whole workflow:

    security scan
        → system prompt + fenced ticket
        → model
        → tool call → validate args → execute → fenced observation
        → (repeat, bounded by a step budget)
        → final text → extract JSON → validate → one repair → else fallback
        → CITATION CHECK against the evidence ledger
        → enforce safety post-conditions in code
        → persist everything
        → return one predictable shape

THE CONTRACT: this function NEVER raises for an expected failure. Bad JSON, a
provider timeout, a tool blowing up, an unexpected bug — every path ends in a
PERSISTED run whose `result` is either a validated TriageResult or a safe
fallback with `requires_human=True`. The UI therefore has exactly one shape to
render, always.

THE FIVE BOUNDS THAT KEEP AN AUTONOMOUS AGENT SAFE (name these in interviews)
  1. step budget         — the loop cannot run forever
  2. model retries       — transient errors only, with exponential backoff
  3. tool-arg validation — every call is schema-checked before it executes
  4. code-level gate     — privileged actions cannot execute; a human approves
  5. citation check      — claims must trace to evidence actually retrieved

WHAT THIS FILE DOES NOT KNOW
It has no idea what FAISS, BM25, RRF or an embedding model are. It calls
`search_knowledge_base` and receives evidence dicts. Search this file for
"faiss" or "embedding" and you will find nothing — that is the layering
boundary working, and it is deliberate.

The one RAG concept the agent DOES own is the EVIDENCE LEDGER: the union of
every chunk retrieved during the run. Citations are validated against that
ledger, so the model gets credit for citing something it saw at step 1 even
when it answers at step 4 — but gets no credit for inventing an id.
"""

from __future__ import annotations

import json
import logging
import time

from app.config import Settings, get_settings
from app.core import prompts, security
from app.core.tools import REGISTRY, ToolError, call_tool, tool_schemas
from app.core.validation import (ValidationFailure, contains_refund,
                                 extract_json, fallback_result,
                                 repair_instruction, validate_triage_result)
from app.database import db
from app.providers import (ProviderError, ProviderRateLimit, ProviderTimeout,
                           get_provider)
from app.rag.citations import sources_for, validate_citations

logger = logging.getLogger("resolveai.agent")

# Backoff base kept small so tests stay fast. The PATTERN (exponential) is what
# matters, not the exact seconds.
_BACKOFF_BASE = 0.1

KNOWLEDGE_TOOL = "search_knowledge_base"


def _complete_with_retries(provider, system, messages, tools, timeout_s, retries):
    """
    Call the model, retrying TRANSIENT errors only, with exponential backoff.

    A timeout or a rate limit is worth retrying — the request may simply
    succeed next time. Any other ProviderError signals a real problem (bad
    request, auth failure, a bug) and is re-raised immediately, because
    retrying it would just waste the caller's time and money.
    """
    delay = _BACKOFF_BASE
    for attempt in range(retries + 1):
        try:
            return provider.complete(system, messages, tools, timeout_s)
        except (ProviderTimeout, ProviderRateLimit):
            if attempt >= retries:
                raise
            time.sleep(delay)
            delay *= 2


def _is_refund(action) -> bool:
    if isinstance(action, dict):
        return action.get("action_type") == "issue_refund"
    return action == "issue_refund"


def run_triage(ticket: dict, provider=None, prompt_version: str = prompts.DEFAULT_VERSION,
               max_steps: int | None = None, timeout_s: float | None = None,
               persist: bool = True, settings: Settings | None = None) -> dict:
    """Run the full triage workflow for one ticket and return a result dict."""
    settings = settings or get_settings()
    max_steps = max_steps or settings.agent_max_steps
    timeout_s = timeout_s or settings.agent_timeout_s

    started = time.perf_counter()
    provider = provider or get_provider(settings=settings)
    system = prompts.system_prompt(prompt_version)   # raises on a bad version; the API guards it

    run_id = db.new_id("run")
    ticket_id = ticket["id"]

    # Named stages, appended as they happen. This is the observability spine:
    # a run's `stages` list reads as the story of what the agent actually did.
    stages: list[str] = []

    # ── stage 1: scan the ticket BEFORE the model ever sees it ────────
    scan = security.scan_for_injection(ticket.get("body", ""))
    injection_flagged = scan["flagged"]
    stages.append("security_scan")
    if injection_flagged:
        logger.warning("run %s: injection flagged categories=%s",
                       run_id, scan["categories"])

    # Create the run row BEFORE the loop: tool_calls.run_id and
    # retrievals.run_id are foreign keys, so the parent must exist before any
    # child row can be written. It also means a crash mid-loop leaves a visible
    # 'running' record instead of no evidence at all.
    if persist:
        db.start_run(run_id, ticket_id, prompt_version, provider.name,
                     provider.model, settings.rag_mode, injection_flagged)

    messages = [{"role": "user", "content": prompts.build_user_message(ticket)}]

    trace: list[dict] = []            # one entry per tool call, for the UI
    tools_used: list[str] = []
    evidence_ledger: list[dict] = []  # EVERY chunk retrieved during this run
    ledger_ids: set[str] = set()
    retrieval_traces: list[dict] = []
    pending_action = None             # a privileged proposal awaiting approval
    evidence_injection = {"flagged": False, "chunk_ids": [], "categories": []}

    input_tokens = output_tokens = 0
    steps_used = 0
    validation_failures = 0
    result = None
    error = None
    hard_failed = False               # validation/provider/unexpected → 'failed'
    provider_failed = False           # transient failure after retries → 'escalated'
    budget_exhausted = False

    try:
        # ── stage 2: the bounded loop ─────────────────────────────────
        stages.append("agent_loop")
        for step in range(1, max_steps + 1):
            steps_used = step
            response = _complete_with_retries(provider, system, messages,
                                              tool_schemas(), timeout_s,
                                              settings.agent_max_retries)
            input_tokens += response.input_tokens
            output_tokens += response.output_tokens

            if response.tool_calls:
                tool_call = response.tool_calls[0]      # one tool per turn
                messages.append({"role": "assistant", "content": response.text,
                                 "tool_calls": [tool_call]})
                tools_used.append(tool_call.name)
                tool = REGISTRY.get(tool_call.name)

                try:
                    # call_tool VALIDATES arguments before executing (bound #3).
                    tool_result, latency = call_tool(tool_call.name, tool_call.args)
                    ok, tool_error = True, None
                except ToolError as exc:
                    tool_result, latency, ok, tool_error = None, 0, False, str(exc)

                # -- knowledge tool: grow the evidence ledger ----------
                if ok and tool_call.name == KNOWLEDGE_TOOL:
                    if "retrieval" not in stages:
                        stages.append("retrieval")
                    retrieval = tool_result.get("retrieval", {})
                    retrieval_traces.append(retrieval)
                    for item in tool_result.get("evidence", []):
                        if item["chunk_id"] not in ledger_ids:
                            ledger_ids.add(item["chunk_id"])
                            evidence_ledger.append(item)

                    # Indirect injection: a retrieved document trying to give
                    # instructions. Detected here, enforced in the
                    # post-conditions below.
                    found = security.scan_evidence(tool_result.get("evidence", []))
                    if found["flagged"]:
                        evidence_injection = {
                            "flagged": True,
                            "chunk_ids": sorted(set(evidence_injection["chunk_ids"])
                                                | set(found["chunk_ids"])),
                            "categories": sorted(set(evidence_injection["categories"])
                                                 | set(found["categories"])),
                        }
                        logger.warning("run %s: suspicious retrieved content in %s",
                                       run_id, found["chunk_ids"])

                    if persist:
                        db.log_retrieval(run_id, {
                            **retrieval,
                            "latency_ms": int(retrieval.get("latency_ms", {}).get("total", 0)),
                        })

                if persist:
                    db.log_tool_call(
                        run_id, step, tool_call.name, json.dumps(tool_call.args), ok,
                        json.dumps(tool_result) if tool_result is not None else None,
                        tool_error, latency)

                trace.append({"step": step, "tool": tool_call.name, "args": tool_call.args,
                              "ok": ok, "result": tool_result, "error": tool_error,
                              "latency_ms": latency})

                # A privileged tool records a PENDING proposal. It is NOT
                # treated as executed — this is the human-in-the-loop gate.
                if ok and tool is not None and tool.privileged:
                    pending_action = tool_result

                # Feed the observation back. Knowledge results are fenced as
                # DATA by build_tool_observation; a ToolError comes back as
                # readable text so the model can self-correct.
                observation = (prompts.build_tool_observation(tool_call.name, tool_result)
                               if ok else f"TOOL_ERROR: {tool_error}")
                messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                 "name": tool_call.name, "ok": ok, "content": observation})
                continue

            # ── the model produced final text → validate the JSON ─────
            if "llm" not in stages:
                stages.append("llm")
            try:
                result = validate_triage_result(extract_json(response.text))
                break
            except ValidationFailure as failure:
                validation_failures += 1
                if validation_failures >= 2:
                    raise                  # second failure → give up → fallback
                messages.append({"role": "user", "content": repair_instruction(failure)})
                continue
        else:
            budget_exhausted = True        # loop finished without `break`

    except ValidationFailure as exc:
        result = fallback_result(f"could not produce valid output: {exc}")
        error, hard_failed = str(exc), True
    except (ProviderTimeout, ProviderRateLimit) as exc:
        # Transient model failure after retries → a human handles the ticket.
        # Not a hard failure: nothing is broken, the model was just unavailable.
        result = fallback_result("model timed out or was rate limited")
        error, provider_failed = str(exc), True
    except ProviderError as exc:
        result = fallback_result(f"provider error: {exc}")
        error, hard_failed = str(exc), True
    except Exception as exc:               # last-resort safety net
        logger.exception("run %s: unexpected error", run_id)
        result = fallback_result(f"unexpected error: {exc}")
        error, hard_failed = str(exc), True

    if budget_exhausted and result is None:
        result = fallback_result("step budget exhausted before a final answer")

    # ── stage 3: citation check (bound #5) ────────────────────────────
    stages.append("citation_check")
    citation_report = validate_citations(result.get("citations", []), evidence_ledger)

    if citation_report.has_fabrication:
        # The model cited evidence it was never shown. Strip the fabricated ids
        # AND escalate — removing the citation alone would let the unsupported
        # claim go out with its audit trail quietly deleted.
        logger.warning("run %s: fabricated citations %s", run_id, citation_report.invalid)
        result["citations"] = list(citation_report.valid)
        result["requires_human"] = True
        result["escalation_reason"] = (result.get("escalation_reason")
                                       or "citation did not match retrieved evidence")

    result["sources"] = sources_for(citation_report.valid, evidence_ledger)

    # An answer that makes product claims with no grounding is escalated rather
    # than sent. `require_citations` makes this configurable, but the default
    # is on: an unsupported confident answer is the failure mode RAG exists to
    # prevent.
    if (settings.require_citations and not result["requires_human"]
            and (result.get("suggested_reply") or "")
            and tools_used.count(KNOWLEDGE_TOOL) > 0
            and not citation_report.valid):
        result["requires_human"] = True
        result["escalation_reason"] = (result.get("escalation_reason")
                                       or "answer was not grounded in retrieved evidence")

    # ── stage 4: safety post-conditions, enforced in CODE ─────────────
    stages.append("post_conditions")

    # Direct injection: force human review, and STRIP any refund the model
    # proposed, including dropping the pending action so no approval — and
    # therefore no money path — is ever created.
    if injection_flagged:
        result["requires_human"] = True
        result["escalation_reason"] = (result.get("escalation_reason")
                                       or "possible prompt injection")
        result["proposed_actions"] = [a for a in result.get("proposed_actions", [])
                                      if not _is_refund(a)]
        if pending_action and pending_action.get("action_type") == "issue_refund":
            pending_action = None

    # Indirect injection: a retrieved document tried to issue instructions.
    # Same treatment — the ticket goes to a person, and the poisoned chunk ids
    # are surfaced so someone can clean the document.
    if evidence_injection["flagged"]:
        result["requires_human"] = True
        result["escalation_reason"] = (result.get("escalation_reason")
                                       or "suspicious content in retrieved knowledge")
        result["proposed_actions"] = [a for a in result.get("proposed_actions", [])
                                      if not _is_refund(a)]
        if pending_action and pending_action.get("action_type") == "issue_refund":
            pending_action = None

    # A pending privileged action always forces human review, and so does a
    # refund appearing in proposed_actions by any route.
    if pending_action or contains_refund(result.get("proposed_actions")):
        result["requires_human"] = True

    # ── decide the final status ───────────────────────────────────────
    if hard_failed:
        status, ticket_status = "failed", "escalated"
    elif provider_failed or budget_exhausted:
        status, ticket_status = "escalated", "escalated"
    elif pending_action:
        status, ticket_status = "escalated", "awaiting_approval"
    elif result.get("requires_human"):
        status, ticket_status = "escalated", "escalated"
    else:
        status, ticket_status = "completed", "triaged"

    latency_ms = int((time.perf_counter() - started) * 1000)
    rag_mode_used = retrieval_traces[-1].get("mode_used") if retrieval_traces else None

    # ── stage 5: persist ──────────────────────────────────────────────
    approval_id = None
    if persist:
        db.finalize_run(run_id, status, json.dumps(result), error, steps_used,
                        latency_ms, input_tokens, output_tokens,
                        not citation_report.has_fabrication, rag_mode_used)
        if pending_action:
            approval_id = db.new_id("apr")
            db.create_approval(
                approval_id, run_id, ticket_id, pending_action["action_type"],
                json.dumps(pending_action),
                result.get("escalation_reason") or "privileged action proposed")
        db.execute("UPDATE tickets SET status=? WHERE id=?", (ticket_status, ticket_id))

    logger.info("run %s status=%s steps=%s latency=%dms intent=%s rag=%s subject=%s",
                run_id, status, steps_used, latency_ms, result.get("intent"),
                rag_mode_used, security.redact_pii(ticket.get("subject", "")))

    return {
        "run_id": run_id,
        "ticket_id": ticket_id,
        "status": status,
        "result": result,
        "trace": trace,
        "stages": stages,
        "approval_id": approval_id,
        "injection": scan,
        "evidence_injection": evidence_injection,
        "citations": citation_report.to_dict(),
        "evidence": evidence_ledger,
        "retrieval": retrieval_traces,
        "rag_mode": rag_mode_used,
        "tools_used": tools_used,
        "prompt_version": prompt_version,
        "provider": provider.name,
        "model": provider.model,
        "steps_used": steps_used,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "error": error,
    }
