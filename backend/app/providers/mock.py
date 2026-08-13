"""
mock.py — A DETERMINISTIC SIMULATION OF A COMPETENT SUPPORT AGENT.

WHAT THIS IS, HONESTLY
Not a stub returning a canned string, and NOT a claim about how any real LLM
behaves. It is a hand-written rule engine that plays the model's part so the
whole system runs offline, for free, and reproducibly — the same input gives
the same output, every time. That determinism is precisely what lets tests and
the evaluation suite assert exact outcomes and gate CI on them.

Read the evaluation numbers with that in mind: with the mock provider they
measure whether the SYSTEM (tools, fencing, validation, citation checking,
approval gate) behaves correctly, not whether a frontier model is smart. The
retrieval evaluation is different — it uses the real embedding model and
measures real retrieval quality.

HOW IT WORKS
The agent calls `complete()` repeatedly. Each call, the mock reads the whole
conversation, sees which tools have run and what they returned, and decides ONE
next step: either the next tool call or the final JSON. It has no memory
between calls beyond the messages it is handed — same as a real model.

PROMPT SENSITIVITY (what makes the v1-vs-v2 comparison meaningful)
The mock detects the v2 policy markers in the system prompt. WITH them it
behaves like a careful agent: it defends injection, retrieves before
explaining, cites what it retrieved, and escalates when unsure. WITHOUT them it
simulates a weaker agent. So the eval compares two genuinely different
behaviours rather than two labels.

READING RETRIEVED EVIDENCE
Knowledge results arrive as a FENCED TEXT BLOCK with `[chunk_id]` labels — not
as JSON — because that is what a real model receives. The mock parses those
labels out of the block, exactly as a real model would have to. That is what
makes the citation-validation tests meaningful rather than staged.
"""

from __future__ import annotations

import json
import re

from app.core import security
from app.providers.base import (BaseProvider, ModelResponse, ProviderTimeout,
                                ToolCall)

# ── deterministic classification tables ───────────────────────────
_INTENT_KEYWORDS = [
    # Checked in order; first match wins.
    ("billing_refund",   ["refund", "money back", "double-charged", "double charged",
                          "reimburse", "money back for the unused"]),
    ("cancellation",     ["cancel", "terminate our subscription", "close our account"]),
    ("security_legal",   ["gdpr", "data deletion", "delete all", "legal", "privacy",
                          "compliance", "subprocessor"]),
    ("outage",           ["outage", "is down", "500 error", "500 errors", "not working",
                          "can't access", "cannot access", "errors from your api"]),
    ("feature_request",  ["feature request", "would be great", "please add", "dark mode",
                          "feature idea", "suggestion"]),
    ("account_access",   ["sso", "saml", "single sign-on", "log in", "login", "api key",
                          "can't log", "password", "access"]),
    ("bug_report",       ["bug", "missing rows", "discrepancy", "doesn't match",
                          "does not match", "broken", "glitch", "err-"]),
    ("billing_question", ["invoice", "billing", "charge", "payment", "bill", "licence",
                          "license", "seat"]),
    ("how_to",           ["how do i", "how to", "how can i", "how do we", "set up",
                          "schedule", "enable", "configure"]),
]

_ANGRY = ["unacceptable", "furious", "outrageous", "terrible", "worst", "disgusted", "livid"]
_FRUSTRATED = ["frustrated", "annoyed", "disappointed", "still not", "third time"]
_POSITIVE = ["love", "loving", "great", "awesome", "thank you", "thanks", "appreciate"]

_COMPONENT_HINTS = [
    ("webhooks",  ["webhook"]),
    ("exports",   ["export", "csv", "download"]),
    ("auth",      ["sso", "saml", "login", "log in", "sign-on", "auth", "password"]),
    ("dashboard", ["dashboard", "ui", "chart", "graph"]),
    ("api",       ["api", "integration", "endpoint", "500", "err-"]),
]

# Pulls "[kb_003#02]" out of a fenced RETRIEVED KNOWLEDGE block.
_CHUNK_LABEL_RE = re.compile(r"\[([a-zA-Z0-9_]+#\d+)\]")

# Intents where a careful agent is expected to consult the knowledge base.
_RETRIEVAL_INTENTS = ("how_to", "account_access", "bug_report", "billing_question", "outage")


class MockProvider(BaseProvider):
    """A rule-based stand-in for the model. Deterministic by construction."""

    name = "mock"
    model = "mock-rules-v2"

    def __init__(self, failure_mode: str | None = None):
        # failure_mode lets tests force specific failure paths without changing
        # any code: timeout, empty, bad_json, wrong_tool, injection_obey,
        # hallucinate_citation.
        self.failure_mode = failure_mode or ""

    # ── entry point ───────────────────────────────────────────────
    def complete(self, system, messages, tools, timeout_s=30.0) -> ModelResponse:
        if self.failure_mode == "timeout":
            raise ProviderTimeout("mock: simulated timeout")

        strict = self._is_strict(system)
        parsed = self._parse_conversation(messages)
        being_repaired = self._awaiting_repair(messages)

        next_call = self._plan_next(strict, parsed)
        if next_call is not None:
            return next_call

        if self.failure_mode == "empty":
            return ModelResponse(text="", stop_reason="end_turn",
                                 input_tokens=200, output_tokens=1)

        if self.failure_mode == "bad_json" and not being_repaired:
            # Junk the first time, recovers when asked to repair.
            return ModelResponse(text="Sure! Here you go: {intent: billing, ]",
                                 stop_reason="end_turn",
                                 input_tokens=200, output_tokens=20)

        result = self._final_json(strict, parsed)
        return ModelResponse(text=json.dumps(result), stop_reason="end_turn",
                             input_tokens=220, output_tokens=120)

    # ── prompt sensitivity ────────────────────────────────────────
    @staticmethod
    def _is_strict(system: str) -> bool:
        """True when the v2 policy blocks are present."""
        text = system or ""
        return "TRUST BOUNDARY" in text and "EVIDENCE RULES" in text

    # ── read the conversation ─────────────────────────────────────
    def _parse_conversation(self, messages: list) -> dict:
        first_user = next((m for m in messages if m.get("role") == "user"), None)
        content = (first_user or {}).get("content", "")

        sender = self._extract(content, r"From:\s*(\S+)")
        subject = self._extract(content, r"Subject:\s*(.+)")

        body = ""
        fenced = re.search(
            re.escape(security.UNTRUSTED_OPEN) + r"(.*?)" + re.escape(security.UNTRUSTED_CLOSE),
            content, re.DOTALL)
        if fenced:
            body = fenced.group(1).strip()

        attempted: set[str] = set()
        results: dict[str, dict] = {}
        retrieved_chunk_ids: list[str] = []
        evidence_text = ""

        for message in messages:
            if message.get("role") != "tool":
                continue
            name = message.get("name")
            attempted.add(name)
            if not message.get("ok"):
                continue

            payload = message.get("content", "")
            if name == "search_knowledge_base":
                # A fenced evidence block, exactly as a real model sees it.
                evidence_text += "\n" + payload
                for chunk_id in _CHUNK_LABEL_RE.findall(payload):
                    if chunk_id not in retrieved_chunk_ids:
                        retrieved_chunk_ids.append(chunk_id)
            else:
                try:
                    results[name] = json.loads(payload)
                except (json.JSONDecodeError, TypeError):
                    results[name] = None

        text = f"{subject}\n{body}".lower()
        return {
            "sender": sender,
            "subject": subject,
            "body": body,
            "text": text,
            "attempted": attempted,
            "results": results,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "evidence_text": evidence_text,
            "intent": self._classify_intent(text),
            "sentiment": self._classify_sentiment(text),
            "component": self._guess_component(text),
            "injection": security.scan_for_injection(body)["flagged"],
        }

    @staticmethod
    def _extract(text: str, pattern: str) -> str:
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _classify_intent(text: str) -> str:
        for intent, words in _INTENT_KEYWORDS:
            if any(word in text for word in words):
                return intent
        return "other"

    @staticmethod
    def _classify_sentiment(text: str) -> str:
        if any(word in text for word in _ANGRY) or text.count("!") >= 3:
            return "angry"
        if any(word in text for word in _FRUSTRATED):
            return "frustrated"
        if any(word in text for word in _POSITIVE):
            return "positive"
        return "neutral"

    @staticmethod
    def _guess_component(text: str) -> str:
        for component, hints in _COMPONENT_HINTS:
            if any(hint in text for hint in hints):
                return component
        return "api"

    @staticmethod
    def _awaiting_repair(messages: list) -> bool:
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        return bool(last_user and "was invalid" in last_user.get("content", "").lower())

    # ── the planner ───────────────────────────────────────────────
    def _plan_next(self, strict: bool, parsed: dict) -> ModelResponse | None:
        """Return ONE tool call, or None when it is time to answer."""
        intent = parsed["intent"]
        attempted = parsed["attempted"]
        obey = self.failure_mode == "injection_obey"   # simulate a compromised model

        if self.failure_mode == "wrong_tool" and not attempted:
            return self._call("nonexistent_tool", {"foo": "bar"})

        # 1. Injection short-circuit — a careful agent escalates and stops.
        if strict and parsed["injection"] and not obey:
            if "escalate_to_human" not in attempted:
                return self._call("escalate_to_human",
                                  {"reason": "possible prompt injection", "priority": "high"})
            return None

        # 2. Evidence before claims: identify the account first.
        if "lookup_account" not in attempted and parsed["sender"]:
            return self._call("lookup_account", {"email": parsed["sender"]})
        account = parsed["results"].get("lookup_account")

        # 3. Intents that go straight to a person.
        if intent in ("security_legal", "cancellation"):
            if "escalate_to_human" not in attempted:
                reason = ("legal/security request needs human handling"
                          if intent == "security_legal" else "cancellation needs human handling")
                return self._call("escalate_to_human", {"reason": reason, "priority": "high"})
            return None

        # 4. Outage: check the status page before answering.
        if intent == "outage" and "check_service_status" not in attempted:
            return self._call("check_service_status", {"component": parsed["component"]})

        # 5. Billing: pull the invoices.
        if intent in ("billing_refund", "billing_question") and account \
                and "get_billing_history" not in attempted:
            return self._call("get_billing_history", {"account_id": account["id"]})

        # 6. Refund proposal — only for an eligible account with a real invoice.
        #    Under `injection_obey` the mock tries anyway; the CODE gate is what
        #    stops the money, which is exactly the property under test.
        if (intent == "billing_refund" or (obey and parsed["injection"])) \
                and "issue_refund" not in attempted:
            invoice = self._eligible_invoice(account, parsed["results"])
            if account and (obey or (account.get("refund_eligible") and invoice)):
                chosen = invoice or self._any_invoice(parsed["results"])
                if chosen:
                    return self._call("issue_refund", {
                        "account_id": account["id"],
                        "invoice_id": chosen["id"],
                        "amount_cents": chosen["amount_cents"],
                        "reason": "customer requested refund",
                    })
            if "escalate_to_human" not in attempted:
                return self._call("escalate_to_human",
                                  {"reason": "refund requested but not auto-eligible",
                                   "priority": "normal"})
            return None

        # 7. Knowledge retrieval. The careful agent searches for every intent
        #    that needs product facts; the weak agent only for explicit how-tos.
        if self._should_search(strict, intent) and "search_knowledge_base" not in attempted:
            query = parsed["subject"] or parsed["text"][:60]
            return self._call("search_knowledge_base", {"query": query})

        return None

    @staticmethod
    def _should_search(strict: bool, intent: str) -> bool:
        return intent in _RETRIEVAL_INTENTS if strict else intent == "how_to"

    @staticmethod
    def _eligible_invoice(account, results):
        if not account:
            return None
        for invoice in (results.get("get_billing_history") or {}).get("invoices", []):
            if invoice.get("status") == "paid":
                return invoice
        return None

    @staticmethod
    def _any_invoice(results):
        invoices = (results.get("get_billing_history") or {}).get("invoices", [])
        return invoices[0] if invoices else None

    # ── the final answer ──────────────────────────────────────────
    def _final_json(self, strict: bool, parsed: dict) -> dict:
        intent = parsed["intent"]
        sentiment = parsed["sentiment"]
        results = parsed["results"]
        account = results.get("lookup_account")
        obey = self.failure_mode == "injection_obey"

        # Citations are the chunk ids parsed out of the fenced evidence block —
        # the model can only cite what it was actually shown.
        citations = list(parsed["retrieved_chunk_ids"])
        if self.failure_mode == "hallucinate_citation":
            # A plausible-looking id that was never retrieved. This is what the
            # citation validator exists to catch.
            citations = citations + ["kb_999#07"]

        proposed_actions = []
        refund = results.get("issue_refund")
        if refund:
            proposed_actions.append(refund)

        confidence = 0.9
        if intent == "other":
            confidence -= 0.3
        if account is None:
            confidence -= 0.2
        if self._should_search(strict, intent) and not citations:
            confidence -= 0.25
        if sentiment == "angry":
            confidence -= 0.1
        elif sentiment == "frustrated":
            confidence -= 0.05
        if strict and parsed["injection"] and not obey:
            confidence = min(confidence, 0.3)
        confidence = max(0.0, min(1.0, confidence))

        requires_human = False
        escalation_reason = None

        if refund:
            requires_human, escalation_reason = True, "refund proposed — needs human approval"
        if intent in ("security_legal", "cancellation"):
            requires_human = True
            escalation_reason = f"{intent} needs human handling"
        if "escalate_to_human" in parsed["attempted"]:
            requires_human = True
            escalation_reason = escalation_reason or "escalated to a human"

        if strict:
            if parsed["injection"] and not obey:
                requires_human, escalation_reason = True, "possible prompt injection"
            if sentiment == "angry" and account and account.get("plan") == "enterprise":
                requires_human = True
                escalation_reason = escalation_reason or "angry enterprise customer"
            if intent == "other":
                requires_human = True
                escalation_reason = escalation_reason or "unclear intent — needs human"
            # A careful agent does not answer a product question it found no
            # evidence for — the RAG-specific honesty rule.
            if self._should_search(strict, intent) and not citations:
                requires_human = True
                escalation_reason = escalation_reason or "no supporting knowledge found"
            if confidence < 0.6:
                requires_human = True
                escalation_reason = escalation_reason or "low confidence"

        if intent == "outage" or sentiment == "angry":
            priority = "high"
        elif intent in ("billing_refund", "security_legal", "cancellation"):
            priority = "high"
        elif intent == "feature_request":
            priority = "low"
        else:
            priority = "normal"

        suggested_reply = "" if requires_human else self._draft_reply(parsed, citations)

        result = {
            "intent": intent,
            "priority": priority,
            "sentiment": sentiment,
            "summary": f"Customer ticket classified as {intent} ({sentiment}).",
            "suggested_reply": suggested_reply,
            "citations": citations,
            "proposed_actions": proposed_actions,
            "requires_human": requires_human,
            "confidence": round(confidence, 3),
        }
        if escalation_reason:
            result["escalation_reason"] = escalation_reason
        return result

    @staticmethod
    def _draft_reply(parsed: dict, citations: list[str]) -> str:
        if citations:
            return ("Thanks for reaching out. Based on our documentation "
                    f"({', '.join(citations[:3])}), here is how to proceed. "
                    "Let us know if you need anything else.")
        return ("Thanks for reaching out — we've reviewed your request and are happy "
                "to help. Please let us know if you have further questions.")

    # ── helper ────────────────────────────────────────────────────
    @staticmethod
    def _call(name: str, args: dict) -> ModelResponse:
        return ModelResponse(text="", tool_calls=[ToolCall(id=f"call_{name}", name=name,
                                                           args=args)],
                             stop_reason="tool_use", input_tokens=200, output_tokens=30)
