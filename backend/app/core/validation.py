"""
validation.py — THE STRUCTURED-OUTPUT CONTRACT.

"The model usually returns valid JSON" is not a contract. A user interface
needs EXACTLY ONE shape to render, on every single run, including the runs
where the model returns an apology, a markdown fence, or nothing at all.

THREE STAGES

  1. TOLERATE NOISE — `extract_json()` digs the object out of messy output:
     ```json fences, a chatty preamble, trailing commentary.

  2. VALIDATE HARD — `validate_triage_result()` checks every field and
     collects ALL the errors before raising. Collecting rather than failing
     fast is what lets ONE repair round-trip fix everything, instead of
     discovering problems one expensive turn at a time.

  3. REPAIR, THEN FALL BACK — `repair_instruction()` gives the model one
     chance to fix its own output. If that fails, `fallback_result()` returns
     a safe object that is ITSELF schema-valid and routes the ticket to a
     human. That is the trick: because the fallback satisfies the same
     contract, the frontend renders success, repair and total failure with one
     code path.

The enum lists live HERE and are imported by `prompts.py`, so the prompt and
the validator cannot drift apart. If you add an intent, the prompt learns
about it automatically.
"""

from __future__ import annotations

import json

# ── the allowed vocabularies ──────────────────────────────────────
INTENTS = [
    "billing_refund", "billing_question", "bug_report", "outage", "how_to",
    "feature_request", "account_access", "cancellation", "security_legal", "other",
]
PRIORITIES = ["low", "normal", "high", "urgent"]
SENTIMENTS = ["angry", "frustrated", "neutral", "positive"]

# Every triage result must carry these nine fields.
REQUIRED_FIELDS = [
    "intent", "priority", "sentiment", "summary", "suggested_reply",
    "citations", "proposed_actions", "requires_human", "confidence",
]

MIN_REPLY_CHARS = 15


class ValidationFailure(Exception):
    """
    Model output failed the contract.

    Carries a LIST of errors, not one, so a single re-prompt can address them
    all at once.
    """

    def __init__(self, errors):
        self.errors = errors if isinstance(errors, list) else [errors]
        super().__init__("; ".join(self.errors))


def extract_json(text: str) -> dict:
    """
    Pull the first JSON object out of raw model text.

    `raw_decode` stops cleanly at the end of the first object, so trailing
    prose after the JSON is simply ignored rather than breaking the parse.
    """
    if not text or not text.strip():
        raise ValidationFailure("empty model output — no JSON to parse")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Drop a leading ```json / ``` fence and anything after the closing one.
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]

    # Start at the first "{" OR "[", whichever comes first. Anchoring only on
    # "{" would silently pull the first object OUT of a top-level array, which
    # would quietly accept output that does not meet the contract.
    candidates = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
    if not candidates:
        raise ValidationFailure("no JSON object found in model output")
    start = min(candidates)

    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise ValidationFailure(f"could not decode JSON: {exc}")

    if not isinstance(obj, dict):
        raise ValidationFailure("top-level JSON must be an object")
    return obj


def contains_refund(actions) -> bool:
    """True if any proposed action is an issue_refund (object or bare string)."""
    for action in actions or []:
        if isinstance(action, dict) and action.get("action_type") == "issue_refund":
            return True
        if action == "issue_refund":
            return True
    return False


def validate_triage_result(data: dict) -> dict:
    """
    Validate a triage result, collecting every error, then return a normalised copy.
    """
    if not isinstance(data, dict):
        raise ValidationFailure("result must be a JSON object")

    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"missing required field '{field}'")

    if "intent" in data and data["intent"] not in INTENTS:
        errors.append(f"intent must be one of {INTENTS}")
    if "priority" in data and data["priority"] not in PRIORITIES:
        errors.append(f"priority must be one of {PRIORITIES}")
    if "sentiment" in data and data["sentiment"] not in SENTIMENTS:
        errors.append(f"sentiment must be one of {SENTIMENTS}")

    # In Python a bool IS an int, so `isinstance(True, int)` is True. Without
    # the explicit bool guard, a confidence of `true` would sail through.
    confidence = data.get("confidence")
    if "confidence" in data:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            errors.append("confidence must be a number")
        elif not 0.0 <= confidence <= 1.0:
            errors.append("confidence must be between 0 and 1")

    requires_human = data.get("requires_human")
    if "requires_human" in data and not isinstance(requires_human, bool):
        errors.append("requires_human must be a boolean (true/false)")

    if "citations" in data and not isinstance(data["citations"], list):
        errors.append("citations must be an array")
    if "proposed_actions" in data and not isinstance(data["proposed_actions"], list):
        errors.append("proposed_actions must be an array")

    # A too-short reply is only acceptable when the ticket goes to a human —
    # an escalated ticket legitimately has no customer-facing draft.
    reply = data.get("suggested_reply", "")
    if isinstance(reply, str) and len(reply) < MIN_REPLY_CHARS and requires_human is not True:
        errors.append(
            f"suggested_reply is too short (min {MIN_REPLY_CHARS} chars) "
            "unless requires_human is true")

    # BUSINESS RULE, enforced here as well as in the agent: proposing a refund
    # requires human review. Defence in depth — two independent checks.
    if contains_refund(data.get("proposed_actions")) and requires_human is not True:
        errors.append(
            "requires_human must be true when proposed_actions contains an issue_refund")

    if errors:
        raise ValidationFailure(errors)

    normalised = dict(data)
    normalised["confidence"] = round(float(confidence), 3)
    return normalised


def repair_instruction(errors) -> str:
    """Build the ONE re-prompt the model gets to fix its own output."""
    error_list = errors.errors if isinstance(errors, ValidationFailure) else errors
    bullets = "\n".join(f"- {error}" for error in error_list)
    return (
        "Your previous JSON was invalid. Fix these problems:\n"
        f"{bullets}\n"
        "Return ONLY the corrected JSON object, with no markdown fences and no commentary."
    )


def fallback_result(reason: str) -> dict:
    """
    A safe, SCHEMA-VALID result for when the model cannot produce one.

    Routes to a human, carries zero confidence and an empty reply. The
    frontend renders it exactly like any other result — one shape, always.
    """
    return {
        "intent": "other",
        "priority": "normal",
        "sentiment": "neutral",
        "summary": f"Automated fallback: {reason}",
        "suggested_reply": "",          # allowed, because requires_human is True
        "citations": [],
        "proposed_actions": [],
        "requires_human": True,
        "confidence": 0.0,
        "escalation_reason": reason,
    }
