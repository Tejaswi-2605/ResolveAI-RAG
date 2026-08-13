"""Tests for the structured-output contract."""

from __future__ import annotations

import pytest

from app.core.validation import (REQUIRED_FIELDS, ValidationFailure,
                                 extract_json, fallback_result,
                                 repair_instruction, validate_triage_result)


def _valid(**overrides) -> dict:
    base = {
        "intent": "how_to", "priority": "normal", "sentiment": "neutral",
        "summary": "Customer asked how to schedule a report.",
        "suggested_reply": "Here is how you schedule a recurring report.",
        "citations": ["kb_001#02"], "proposed_actions": [],
        "requires_human": False, "confidence": 0.8,
    }
    return {**base, **overrides}


# ── stage 1: tolerate noise ───────────────────────────────────────
def test_plain_json_is_parsed():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_a_markdown_fence_is_stripped():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_a_bare_fence_is_stripped():
    assert extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_chatty_preamble_is_ignored():
    assert extract_json('Sure! Here is the result:\n{"a": 1}') == {"a": 1}


def test_trailing_commentary_is_ignored():
    """raw_decode stops at the end of the first object."""
    assert extract_json('{"a": 1}\nHope that helps!') == {"a": 1}


def test_empty_output_is_a_validation_failure():
    with pytest.raises(ValidationFailure, match="empty model output"):
        extract_json("")


def test_text_with_no_json_fails():
    with pytest.raises(ValidationFailure, match="no JSON object"):
        extract_json("I am afraid I cannot help with that.")


def test_malformed_json_fails():
    with pytest.raises(ValidationFailure, match="could not decode"):
        extract_json('{"a": 1, }}}')


def test_a_top_level_array_is_rejected():
    with pytest.raises(ValidationFailure, match="must be an object"):
        extract_json('[{"a": 1}]')


# ── stage 2: validate hard ────────────────────────────────────────
def test_a_valid_result_passes():
    assert validate_triage_result(_valid())["intent"] == "how_to"


def test_confidence_is_normalised():
    assert validate_triage_result(_valid(confidence=0.87654))["confidence"] == 0.877


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_every_required_field_is_enforced(field):
    data = _valid()
    del data[field]
    with pytest.raises(ValidationFailure, match=f"missing required field '{field}'"):
        validate_triage_result(data)


@pytest.mark.parametrize("field,bad", [
    ("intent", "not_a_real_intent"),
    ("priority", "extremely_urgent"),
    ("sentiment", "confused"),
])
def test_enum_fields_are_enforced(field, bad):
    with pytest.raises(ValidationFailure, match=field):
        validate_triage_result(_valid(**{field: bad}))


@pytest.mark.parametrize("bad", [-0.1, 1.5, "high", None])
def test_confidence_must_be_a_number_in_range(bad):
    with pytest.raises(ValidationFailure, match="confidence"):
        validate_triage_result(_valid(confidence=bad))


def test_a_boolean_confidence_is_rejected():
    """bool is an int in Python; "confidence: true" is nonsense."""
    with pytest.raises(ValidationFailure, match="confidence must be a number"):
        validate_triage_result(_valid(confidence=True))


def test_requires_human_must_be_a_real_boolean():
    with pytest.raises(ValidationFailure, match="must be a boolean"):
        validate_triage_result(_valid(requires_human="yes"))


def test_citations_must_be_an_array():
    with pytest.raises(ValidationFailure, match="citations must be an array"):
        validate_triage_result(_valid(citations="kb_001"))


def test_a_short_reply_is_rejected_when_sending_to_the_customer():
    with pytest.raises(ValidationFailure, match="too short"):
        validate_triage_result(_valid(suggested_reply="ok"))


def test_a_short_reply_is_allowed_when_escalating():
    """An escalated ticket legitimately has no customer-facing draft."""
    assert validate_triage_result(_valid(suggested_reply="",
                                         requires_human=True))["suggested_reply"] == ""


def test_a_refund_without_human_review_is_rejected():
    """
    The business rule, enforced a second time. Defence in depth: the agent
    enforces it too, and neither trusts the other.
    """
    with pytest.raises(ValidationFailure, match="requires_human must be true"):
        validate_triage_result(_valid(
            proposed_actions=[{"action_type": "issue_refund", "amount_cents": 100}]))


def test_a_refund_with_human_review_passes():
    assert validate_triage_result(_valid(
        proposed_actions=[{"action_type": "issue_refund"}],
        requires_human=True, suggested_reply=""))


def test_all_errors_are_collected_at_once():
    """
    One repair round-trip must be able to fix everything, rather than
    discovering problems one expensive turn at a time.
    """
    with pytest.raises(ValidationFailure) as caught:
        validate_triage_result({"intent": "bogus", "priority": "bogus"})
    assert len(caught.value.errors) > 3


# ── stage 3: repair, then fall back ───────────────────────────────
def test_the_repair_prompt_names_every_error():
    instruction = repair_instruction(ValidationFailure(["error one", "error two"]))
    assert "error one" in instruction and "error two" in instruction
    assert "ONLY the corrected JSON" in instruction


def test_the_fallback_is_itself_schema_valid():
    """
    The trick that gives the UI one shape: the fallback satisfies the same
    contract as a successful result, so it renders through the same code path.
    """
    result = fallback_result("model exploded")
    assert validate_triage_result(result) == result


def test_the_fallback_routes_to_a_human():
    result = fallback_result("model exploded")
    assert result["requires_human"] is True
    assert result["confidence"] == 0.0
    assert result["proposed_actions"] == []
    assert result["escalation_reason"] == "model exploded"
