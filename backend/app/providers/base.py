"""
base.py — THE PROVIDER INTERFACE.

A "provider" is anything that can act as the model: Anthropic's API, or our
offline mock. The agent talks ONLY to this interface and never imports a
vendor SDK. Three concrete wins from that one decision:

  1. a free, deterministic mock — tests and CI need no API key and no network
  2. one place to enforce timeouts, retries and token accounting
  3. swapping vendors touches one file, not the agent

THE INTERNAL MESSAGE FORMAT (vendor-neutral)

    {"role": "user",      "content": "..."}
    {"role": "assistant", "content": "...", "tool_calls": [ToolCall, ...]}
    {"role": "tool",      "tool_call_id": "...", "name": "...",
                          "ok": bool, "content": "<observation>"}

Each provider translates THIS into its own API's shape. Owning our own format
is what keeps the agent uncoupled from any vendor's message schema — and it
means a vendor changing their schema is a one-file fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """One request from the model to run a tool."""

    id: str        # unique, so the result can be matched back to the request
    name: str      # which tool, e.g. "search_knowledge_base"
    args: dict     # the arguments the model chose — UNTRUSTED until validated


@dataclass
class ModelResponse:
    """
    One reply from the model. It is EITHER tool calls OR final text.

    Token counts are recorded uniformly across providers so cost tracking does
    not depend on which vendor answered.
    """

    text: str = ""
    tool_calls: list = field(default_factory=list)   # list[ToolCall]
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0


# ── the exception hierarchy ───────────────────────────────────────
# The agent's retry logic keys off these types, so it never has to know a
# vendor's exception class names.
class ProviderError(Exception):
    """Base class for provider problems. NOT retried — assume a real fault."""


class ProviderTimeout(ProviderError):
    """The model took too long. Retried with backoff."""


class ProviderRateLimit(ProviderError):
    """Rate limited. Retried with backoff."""


class BaseProvider:
    """The contract: a name, a model id, and `complete()`."""

    name: str = "base"
    model: str = "none"

    def complete(self, system: str, messages: list, tools: list,
                 timeout_s: float = 30.0) -> ModelResponse:
        """Return the model's next response given the conversation so far."""
        raise NotImplementedError("providers must implement complete()")
