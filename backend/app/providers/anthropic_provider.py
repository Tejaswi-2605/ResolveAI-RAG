"""
anthropic_provider.py — THE REAL MODEL, behind the same interface as the mock.

Two translation jobs live here, and only here. That is the entire value of the
provider abstraction: every vendor-specific detail is isolated in one file,
so the agent's code is identical whether it runs offline or against a live API.

  1. OUR message format → Anthropic's typed content blocks.
  2. Anthropic's exceptions → our ProviderTimeout / ProviderRateLimit /
     ProviderError, so the agent's retry policy never mentions a vendor.

The SDK is imported LAZILY, inside methods. The rest of the app and the whole
test suite therefore run fine with no `anthropic` package and no API key.
"""

from __future__ import annotations

import os

from app.providers.base import (BaseProvider, ModelResponse, ProviderError,
                                ProviderRateLimit, ProviderTimeout, ToolCall)

DEFAULT_MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 1024


class AnthropicProvider(BaseProvider):
    """Calls Anthropic's Messages API with tool use."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        if not self.api_key:
            # Raised here so the factory can decide whether to fall back.
            raise ProviderError("anthropic: ANTHROPIC_API_KEY is not set")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ProviderError(f"anthropic SDK not installed: {exc}") from exc
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    @staticmethod
    def _to_anthropic(messages: list) -> list:
        """
        Convert our messages into Anthropic's block format.

        The non-obvious part: a TOOL RESULT is delivered back to the model as a
        *user* turn containing a `tool_result` block, not as its own role.
        """
        converted = []
        for message in messages:
            role = message["role"]

            if role == "tool":
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": message.get("content", ""),
                        "is_error": not message.get("ok", True),
                    }],
                })
            elif role == "assistant" and message.get("tool_calls"):
                blocks = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": message["content"]})
                for call in message["tool_calls"]:
                    blocks.append({"type": "tool_use", "id": call.id,
                                   "name": call.name, "input": call.args})
                converted.append({"role": "assistant", "content": blocks})
            else:
                converted.append({"role": role, "content": message.get("content", "")})
        return converted

    def complete(self, system, messages, tools, timeout_s=30.0) -> ModelResponse:
        client = self._get_client()
        import anthropic   # lazy again, for the exception types

        try:
            response = client.messages.create(
                model=self.model,
                system=system,
                messages=self._to_anthropic(messages),
                tools=tools,                 # already in the right shape
                max_tokens=MAX_TOKENS,
                timeout=timeout_s,
            )
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeout(str(exc)) from exc
        except anthropic.RateLimitError as exc:
            raise ProviderRateLimit(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"anthropic API error: {exc}") from exc

        text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name,
                                           args=dict(block.input)))

        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,      # real usage, for real cost
            output_tokens=response.usage.output_tokens,
        )
