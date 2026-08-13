"""
providers/__init__.py — THE PROVIDER FACTORY.

`get_provider()` is the one function the rest of the app calls to obtain a
model. It reads MODEL_PROVIDER from configuration and returns the right
implementation.

SAFE DEFAULT: if the anthropic provider is requested but unusable — no API key,
SDK missing — we log a warning and fall back to the mock, so the application
never crashes merely because a key is absent. Pass `strict=True` to disable
that fallback, which is what a test asserting the real provider is wired up
should do.

Note providers live at `app/providers/`, not inside `app/core/`. A vendor
adapter is infrastructure, not agent logic; keeping it out of `core` makes the
dependency direction obvious at a glance.
"""

from __future__ import annotations

import logging

from app.config import Settings, get_settings
from app.providers.base import (BaseProvider, ModelResponse, ProviderError,
                                ProviderRateLimit, ProviderTimeout, ToolCall)
from app.providers.mock import MockProvider

logger = logging.getLogger("resolveai.providers")

__all__ = [
    "BaseProvider", "ModelResponse", "ToolCall",
    "ProviderError", "ProviderTimeout", "ProviderRateLimit",
    "MockProvider", "get_provider",
]


def get_provider(name: str | None = None, strict: bool = False,
                 settings: Settings | None = None, **kwargs) -> BaseProvider:
    """
    Return a provider instance.

    name:     "mock" or "anthropic"; defaults to the MODEL_PROVIDER setting.
    strict:   if True, do not fall back to the mock — raise instead.
    """
    settings = settings or get_settings()
    name = (name or settings.model_provider or "mock").lower()

    if name == "mock":
        kwargs.setdefault("failure_mode", settings.mock_failure_mode)
        return MockProvider(**kwargs)

    if name == "anthropic":
        try:
            from app.providers.anthropic_provider import AnthropicProvider
            kwargs.setdefault("model", settings.anthropic_model)
            return AnthropicProvider(**kwargs)
        except ProviderError as exc:
            if strict:
                raise
            logger.warning("anthropic provider unusable (%s); falling back to mock", exc)
            return MockProvider(failure_mode=settings.mock_failure_mode)

    logger.warning("unknown provider '%s'; falling back to mock", name)
    return MockProvider(failure_mode=settings.mock_failure_mode)
