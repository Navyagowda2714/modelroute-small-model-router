"""Choose an LLM provider from environment configuration."""

from __future__ import annotations

import logging
import os

from .base import LLMClient
from .offline import Handler, OfflineProvider
from .providers import AnthropicProvider, OllamaProvider, OpenAIProvider

log = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1",
    "offline": "offline-heuristic",
}
SMALL_MODELS = {"anthropic": "claude-haiku-5-5", "openai": "gpt-4o-mini"}


def get_llm_client(provider: str | None = None, model: str | None = None,
                   offline_handler: Handler | None = None) -> LLMClient:
    """Build an :class:`LLMClient`.

    ``provider`` defaults to ``$LLM_PROVIDER`` (anthropic|openai|ollama|offline), ``model`` to
    ``$LLM_MODEL``. If a hosted provider is requested but no API key is present we fall back
    to the deterministic offline provider and log a warning instead of failing.
    """
    provider = (provider or os.getenv("LLM_PROVIDER") or "offline").lower()
    model = model or os.getenv("LLM_MODEL") or None

    if provider == "anthropic":
        if os.getenv("ANTHROPIC_API_KEY"):
            return AnthropicProvider(model=model or DEFAULT_MODELS["anthropic"])
        log.warning("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set; using offline mode")
    elif provider == "openai":
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIProvider(model=model or DEFAULT_MODELS["openai"])
        log.warning("LLM_PROVIDER=openai but OPENAI_API_KEY is not set; using offline mode")
    elif provider == "ollama":
        return OllamaProvider(model=model or DEFAULT_MODELS["ollama"],
                              base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    elif provider != "offline":
        log.warning("Unknown LLM_PROVIDER=%r; using offline mode", provider)
    return OfflineProvider(handler=offline_handler)
