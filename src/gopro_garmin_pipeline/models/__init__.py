"""Model provider adapters — one provider powers an entire pipeline run."""

from __future__ import annotations

from .base import VISION_SOURCES, ModelAdapter
from .gemini import GeminiAdapter
from .openai import OpenAIAdapter

__all__ = [
    "VISION_SOURCES",
    "ModelAdapter",
    "GeminiAdapter",
    "OpenAIAdapter",
    "get_model_adapter",
    "provider_api_key",
    "cache_dir_name",
]


def provider_api_key(settings) -> str:
    """Return the API key for the selected MODEL_PROVIDER (empty if unset)."""
    provider = (settings.model_provider or "gemini").lower()
    if provider == "openai":
        return settings.openai_api_key or ""
    if provider == "gemini":
        return settings.gemini_api_key or ""
    return ""


def cache_dir_name(provider: str) -> str:
    """Per-provider cache directory name (e.g. .gemini_cache, .openai_cache)."""
    return f".{provider}_cache"


def get_model_adapter(settings) -> ModelAdapter:
    """Construct a fresh adapter for the configured provider.

    Callers should create one adapter per clip / call site so nested
    thread pools do not share a single client instance.
    """
    provider = (settings.model_provider or "gemini").lower()
    if provider == "gemini":
        key = settings.gemini_api_key
        if not key:
            raise ValueError("GEMINI_API_KEY is required when MODEL_PROVIDER=gemini")
        return GeminiAdapter(api_key=key, model=settings.gemini_model)
    if provider == "openai":
        key = settings.openai_api_key
        if not key:
            raise ValueError("OPENAI_API_KEY is required when MODEL_PROVIDER=openai")
        return OpenAIAdapter(api_key=key, model=settings.openai_model)
    raise ValueError(
        f"Unknown MODEL_PROVIDER={provider!r}; expected 'gemini' or 'openai'"
    )
