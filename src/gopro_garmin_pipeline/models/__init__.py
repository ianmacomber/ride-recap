"""Model provider adapters — one provider powers an entire pipeline run."""

from __future__ import annotations

from .base import VISION_SOURCES, ModelAdapter
from .gemini import GeminiAdapter
from .local_vlm import LocalOpenAIAdapter
from .openai import OpenAIAdapter

__all__ = [
    "VISION_SOURCES",
    "ModelAdapter",
    "GeminiAdapter",
    "OpenAIAdapter",
    "LocalOpenAIAdapter",
    "get_model_adapter",
    "provider_api_key",
    "provider_model_id",
    "cache_dir_name",
]


def provider_api_key(settings) -> str:
    """Return the API key for the selected MODEL_PROVIDER (empty if unset).

    For ``local``, a truthy sentinel is returned only when both base URL and
    model are configured — the scan/eval gates treat an empty string as
    "skip this AI feature".
    """
    provider = (settings.model_provider or "gemini").lower()
    if provider == "openai":
        return settings.openai_api_key or ""
    if provider == "gemini":
        return settings.gemini_api_key or ""
    if provider == "local":
        if settings.local_base_url and settings.local_model:
            return settings.local_api_key or "local"
        return ""
    return ""


def provider_model_id(settings) -> str:
    """Return the configured model id for the active provider."""
    provider = (settings.model_provider or "gemini").lower()
    if provider == "gemini":
        return settings.gemini_model
    if provider == "openai":
        return settings.openai_model
    if provider == "local":
        return settings.local_model
    raise ValueError(
        f"Unknown MODEL_PROVIDER={provider!r}; expected 'gemini', 'openai', or 'local'"
    )


def cache_dir_name(provider: str) -> str:
    """Per-provider cache directory name (e.g. .gemini_cache, .openai_cache)."""
    return f".{provider}_cache"


def get_model_adapter(settings) -> ModelAdapter:
    """Construct a fresh adapter for the configured provider.

    Callers should create one adapter per clip / call site so nested
    thread pools do not share a single client instance. The local adapter
    still shares a process-wide request semaphore across instances.
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
    if provider == "local":
        model = (settings.local_model or "").strip()
        if not model:
            raise ValueError("LOCAL_MODEL is required when MODEL_PROVIDER=local")
        base_url = (settings.local_base_url or "").strip()
        if not base_url:
            raise ValueError("LOCAL_BASE_URL is required when MODEL_PROVIDER=local")
        return LocalOpenAIAdapter(
            base_url=base_url,
            api_key=settings.local_api_key or "local",
            model=model,
            max_concurrency=settings.local_max_concurrency,
            timeout=settings.local_timeout_seconds,
        )
    raise ValueError(
        f"Unknown MODEL_PROVIDER={provider!r}; expected 'gemini', 'openai', or 'local'"
    )
