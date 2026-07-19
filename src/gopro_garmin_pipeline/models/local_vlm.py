"""Local OpenAI-compatible VLM adapter (MLX-VLM, Ollama, LM Studio, vLLM).

Uses Chat Completions with json_schema structured outputs when the server
supports them (MLX-VLM). Falls back once to free-form JSON parsing for less
capable servers. A process-wide semaphore caps concurrent requests because
the scanner constructs a fresh adapter per clip.
"""

from __future__ import annotations

import base64
import json
import re
import threading
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..utils import parse_json_response
from .base import (
    ClipRubric,
    CoarseFrameBatch,
    NarrativeIndices,
    PromptEvalResult,
)

T = TypeVar("T", bound=BaseModel)

# Process-wide: each clip builds its own adapter, so a per-instance
# semaphore cannot throttle the scanner's nested thread pools.
_LOCAL_REQUEST_SEMAPHORE: threading.Semaphore | None = None
_LOCAL_SEMAPHORE_LIMIT: int | None = None
_LOCAL_SEMAPHORE_LOCK = threading.Lock()

_DEFAULT_MAX_CONCURRENCY = 1
_DEFAULT_TIMEOUT_S = 600.0


def _get_local_request_semaphore(max_concurrency: int) -> threading.Semaphore:
    """Return the shared request semaphore, initializing it on first use."""
    global _LOCAL_REQUEST_SEMAPHORE, _LOCAL_SEMAPHORE_LIMIT
    with _LOCAL_SEMAPHORE_LOCK:
        if _LOCAL_REQUEST_SEMAPHORE is None:
            limit = max(1, int(max_concurrency))
            _LOCAL_REQUEST_SEMAPHORE = threading.Semaphore(limit)
            _LOCAL_SEMAPHORE_LIMIT = limit
        return _LOCAL_REQUEST_SEMAPHORE


def _reset_local_request_semaphore_for_tests() -> None:
    """Clear the process-wide semaphore (tests only)."""
    global _LOCAL_REQUEST_SEMAPHORE, _LOCAL_SEMAPHORE_LIMIT
    with _LOCAL_SEMAPHORE_LOCK:
        _LOCAL_REQUEST_SEMAPHORE = None
        _LOCAL_SEMAPHORE_LIMIT = None


def _response_format_for(model: type[BaseModel], name: str) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": model.model_json_schema(),
        },
    }


def _is_structured_format_rejected(exc: BaseException) -> bool:
    """True when the server rejected response_format / json_schema."""
    status = getattr(exc, "status_code", None)
    msg = str(exc).lower()
    markers = (
        "response_format",
        "json_schema",
        "structured output",
        "structured_output",
        "unsupported",
        "invalid_request",
        "unknown parameter",
        "extra inputs are not permitted",
    )
    if any(m in msg for m in markers):
        # Auth / missing model must not trigger free-form fallback.
        if any(bad in msg for bad in ("api key", "unauthorized", "authentication", "model not found", "does not exist")):
            return False
        if status is None or status == 400 or status == 422:
            return True
    return False


def _is_connection_failure(exc: BaseException) -> bool:
    """True when the local server is unreachable (do not retry / backoff)."""
    try:
        from openai import APIConnectionError
    except ImportError:
        APIConnectionError = ()  # type: ignore[misc,assignment]

    if isinstance(exc, APIConnectionError):
        return True

    msg = str(exc).lower()
    return any(
        s in msg
        for s in (
            "connection refused",
            "failed to connect",
            "connect error",
            "connection error",
            "nodename nor servname",
            "name or service not known",
            "network is unreachable",
            "no route to host",
        )
    )


def _is_permanent_configuration_error(exc: BaseException) -> bool:
    """True when retrying cannot fix the local server/model configuration."""
    status = getattr(exc, "status_code", None)
    msg = str(exc).lower()
    markers = (
        "failed to load model",
        "model not found",
        "model does not exist",
        "invalid model",
        "invalid repo id",
        "repo id must use",
        "repository not found",
    )
    return status == 404 or any(marker in msg for marker in markers)


class LocalOpenAIAdapter:
    """Chat Completions adapter for local OpenAI-compatible VLM servers."""

    name = "local"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
        timeout: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self.model_id = model
        self.base_url = base_url.rstrip("/")
        self._timeout = float(timeout)
        self._semaphore = _get_local_request_semaphore(max_concurrency)
        from openai import OpenAI
        self._client = OpenAI(
            api_key=api_key or "local",
            base_url=self.base_url,
            timeout=self._timeout,
        )

    def _image_content(self, images: list[bytes], user: str) -> list[dict]:
        content: list[dict] = []
        for img in images:
            b64 = base64.b64encode(img).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        content.append({"type": "text", "text": user})
        return content

    def _messages(
        self,
        *,
        system: str | None,
        user_content: str | list[dict],
    ) -> list[dict]:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _create(
        self,
        *,
        messages: list[dict],
        response_format: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        with self._semaphore:
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                if _is_connection_failure(exc):
                    raise ConnectionError(
                        f"Local VLM unreachable at {self.base_url} — "
                        "is the server running?"
                    ) from exc
                raise

    def _message_text(self, response: Any) -> str:
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("Local VLM returned no message content") from exc
        if not content or not str(content).strip():
            raise ValueError("Local VLM returned empty message content")
        return str(content)

    def _complete_structured(
        self,
        *,
        messages: list[dict],
        schema: type[T],
        schema_name: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        fmt = _response_format_for(schema, schema_name)
        try:
            response = self._create(
                messages=messages,
                response_format=fmt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            if not _is_structured_format_rejected(exc):
                raise
            response = self._create(
                messages=messages,
                response_format=None,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return self._parse_freeform_as(schema, self._message_text(response))

        text = self._message_text(response)
        try:
            return schema.model_validate_json(text)
        except (ValidationError, json.JSONDecodeError):
            # Some servers honor response_format loosely; validate via free-form
            # parse only if the text is still the right shape.
            return self._parse_freeform_as(schema, text)

    @staticmethod
    def _parse_json_or_raise(text: str) -> dict | list:
        """Parse model text as JSON; never silently return an empty sentinel."""
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()
        try:
            result = json.loads(cleaned)
            if isinstance(result, (dict, list)):
                return result
            raise ValueError(
                f"Local VLM returned JSON {type(result).__name__}, "
                "expected object or array"
            )
        except json.JSONDecodeError:
            pass

        parsed = parse_json_response(text)
        # parse_json_response returns {} / [] on failure — reject those
        # sentinels unless the cleaned text is literally an empty container.
        if parsed == {} and cleaned not in ("{}",):
            raise ValueError("Local VLM returned unparseable JSON")
        if parsed == [] and cleaned not in ("[]",):
            raise ValueError("Local VLM returned unparseable JSON")
        if not isinstance(parsed, (dict, list)):
            raise ValueError("Local VLM returned unparseable JSON")
        return parsed

    def _parse_freeform_as(self, schema: type[T], text: str) -> T:
        parsed = self._parse_json_or_raise(text)
        if schema is CoarseFrameBatch:
            if isinstance(parsed, list):
                try:
                    return schema.model_validate({"frames": parsed})  # type: ignore[return-value]
                except ValidationError as exc:
                    raise ValueError(
                        "Local VLM coarse pass returned invalid frame list"
                    ) from exc
            if isinstance(parsed, dict) and "frames" in parsed:
                try:
                    return schema.model_validate(parsed)  # type: ignore[return-value]
                except ValidationError as exc:
                    raise ValueError(
                        "Local VLM coarse pass returned invalid frames object"
                    ) from exc
            raise ValueError(
                "Local VLM coarse pass expected a JSON array of frames "
                f"(got {type(parsed).__name__})"
            )

        if schema is NarrativeIndices:
            if isinstance(parsed, list):
                if not all(isinstance(x, int) and not isinstance(x, bool) for x in parsed):
                    raise ValueError(
                        "Local VLM narrative select expected a JSON array of ints"
                    )
                return schema.model_validate({"indices": parsed})  # type: ignore[return-value]
            if isinstance(parsed, dict) and "indices" in parsed:
                try:
                    return schema.model_validate(parsed)  # type: ignore[return-value]
                except ValidationError as exc:
                    raise ValueError(
                        "Local VLM narrative select returned invalid indices"
                    ) from exc
            raise ValueError(
                "Local VLM narrative select expected a JSON array of indices "
                f"(got {type(parsed).__name__})"
            )

        if schema is ClipRubric:
            if isinstance(parsed, list):
                if not parsed or not isinstance(parsed[0], dict):
                    raise ValueError(
                        "Local VLM fine pass expected a rubric object "
                        "(got empty or non-object list)"
                    )
                parsed = parsed[0]
            if not isinstance(parsed, dict):
                raise ValueError(
                    "Local VLM fine pass expected a JSON object "
                    f"(got {type(parsed).__name__})"
                )
            try:
                return schema.model_validate(parsed)  # type: ignore[return-value]
            except ValidationError as exc:
                raise ValueError(
                    "Local VLM fine pass returned invalid rubric object"
                ) from exc

        # PromptEvalResult and other object schemas
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Local VLM expected a JSON object (got {type(parsed).__name__})"
            )
        try:
            return schema.model_validate(parsed)  # type: ignore[return-value]
        except ValidationError as exc:
            raise ValueError(
                f"Local VLM returned JSON that does not match {schema.__name__}"
            ) from exc

    def score_frames(
        self, *, images: list[bytes], system: str, user: str,
    ) -> list[dict]:
        # Structured Outputs need an object root; wrap the bare array.
        system_wrapped = (
            f"{system}\n\n"
            "Return a JSON object with key \"frames\" whose value is the "
            "JSON array described above (empty array if nothing qualifies)."
        )
        messages = self._messages(
            system=system_wrapped,
            user_content=self._image_content(images, user),
        )
        parsed = self._complete_structured(
            messages=messages,
            schema=CoarseFrameBatch,
            schema_name="coarse_frame_batch",
        )
        return [f.model_dump() for f in parsed.frames]

    def score_clip_rubric(
        self, *, images: list[bytes], system: str, user: str,
    ) -> dict | None:
        messages = self._messages(
            system=system,
            user_content=self._image_content(images, user),
        )
        parsed = self._complete_structured(
            messages=messages,
            schema=ClipRubric,
            schema_name="clip_rubric",
        )
        return parsed.model_dump()

    def complete_json(
        self,
        *,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
    ) -> dict | list:
        if system:
            instructions = (
                f"{system.rstrip()}\n\n"
                "Respond with a JSON object matching the requested schema "
                "(analysis, prompt_changes, suggested_score_adjustments)."
            )
            messages = self._messages(system=instructions, user_content=prompt)
            parsed = self._complete_structured(
                messages=messages,
                schema=PromptEvalResult,
                schema_name="prompt_eval_result",
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
            return parsed.model_dump()

        instructions = (
            "Select clip indices as instructed. Respond with a JSON object "
            "{\"indices\": [<int>, ...]} — a flat array of candidate indices "
            "in narrative order (empty array if none)."
        )
        messages = self._messages(system=instructions, user_content=prompt)
        parsed = self._complete_structured(
            messages=messages,
            schema=NarrativeIndices,
            schema_name="narrative_indices",
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        return list(parsed.indices)

    def is_transient_error(self, exc: BaseException) -> bool:
        # Connection refusal / unreachable host: fail fast — backoff wastes
        # ~155s per request when the local server is simply not running.
        if isinstance(exc, ConnectionError) or _is_connection_failure(exc):
            return False

        # MLX-VLM reports invalid model ids as HTTP 500 after attempting a
        # lazy model swap. Retrying only repeats the same expensive load and
        # obscures the actionable configuration error.
        if _is_permanent_configuration_error(exc):
            return False

        try:
            from openai import (
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )
        except ImportError:
            RateLimitError = APITimeoutError = InternalServerError = ()  # type: ignore[misc,assignment]

        if isinstance(exc, (RateLimitError, APITimeoutError, InternalServerError)):
            return True

        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and (status == 429 or status >= 500):
            return True

        msg = str(exc).lower()
        return any(
            code in msg
            for code in ("429", "500", "502", "503", "504", "timeout")
        )
