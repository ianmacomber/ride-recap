"""Gemini implementation of ModelAdapter."""

from __future__ import annotations

from ..utils import parse_json_response
from .base import _REQUEST_TIMEOUT_S


class GeminiAdapter:
    """google-genai backed adapter. Behavior matches the pre-adapter call sites."""

    name = "gemini"

    def __init__(self, *, api_key: str, model: str) -> None:
        self.model_id = model
        from google import genai
        self._client = genai.Client(api_key=api_key)

    def score_frames(
        self, *, images: list[bytes], system: str, user: str,
    ) -> list[dict]:
        from google.genai import types

        parts = [
            types.Part.from_bytes(data=img, mime_type="image/jpeg")
            for img in images
        ]
        parts.append(types.Part.from_text(text=user))

        response = self._client.models.generate_content(
            model=self.model_id,
            contents=[types.Content(parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=system,
                http_options=types.HttpOptions(
                    timeout=int(_REQUEST_TIMEOUT_S * 1000),
                ),
            ),
        )
        result = parse_json_response(response.text)
        return result if isinstance(result, list) else []

    def score_clip_rubric(
        self, *, images: list[bytes], system: str, user: str,
    ) -> dict | None:
        from google.genai import types

        parts = [
            types.Part.from_bytes(data=img, mime_type="image/jpeg")
            for img in images
        ]
        parts.append(types.Part.from_text(text=user))

        response = self._client.models.generate_content(
            model=self.model_id,
            contents=[types.Content(parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=system,
                http_options=types.HttpOptions(
                    timeout=int(_REQUEST_TIMEOUT_S * 1000),
                ),
            ),
        )
        result = parse_json_response(response.text)
        if isinstance(result, dict):
            return result
        if isinstance(result, list) and result:
            first = result[0]
            return first if isinstance(first, dict) else None
        return None

    def complete_json(
        self,
        *,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
    ) -> dict | list:
        from google.genai import types

        config_kwargs: dict = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if system:
            # prompt_eval path — system instruction, no thinking budget
            config_kwargs["system_instruction"] = system
        else:
            # narrative_select path — thinking budget, no system instruction
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=256,
            )

        response = self._client.models.generate_content(
            model=self.model_id,
            contents=[types.Content(parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return parse_json_response(response.text)

    def is_transient_error(self, exc: BaseException) -> bool:
        msg = str(exc)
        return any(code in msg for code in ("503", "429", "UNAVAILABLE"))
