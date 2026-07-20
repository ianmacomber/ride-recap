"""OpenAI Responses API implementation of ModelAdapter."""

from __future__ import annotations

import base64

from .base import (
    ClipRubric,
    CoarseFrameBatch,
    NarrativeIndices,
    PromptEvalResult,
    _REQUEST_TIMEOUT_S,
)


class OpenAIAdapter:
    """openai Responses API adapter with Structured Outputs for all passes."""

    name = "openai"

    def __init__(self, *, api_key: str, model: str) -> None:
        self.model_id = model
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, timeout=_REQUEST_TIMEOUT_S)

    def _image_input(self, images: list[bytes], user: str) -> list[dict]:
        content: list[dict] = []
        for img in images:
            b64 = base64.b64encode(img).decode("ascii")
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{b64}",
            })
        content.append({"type": "input_text", "text": user})
        return content

    def score_frames(
        self, *, images: list[bytes], system: str, user: str,
    ) -> list[dict]:
        # Prompt asks for a bare array; Structured Outputs need an object root.
        system_wrapped = (
            f"{system}\n\n"
            "Return a JSON object with key \"frames\" whose value is the "
            "JSON array described above (empty array if nothing qualifies)."
        )
        response = self._client.responses.parse(
            model=self.model_id,
            instructions=system_wrapped,
            input=[{"role": "user", "content": self._image_input(images, user)}],
            text_format=CoarseFrameBatch,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("OpenAI coarse pass returned no structured output")
        return [f.model_dump() for f in parsed.frames]

    def score_clip_rubric(
        self, *, images: list[bytes], system: str, user: str,
    ) -> dict | None:
        response = self._client.responses.parse(
            model=self.model_id,
            instructions=system,
            input=[{"role": "user", "content": self._image_input(images, user)}],
            text_format=ClipRubric,
        )
        parsed = response.output_parsed
        if parsed is None:
            # Must raise — returning None is treated as a normal empty rubric
            # and can be cached permanently as "scanned, found nothing."
            raise ValueError("OpenAI fine pass returned no structured output")
        return parsed.model_dump()

    def complete_json(
        self,
        *,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
    ) -> dict | list:
        # system set → prompt_eval object; otherwise narrative index list.
        if system:
            instructions = (
                f"{system.rstrip()}\n\n"
                "Respond with a JSON object matching the requested schema "
                "(analysis, prompt_changes, suggested_score_adjustments)."
            )
            response = self._client.responses.parse(
                model=self.model_id,
                instructions=instructions,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                text_format=PromptEvalResult,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("OpenAI prompt-eval returned no structured output")
            return parsed.model_dump()

        instructions = (
            "Select clip indices as instructed. Respond with a JSON object "
            "{\"indices\": [<int>, ...]} — a flat array of candidate indices "
            "in narrative order (empty array if none)."
        )
        response = self._client.responses.parse(
            model=self.model_id,
            instructions=instructions,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            text_format=NarrativeIndices,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("OpenAI narrative select returned no structured output")
        return list(parsed.indices)

    def is_transient_error(self, exc: BaseException) -> bool:
        try:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )
        except ImportError:
            RateLimitError = APITimeoutError = APIConnectionError = InternalServerError = ()  # type: ignore[misc,assignment]

        if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)):
            return True

        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and (status == 429 or status >= 500):
            return True

        msg = str(exc)
        return any(code in msg for code in ("429", "500", "502", "503", "504", "timeout", "Timeout"))
