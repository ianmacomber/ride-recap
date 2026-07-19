"""Shared model-provider interface for vision scan and text JSON completion."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


VISION_SOURCES = frozenset({"gemini", "openai"})

_REQUEST_TIMEOUT_S = 180.0


# ─── Structured-output schemas (OpenAI; Gemini keeps free-form JSON) ───

class FrameScore(BaseModel):
    frame_index: int
    visual: int = Field(ge=1, le=5)
    action: int = Field(ge=1, le=5)
    clip_type: str = ""
    crop_x: int = Field(default=50, ge=0, le=100)
    reason: str = ""


class CoarseFrameBatch(BaseModel):
    frames: list[FrameScore] = Field(default_factory=list)


class ClipRubric(BaseModel):
    light: int = Field(ge=1, le=10)
    composition: int = Field(ge=1, le=10)
    motion: int = Field(ge=1, le=10)
    scenery: int = Field(ge=1, le=10)
    subject: int = Field(ge=1, le=10)
    peak_offset: float = Field(default=0.5, ge=0.0, le=1.0)
    clip_type: str = ""
    crop_x: int = Field(default=50, ge=0, le=100)
    reason: str = ""


class NarrativeIndices(BaseModel):
    """Wrapper for narrative_select — OpenAI Structured Outputs need an object root."""

    indices: list[int] = Field(default_factory=list)


class PromptChange(BaseModel):
    type: str = ""
    section: str = ""
    current_text: str = ""
    suggested_text: str = ""
    rationale: str = ""


class ScoreAdjustments(BaseModel):
    description: str = ""


class PromptEvalResult(BaseModel):
    """Schema matching prompts/eval JSON contract in prompt_eval._EVAL_TEMPLATE."""

    analysis: str = ""
    prompt_changes: list[PromptChange] = Field(default_factory=list)
    suggested_score_adjustments: ScoreAdjustments = Field(
        default_factory=ScoreAdjustments,
    )


@runtime_checkable
class ModelAdapter(Protocol):
    """One provider powers an entire run — vision scan and text JSON."""

    name: str
    model_id: str

    def score_frames(
        self, *, images: list[bytes], system: str, user: str,
    ) -> list[dict]:
        """Coarse pass: score N frames. Returns list of frame dicts."""
        ...

    def score_clip_rubric(
        self, *, images: list[bytes], system: str, user: str,
    ) -> dict | None:
        """Fine pass: rate one clip on the 5-dim rubric."""
        ...

    def complete_json(
        self,
        *,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
    ) -> dict | list:
        """Text-only JSON completion (narrative select, prompt eval)."""
        ...

    def is_transient_error(self, exc: BaseException) -> bool:
        """True if the exception is worth retrying with backoff."""
        ...
