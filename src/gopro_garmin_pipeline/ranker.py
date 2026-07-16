"""Rank cycling video clips using Gemini vision-language model.

Two-stage pipeline:
  Stage 1: Detect if the clip is visually interesting
  Stage 2: Rate visual excitement 1-10

Uses Gemini's native video upload for best results, falls back to
frame-based analysis if upload fails.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import Settings, get_settings
from .frame_sampler import sample_clip_frames
from .utils import parse_json_response

logger = logging.getLogger(__name__)

# --- Prompts ---

_DETECT_PROMPT = """You are analyzing a cycling video filmed with a GoPro mounted on a road bike in New York City.

Determine if this clip contains visually interesting content. Interesting means any of:
- Dynamic urban cycling scenes (traffic, bridges, skyline views)
- Close calls or exciting moments with traffic
- Beautiful scenery or dramatic lighting
- Group riding or peloton action
- Fast descents or technical riding
- Interesting street scenes or landmarks

Respond in JSON:
{
  "is_interesting": true/false,
  "description": "brief description of what's happening",
  "peak_frame_index": 0
}"""

_RATE_PROMPT = """You are rating how visually exciting this cycling video clip is from 1 to 10.

Context: This is filmed on a GoPro mounted on a road bike in New York City.

Rating tiers:
- SPECTACULAR (9-10): Jaw-dropping moments — near-miss, stunning skyline view, dramatic weather, incredible group ride dynamics
- GREAT (7-8): Very engaging — fast riding through interesting streets, beautiful bridges, notable landmarks
- GOOD (5-6): Solid content — pleasant riding scenes, moderate traffic interaction, decent scenery
- MEDIOCRE (3-4): Unremarkable — plain road, no notable features, routine riding
- BORING (1-2): Nothing happening — stopped at light, staring at pavement, stationary

Respond in JSON:
{
  "excitement_rating": 7,
  "tier": "GREAT",
  "description": "Brief description of what makes this exciting or not",
  "best_moment": "Brief description of the single best visual moment"
}"""

# --- Keyword boosting (post-hoc adjustment) ---

_BOOST_KEYWORDS = {
    # Strong positive signals
    "near-miss": 2, "close call": 2, "skyline": 1.5, "sunset": 1.5, "sunrise": 1.5,
    "bridge": 1, "peloton": 1.5, "sprint": 1, "descen": 1,
    "manhattan": 1, "brooklyn bridge": 2, "central park": 1.5,
    "george washington bridge": 1.5, "hudson": 1, "east river": 1,
    # Negative signals
    "stopped": -1, "red light": -1, "waiting": -1.5, "stationary": -2,
    "pavement": -1, "nothing": -1.5,
}


@dataclass
class ClipScore:
    """Result of VLM scoring for a clip."""

    is_interesting: bool
    excitement_rating: float  # 1-10
    tier: str
    description: str
    best_moment: str = ""
    raw_rating: float = 0.0  # before keyword boost
    keyword_boost: float = 0.0
    pipeline_version: str = ""
    model_used: str = ""


def _apply_keyword_boost(rating: float, description: str) -> tuple[float, float]:
    """Adjust rating based on keywords in the VLM description."""
    boost = 0.0
    desc_lower = description.lower()
    for keyword, adjustment in _BOOST_KEYWORDS.items():
        if keyword in desc_lower:
            boost += adjustment

    adjusted = max(1.0, min(10.0, rating + boost))
    return adjusted, boost


# --- Gemini API ---

def _call_gemini_with_video(video_path: Path, prompt: str, settings: Settings) -> str:
    """Call Gemini with native video upload."""
    from google import genai

    client = genai.Client(api_key=settings.gemini_api_key)
    uploaded = client.files.upload(file=video_path)
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[uploaded, prompt],
    )
    return response.text


def _call_gemini_with_frames(frames_b64: list[str], prompt: str, settings: Settings) -> str:
    """Call Gemini with base64 frames (fallback if video upload fails)."""
    import base64

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)

    parts = []
    for frame_b64 in frames_b64:
        parts.append(types.Part.from_bytes(
            data=base64.b64decode(frame_b64),
            mime_type="image/png",
        ))
    parts.append(types.Part.from_text(text=prompt))

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[types.Content(parts=parts)],
    )
    return response.text


def _call_gemini(
    video_path: Path, frames_b64: list[str], prompt: str, settings: Settings
) -> str:
    """Call Gemini, preferring native video upload with frame fallback."""
    try:
        return _call_gemini_with_video(video_path, prompt, settings)
    except Exception:
        logger.warning("Gemini video upload failed, falling back to frames")
        return _call_gemini_with_frames(frames_b64, prompt, settings)


# --- Scoring pipeline ---

def score_clip(
    video_path: str | Path,
    settings: Settings | None = None,
) -> ClipScore:
    """Score a video clip using the two-stage Gemini pipeline.

    Stage 1: Detect if clip contains interesting content.
    Stage 2: Rate visual excitement 1-10 (only if Stage 1 passes).

    Args:
        video_path: Path to the video clip.
        settings: Pipeline settings. Uses defaults if None.

    Returns:
        ClipScore with rating, description, and metadata.
    """
    if settings is None:
        settings = get_settings()

    video_path = Path(video_path)

    # Extract frames (needed for fallback and as a safety net)
    frame_paths, frames_b64 = sample_clip_frames(
        video_path,
        fps=settings.sample_fps,
        max_width=settings.sample_max_width,
        max_frames=settings.max_frames_per_clip,
        motion_ratio=settings.motion_frame_ratio,
    )

    if not frames_b64:
        return ClipScore(
            is_interesting=False,
            excitement_rating=0,
            tier="ERROR",
            description="No frames extracted",
            pipeline_version=settings.score_version,
            model_used=settings.gemini_model,
        )

    try:
        # Stage 1: Detect
        detect_raw = _call_gemini(video_path, frames_b64, _DETECT_PROMPT, settings)
        detect_result = parse_json_response(detect_raw)
        is_interesting = detect_result.get("is_interesting", False)
        description = detect_result.get("description", "")

        if not is_interesting:
            return ClipScore(
                is_interesting=False,
                excitement_rating=2.0,
                tier="BORING",
                description=description,
                raw_rating=2.0,
                pipeline_version=settings.score_version,
                model_used=settings.gemini_model,
            )

        # Stage 2: Rate
        rate_raw = _call_gemini(video_path, frames_b64, _RATE_PROMPT, settings)
        rate_result = parse_json_response(rate_raw)
        raw_rating = float(rate_result.get("excitement_rating", 5))
        tier = rate_result.get("tier", "GOOD")
        rate_desc = rate_result.get("description", description)
        best_moment = rate_result.get("best_moment", "")

        full_desc = f"{description}. {rate_desc}" if rate_desc != description else description
        adjusted_rating, boost = _apply_keyword_boost(raw_rating, full_desc)

        return ClipScore(
            is_interesting=True,
            excitement_rating=adjusted_rating,
            tier=tier,
            description=full_desc,
            best_moment=best_moment,
            raw_rating=raw_rating,
            keyword_boost=boost,
            pipeline_version=settings.score_version,
            model_used=settings.gemini_model,
        )

    except Exception as e:
        logger.exception("Gemini scoring failed for %s", video_path)
        return ClipScore(
            is_interesting=False,
            excitement_rating=0,
            tier="ERROR",
            description=f"Scoring failed: {e}",
            pipeline_version=settings.score_version,
            model_used=settings.gemini_model,
        )
    finally:
        if frame_paths:
            tmpdir = frame_paths[0].parent
            shutil.rmtree(tmpdir, ignore_errors=True)


def score_clips(
    video_paths: list[Path],
    settings: Settings | None = None,
) -> list[tuple[Path, ClipScore]]:
    """Score multiple clips and return (path, score) pairs sorted by excitement rating."""
    if settings is None:
        settings = get_settings()

    results: list[tuple[Path, ClipScore]] = []
    for path in video_paths:
        logger.info("Scoring %s...", path.name)
        result = score_clip(path, settings)
        results.append((path, result))
        logger.info(
            "  %s: %.1f (%s) - %s",
            path.name, result.excitement_rating, result.tier, result.description[:80],
        )

    results.sort(key=lambda r: r[1].excitement_rating, reverse=True)
    return results
