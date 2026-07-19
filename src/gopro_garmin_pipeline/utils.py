"""Shared constants, conversion helpers, and parsing utilities."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

# ─── Unit conversions ────────────────────────────────────────
MS_TO_MPH = 2.23694
M_TO_MILES = 1 / 1609.344
M_TO_FT = 3.28084


@contextmanager
def keep_system_awake(reason: str = "long-running pipeline"):
    """Prevent macOS from sleeping or throttling for the duration of a block.

    The 4K libx264 burns and vision-model scans take 10-30 minutes. If the
    display sleeps mid-run, macOS aggressively throttles the Python
    process via App Nap and Power Nap — clips that normally take 45s
    can run for 90+ minutes overnight.

    macOS ships `caffeinate` for exactly this. We spawn it as a child
    tied to our PID (`-w <pid>`), so it exits automatically when our
    process exits even if the context-manager finally never runs (Ctrl+C,
    crash, OOM kill). On non-macOS or when `caffeinate` is missing, this
    is a no-op.

    Yields True if caffeinate was started, False otherwise — callers can
    use the value to log whether the safety is active.
    """
    proc: subprocess.Popen | None = None
    if sys.platform == "darwin":
        try:
            proc = subprocess.Popen(
                ["caffeinate", "-di", "-w", str(os.getpid())],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (FileNotFoundError, OSError):
            proc = None
    try:
        yield proc is not None
    finally:
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    proc.kill()
                except OSError:
                    pass


def gopro_lrv_proxy(video_path: Path) -> Path | None:
    """Return the GoPro low-res proxy (.LRV) for a chapter file, if present.

    GoPro Hero cameras record GX01NNNN.MP4 (high-res HEVC) alongside
    GL01NNNN.LRV (low-res 848x480 H.264). The two files share identical
    timing and frame count; decoding the proxy is 10-20x faster than
    decoding 5.3K HEVC, which matters a lot for sparse frame extraction.

    Note the prefix swap: GX -> GL. A naive `with_suffix('.LRV')` produces
    GX01NNNN.LRV which does not exist on disk.
    """
    if video_path.suffix.upper() != ".MP4":
        return None
    stem = video_path.stem
    if not stem.startswith("GX"):
        return None
    proxy = video_path.with_name("GL" + stem[2:] + ".LRV")
    return proxy if proxy.exists() else None


def format_time(secs: float) -> str:
    """Format seconds into H:MM:SS or M:SS."""
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def compute_gradient(alt0: float, dist0: float, alt1: float, dist1: float) -> float:
    """Compute gradient (%) between two points.

    Returns 0.0 if horizontal distance is too small to be meaningful.
    """
    d_dist = dist1 - dist0
    if d_dist < 0.5:
        return 0.0
    return ((alt1 - alt0) / d_dist) * 100


LABEL_SCALE_VERSION = 2


def normalize_label_scale(label: dict) -> dict:
    """Return a label dict with visual/action on the v2 (1-10) scale.

    Labels written before scale_version=2 used a 1-5 scale; doubling
    them keeps relative ordering and lifts them into the 1-10 space all
    consumers now expect. Idempotent — already-v2 labels pass through
    unchanged. Non-label rating dicts (e.g. Gemini hits) should not
    flow through this helper.
    """
    if label.get("scale_version", 1) >= LABEL_SCALE_VERSION:
        return label
    out = dict(label)
    if "visual" in label:
        out["visual"] = label.get("visual", 0) * 2
    if "action" in label:
        out["action"] = label.get("action", 0) * 2
    out["scale_version"] = LABEL_SCALE_VERSION
    return out


# Labels rate a moment on visual/action. The v10 rubric rates it on five
# dims instead, so anything that wants to compare or score the two together
# has to fold one onto the other. `light` is left out for the reason
# composer._rubric_score leaves it out — it doesn't discriminate.
VISUAL_RUBRIC_DIMS = ("composition", "scenery")
ACTION_RUBRIC_DIMS = ("motion", "subject")


def _mean_dims(rubric: dict, dims: tuple[str, ...]) -> int:
    vals = [rubric.get(k) for k in dims]
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return 0
    return round(sum(vals) / len(vals))


def rating_visual_action(data: dict) -> tuple[int, int]:
    """Best available (visual, action) for a moment, on the 1-10 scale.

    Reads explicit visual/action when present — human labels and pre-v10
    Gemini hits both carry them. Otherwise folds a v10 rubric down. Returns
    (0, 0) when there's nothing to read, which is the honest answer for a
    telemetry-only candidate that was never looked at.

    A nested "label" dict is checked last so serialized candidates work.
    """
    visual = data.get("visual")
    action = data.get("action")
    if visual is not None or action is not None:
        return int(visual or 0), int(action or 0)

    rubric = data.get("rubric") or {}
    if rubric:
        return _mean_dims(rubric, VISUAL_RUBRIC_DIMS), _mean_dims(rubric, ACTION_RUBRIC_DIMS)

    nested = data.get("label") or {}
    if nested and nested is not data:
        return rating_visual_action(nested)
    return 0, 0


def parse_json_response(text: str) -> dict | list:
    """Robustly parse JSON from VLM output.

    Handles markdown fences, trailing commas, and partial responses.
    Returns an empty dict on failure for object-expected callers,
    or an empty list if the input looks like an array.
    """
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas
    cleaned = re.sub(r",\s*}", "}", text)
    cleaned = re.sub(r",\s*]", "]", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Extract first JSON object or array
    for pattern in (r"\[.*\]", r"\{[^{}]*\}"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    return {} if "{" in text else []
