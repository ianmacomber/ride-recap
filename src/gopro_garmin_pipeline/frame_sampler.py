"""Extract and sample frames from video for VLM analysis.

Mirrors the motion-aware sampling strategy from slalom_tokyo_drift:
half the frames evenly spaced (temporal context), half from motion
spikes (interesting moments). Frames ordered low-motion-first to
exploit VLM recency bias.
"""

from __future__ import annotations

import base64
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def extract_frames(
    video_path: str | Path,
    fps: float = 2.0,
    max_width: int = 640,
) -> list[Path]:
    """Extract frames from a video at a given FPS using ffmpeg.

    Args:
        video_path: Path to video file.
        fps: Frames per second to extract.
        max_width: Downscale to this width (preserving aspect ratio).

    Returns:
        List of paths to extracted PNG frames, sorted by time.
    """
    video_path = Path(video_path)
    tmpdir = tempfile.mkdtemp(prefix="gopro_frames_")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps},scale={max_width}:-2",
        "-q:v", "2",
        "-v", "quiet",
        f"{tmpdir}/frame_%05d.png",
    ]
    subprocess.run(cmd, check=True)

    frames = sorted(Path(tmpdir).glob("frame_*.png"))
    return frames


def compute_motion_scores(frames: list[Path]) -> list[float]:
    """Compute motion score for each frame as mean absolute pixel diff.

    Returns a list of scores (0.0 for the first frame). Higher values
    indicate more visual change between consecutive frames.
    """
    scores = [0.0]
    prev = np.array(Image.open(frames[0]).convert("L"), dtype=np.float32)

    for frame_path in frames[1:]:
        curr = np.array(Image.open(frame_path).convert("L"), dtype=np.float32)
        diff = np.mean(np.abs(curr - prev))
        scores.append(float(diff))
        prev = curr

    return scores


def select_frames(
    frames: list[Path],
    max_frames: int = 16,
    motion_ratio: float = 0.5,
) -> list[Path]:
    """Select a subset of frames using motion-aware sampling.

    Strategy (from slalom_tokyo_drift):
    - Half the frames evenly spaced for temporal context
    - Half from highest-motion frames for detail
    - Ordered low-motion-first to exploit VLM recency bias
      (high-motion frames at end get more attention)

    Args:
        frames: All extracted frame paths.
        max_frames: Maximum frames to return.
        motion_ratio: Fraction of frames selected by motion.

    Returns:
        Selected frame paths, ordered for VLM input.
    """
    if len(frames) <= max_frames:
        return frames

    n_motion = int(max_frames * motion_ratio)
    n_even = max_frames - n_motion

    # Evenly spaced frames
    step = len(frames) / n_even
    even_indices = {int(i * step) for i in range(n_even)}

    # Highest-motion frames (excluding already selected)
    motion_scores = compute_motion_scores(frames)
    candidates = [
        (score, idx)
        for idx, score in enumerate(motion_scores)
        if idx not in even_indices
    ]
    candidates.sort(reverse=True)
    motion_indices = {idx for _, idx in candidates[:n_motion]}

    all_indices = even_indices | motion_indices

    # Order: low-motion first, high-motion last (recency bias)
    ordered = sorted(all_indices, key=lambda i: motion_scores[i])

    return [frames[i] for i in ordered]


def frames_to_base64(frames: list[Path]) -> list[str]:
    """Encode frame images as base64 strings for VLM API calls."""
    result = []
    for frame_path in frames:
        with open(frame_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            result.append(encoded)
    return result


def sample_clip_frames(
    video_path: str | Path,
    fps: float = 2.0,
    max_width: int = 640,
    max_frames: int = 16,
    motion_ratio: float = 0.5,
) -> tuple[list[Path], list[str]]:
    """End-to-end: extract, sample, and encode frames from a video clip.

    Returns:
        Tuple of (selected frame paths, base64-encoded frame strings).
    """
    all_frames = extract_frames(video_path, fps=fps, max_width=max_width)
    if not all_frames:
        return [], []

    selected = select_frames(all_frames, max_frames=max_frames, motion_ratio=motion_ratio)
    encoded = frames_to_base64(selected)
    return selected, encoded
