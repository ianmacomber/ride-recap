"""Extract video clips from GoPro footage using ffmpeg."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .gopro_meta import GoProClip
from .highlights import Highlight
from .sync import SyncedClip


@dataclass
class ExtractedClip:
    """A video clip extracted from a GoPro file."""

    path: Path
    source_clip: GoProClip
    highlight: Highlight
    start_secs: float  # position in source video
    end_secs: float


def extract_clip(
    synced: SyncedClip,
    highlight: Highlight,
    output_dir: str | Path,
    index: int = 0,
) -> ExtractedClip | None:
    """Extract a video clip corresponding to a highlight.

    Uses ffmpeg with stream copy for speed (no re-encoding).

    Args:
        synced: Synced clip with FIT/video alignment.
        highlight: The highlight to extract.
        output_dir: Directory for output clips.
        index: Clip index for filename.

    Returns:
        ExtractedClip if the highlight overlaps the video, None otherwise.
    """
    clip = synced.clip
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert highlight wall times to video times
    start_secs = clip.wall_time_to_video_time(highlight.start_time)
    end_secs = clip.wall_time_to_video_time(highlight.end_time)

    # Clamp to video bounds
    start_secs = max(0, start_secs)
    end_secs = min(clip.duration_secs, end_secs)

    if start_secs >= end_secs or start_secs >= clip.duration_secs or end_secs <= 0:
        return None

    duration = end_secs - start_secs
    reason = highlight.reason.value
    output_path = output_dir / f"clip_{index:03d}_{reason}.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_secs:.2f}",
        "-i", str(clip.path),
        "-t", f"{duration:.2f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]

    subprocess.run(cmd, capture_output=True, check=True)

    return ExtractedClip(
        path=output_path,
        source_clip=clip,
        highlight=highlight,
        start_secs=start_secs,
        end_secs=end_secs,
    )


def extract_all_clips(
    synced_clips: list[SyncedClip],
    highlights: list[Highlight],
    output_dir: str | Path,
) -> list[ExtractedClip]:
    """Extract clips for all highlights across all synced videos.

    Each highlight is matched to the synced clip whose video covers that
    time range. If a highlight spans multiple GoPro files, it's extracted
    from the one with the most overlap.
    """
    output_dir = Path(output_dir)
    extracted = []

    for i, highlight in enumerate(highlights):
        # Find the best matching clip for this highlight
        best_synced = None
        best_overlap = 0.0

        for synced in synced_clips:
            clip = synced.clip
            overlap_start = max(
                highlight.start_time, clip.creation_time
            )
            overlap_end = min(highlight.end_time, clip.end_time)
            overlap = (overlap_end - overlap_start).total_seconds()
            if overlap > best_overlap:
                best_overlap = overlap
                best_synced = synced

        if best_synced and best_overlap > 0:
            result = extract_clip(best_synced, highlight, output_dir, index=i)
            if result:
                extracted.append(result)

    return extracted
