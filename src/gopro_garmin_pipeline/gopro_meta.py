"""Extract metadata from GoPro video files."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GoProClip:
    """Metadata for a single GoPro video file."""

    path: Path
    creation_time: dt.datetime
    duration_secs: float
    width: int
    height: int
    fps: float

    @property
    def end_time(self) -> dt.datetime:
        return self.creation_time + dt.timedelta(seconds=self.duration_secs)

    def video_time_to_wall_time(self, video_secs: float) -> dt.datetime:
        """Convert a position in the video (seconds) to wall-clock time."""
        return self.creation_time + dt.timedelta(seconds=video_secs)

    def wall_time_to_video_time(self, wall_time: dt.datetime) -> float:
        """Convert wall-clock time to position in video (seconds)."""
        return (wall_time - self.creation_time).total_seconds()


def _run_ffprobe(path: Path) -> dict:
    """Run ffprobe and return parsed JSON output."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _parse_creation_time(probe_data: dict) -> dt.datetime:
    """Extract creation time from ffprobe data.

    GoPro Hero 13 stores creation_time in the format tag. We try the format
    metadata first, then fall back to stream-level tags.
    """
    # Try format-level tags first
    tags = probe_data.get("format", {}).get("tags", {})
    raw = tags.get("creation_time")

    # Fall back to first video stream tags
    if raw is None:
        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "video":
                raw = stream.get("tags", {}).get("creation_time")
                if raw:
                    break

    if raw is None:
        raise ValueError("No creation_time found in video metadata")

    # GoPro uses ISO 8601 format: 2024-03-15T14:30:00.000000Z
    raw = raw.replace("Z", "+00:00")
    return dt.datetime.fromisoformat(raw)


def extract_metadata(path: str | Path) -> GoProClip:
    """Extract metadata from a GoPro video file.

    Args:
        path: Path to a GoPro .mp4 file.

    Returns:
        GoProClip with creation time, duration, resolution, and fps.
    """
    path = Path(path)
    probe = _run_ffprobe(path)

    creation_time = _parse_creation_time(probe)

    # Find the video stream
    video_stream = None
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise ValueError(f"No video stream found in {path}")

    duration = float(probe["format"]["duration"])
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    # Parse frame rate (e.g., "60000/1001" or "30/1")
    r_frame_rate = video_stream.get("r_frame_rate", "30/1")
    num, den = r_frame_rate.split("/")
    fps = float(num) / float(den)

    return GoProClip(
        path=path,
        creation_time=creation_time,
        duration_secs=duration,
        width=width,
        height=height,
        fps=fps,
    )


def _parse_gopro_chapter(name: str) -> tuple[str, int]:
    """Parse GoPro filename into (recording_id, chapter_number).

    GoPro Hero 13 naming: GXccRRRR.MP4
      cc   = chapter number (01, 02, ...)
      RRRR = recording ID (shared across chapters)

    Returns (recording_id, chapter) e.g. ("0033", 1) for GX010033.MP4.
    """
    stem = Path(name).stem  # e.g. "GX010033"
    if len(stem) >= 8 and stem[:2] in ("GX", "GH", "GL"):
        chapter = int(stem[2:4])
        recording_id = stem[4:]
        return recording_id, chapter
    return stem, 0


def extract_all(directory: str | Path) -> list[GoProClip]:
    """Extract metadata from all GoPro .mp4 files in a directory, sorted by creation time.

    Handles GoPro chaptering: when a recording exceeds ~4GB the camera splits it
    into chapters (GX010033, GX020033, ...) that all share the same creation_time.
    We offset each chapter's creation_time by the cumulative duration of prior chapters.
    """
    directory = Path(directory)
    clips = []
    for mp4 in sorted(directory.glob("*.MP4")):
        try:
            clips.append(extract_metadata(mp4))
        except (ValueError, subprocess.CalledProcessError) as e:
            print(f"Warning: skipping {mp4.name}: {e}")
    # Also check lowercase
    for mp4 in sorted(directory.glob("*.mp4")):
        if mp4 not in [c.path for c in clips]:
            try:
                clips.append(extract_metadata(mp4))
            except (ValueError, subprocess.CalledProcessError) as e:
                print(f"Warning: skipping {mp4.name}: {e}")

    # Fix chapter timestamps: group by recording ID, sort by chapter number,
    # and offset creation_time so chapters are sequential.
    from collections import defaultdict
    groups: dict[str, list[GoProClip]] = defaultdict(list)
    for clip in clips:
        rec_id, _chapter = _parse_gopro_chapter(clip.path.name)
        groups[rec_id].append(clip)

    for rec_id, group in groups.items():
        if len(group) <= 1:
            continue
        # Sort by chapter number
        group.sort(key=lambda c: _parse_gopro_chapter(c.path.name)[1])
        # Offset each subsequent chapter
        cumulative = 0.0
        for clip in group:
            clip.creation_time = clip.creation_time + dt.timedelta(seconds=cumulative)
            cumulative += clip.duration_secs

    clips.sort(key=lambda c: c.creation_time)
    return clips
