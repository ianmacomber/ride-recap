"""Extract downsampled frames from all GoPro clips, named by ride-elapsed second.

Usage:
    gopro-garmin extract-frames <fit_file> <video_dir> [--output-dir frames/] [--interval 5]

Output:
    frames/frame_00272.jpg   (ride second 272)
    frames/frame_00277.jpg   (ride second 277)
    ...
    frames/manifest.json     (metadata about the extraction)
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .fit_parser import parse_fit
from .gopro_meta import extract_all


def _extract_single_frame(
    video_path: str, seek_secs: float, output_path: str, width: int
) -> bool:
    """Extract one frame from a video using input seeking (fast, no full decode)."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(seek_secs),   # seek BEFORE -i = input seeking (keyframe)
                "-i", video_path,
                "-frames:v", "1",
                "-vf", f"scale={width}:-1",
                "-q:v", "5",
                output_path,
            ],
            capture_output=True,
            timeout=10,
        )
        return Path(output_path).exists()
    except Exception:
        return False


def _extract_clip_frames(
    clip_path: str,
    clip_name: str,
    clip_duration: float,
    clip_offset_into_ride: float,
    ride_duration: float,
    output_dir: Path,
    interval: int,
    width: int,
) -> list[dict]:
    """Extract all frames for one clip. Returns list of frame metadata dicts."""
    frames = []
    # Generate seek points
    for video_secs in range(0, int(clip_duration), interval):
        ride_secs = int(clip_offset_into_ride + video_secs)
        if ride_secs < 0 or ride_secs > ride_duration:
            continue

        dest = output_dir / f"frame_{ride_secs:06d}.jpg"
        if _extract_single_frame(clip_path, video_secs, str(dest), width):
            frames.append({
                "ride_secs": ride_secs,
                "file": dest.name,
                "clip": clip_name,
                "video_secs": video_secs,
            })

    return frames


def extract_ride_frames(
    fit_path: str | Path,
    video_dir: str | Path,
    output_dir: str | Path,
    interval: int = 1,
    width: int = 480,
    offset: float = 0.0,
    workers: int = 8,
) -> Path:
    """Extract one frame every `interval` seconds from each clip, saved as ride-time-indexed JPEGs.

    Uses input-seeking (fast keyframe seek, no full decode) and parallel extraction.
    """
    fit_path = Path(fit_path)
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing FIT file {fit_path.name}...")
    ride = parse_fit(fit_path)
    ride_start = ride.start_time
    ride_duration = ride.duration.total_seconds() if ride.duration else 0

    print(f"Scanning videos in {video_dir}...")
    clips = extract_all(video_dir)
    print(f"  {len(clips)} clips, ride duration {ride_duration:.0f}s ({ride_duration/60:.0f} min)")

    manifest = {
        "fit_file": str(fit_path),
        "video_dir": str(video_dir),
        "interval": interval,
        "width": width,
        "offset": offset,
        "ride_duration_secs": ride_duration,
        "ride_start": ride_start.isoformat() if ride_start else None,
        "clips": [],
        "frames": [],
    }

    # Build work items
    work = []
    for clip in clips:
        from .sync import normalize_tz
        clip_start_wall = normalize_tz(clip.creation_time, ride_start)

        clip_offset_into_ride = (clip_start_wall - ride_start).total_seconds() + offset
        if clip_offset_into_ride + clip.duration_secs < 0 or clip_offset_into_ride > ride_duration:
            continue

        manifest["clips"].append({
            "name": clip.path.name,
            "ride_start_secs": round(max(0, clip_offset_into_ride), 1),
            "ride_end_secs": round(min(ride_duration, clip_offset_into_ride + clip.duration_secs), 1),
            "duration_secs": round(clip.duration_secs, 1),
        })

        n_frames = len(range(0, int(clip.duration_secs), interval))
        print(f"  {clip.path.name}: {clip.duration_secs:.0f}s, ~{n_frames} frames")

        work.append((
            str(clip.path), clip.path.name, clip.duration_secs,
            clip_offset_into_ride, ride_duration,
        ))

    total_expected = sum(
        len(range(0, int(dur), interval))
        for _, _, dur, _, _ in work
    )
    print(f"\nExtracting ~{total_expected} frames across {len(work)} clips (parallel)...")

    # Extract in parallel — one thread per clip
    all_frames = []
    with ThreadPoolExecutor(max_workers=min(workers, len(work))) as pool:
        futures = {
            pool.submit(
                _extract_clip_frames,
                clip_path, clip_name, clip_dur, clip_offset, ride_dur,
                output_dir, interval, width,
            ): clip_name
            for clip_path, clip_name, clip_dur, clip_offset, ride_dur in work
        }
        for future in as_completed(futures):
            clip_name = futures[future]
            try:
                frames = future.result()
                all_frames.extend(frames)
                print(f"  {clip_name}: {len(frames)} frames")
            except Exception as e:
                print(f"  {clip_name}: ERROR {e}")

    all_frames.sort(key=lambda f: f["ride_secs"])
    manifest["frames"] = all_frames

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nDone: {len(all_frames)} frames saved to {output_dir}/")
    return output_dir
