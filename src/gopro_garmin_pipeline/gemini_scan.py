"""Gemini sparse scan — identify visually interesting clips across an entire ride.

Two-pass strategy:
  1. COARSE pass: 1 frame every 10s, large batches (50 frames).
     Cheap and fast — identifies interesting *regions* of the ride.
  2. CLIP pass: each hot region becomes a candidate clip (~10s window).
     Sample N evenly-spaced frames per clip and rate the clip as a
     whole on a 5-dim rubric (light, composition, motion, scenery,
     subject — each 1-5).

Per-clip rating is more consistent than per-frame: a sprint past a bus
is one event, not 5 separately-rated frames that have to be merged
back together.

Parallel API calls (4 concurrent) and system instructions (sent once,
cached by Gemini) reduce latency and token cost.

Per-clip caching means adding a new GoPro file to a folder doesn't
invalidate existing scan results. Cache key includes the prompt version
so a prompt bump invalidates everything cleanly.

Usage:
    candidates = scan_ride(video_dir, synced_clips, ride, config)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .models import (
    ModelAdapter,
    cache_dir_name,
    get_model_adapter,
    provider_api_key,
    provider_model_id,
)
from .prompt_registry import prompt_body
from .utils import MS_TO_MPH, compute_gradient, gopro_lrv_proxy


# ─── Scan parameters ─────────────────────────────────────────

# Coarse pass: find interesting regions quickly
_COARSE_INTERVAL = 10   # 1 frame every 10 seconds
_COARSE_BATCH = 50      # 500s of ride per API call
_COARSE_MIN_SCORE = 3   # threshold for coarse hit to spawn a candidate clip

# Clip pass: rate each candidate clip with multi-frame rubric
_CLIP_PAD = 5.0         # ±5s around hot timestamp → 10s candidate clip
_FRAMES_PER_CLIP = 6    # evenly-spaced sample frames per clip
# Periodic rubric coverage: coarse pass finds obvious hot regions, telemetry
# forces athletic peaks, and this samples the rest of the ride so review can
# inspect the visual-score distribution instead of only pre-filtered peaks.
# 60s→45s: denser sampling = more candidates spread across the ride, so the
# selectors have real coverage instead of clustering on a few hot regions.
# Modest extra Gemini cost (a few more fine-pass frames), still pennies/ride.
_COVERAGE_SAMPLE_INTERVAL = 45.0
_COVERAGE_EDGE_PAD = 5.0

_SCAN_WIDTH = 480
_MAX_WORKERS = 4        # concurrent Gemini API calls

# ─── Prompts ─────────────────────────────────────────────────
# System instruction: sent once, cached by Gemini across calls.
# Per-call message: telemetry + frame count.

_PROMPT_VERSION = "v10"
_CACHE_VERSION = _PROMPT_VERSION  # cache key tracks prompt version
_SYSTEM_INSTRUCTION = prompt_body("gemini_scan", _PROMPT_VERSION)

# Coarse pass still uses the v5 frame-rating prompt for region detection
_COARSE_PROMPT_VERSION = "v5"
_COARSE_SYSTEM_INSTRUCTION = prompt_body("gemini_scan", _COARSE_PROMPT_VERSION)
_COARSE_MIN_VISUAL = 3
_COARSE_MIN_ACTION = 3

_BATCH_TEMPLATE = """Batch of {n_frames} frames ({interval}s apart).

{telemetry}"""

_CLIP_TEMPLATE = """Clip: {duration:.1f}s ({n_frames} frames sampled evenly).

{telemetry}"""


# ─── Frame extraction ────────────────────────────────────────

# Global semaphore caps total concurrent ffmpeg processes across all clips
# and all passes. On a 10-core M-series laptop, 2 HEVC decodes is the
# sweet spot — more just thrashes cache and doesn't improve wall time.
_FFMPEG_SEMAPHORE: threading.Semaphore | None = None
_MAX_FFMPEG_JOBS = 2


def _get_ffmpeg_semaphore() -> threading.Semaphore:
    global _FFMPEG_SEMAPHORE
    if _FFMPEG_SEMAPHORE is None:
        _FFMPEG_SEMAPHORE = threading.Semaphore(_MAX_FFMPEG_JOBS)
    return _FFMPEG_SEMAPHORE


def _ffmpeg_extract(
    video_path: Path, interval: float, width: int,
    output_pattern: str, start: float | None = None, duration: float | None = None,
) -> None:
    """Run a single ffmpeg frame extraction with global throttling.

    Prefers LRV proxy when available. Falls back to -skip_frame nokey
    on the full-res MP4 (decodes only keyframes, ~6x faster than full
    decode on 5.3K HEVC, same output frames).
    """
    lrv = gopro_lrv_proxy(video_path)
    source = lrv if lrv is not None else video_path

    cmd = ["ffmpeg"]

    # -skip_frame nokey: only decode I-frames, skip P/B-frame reconstruction.
    # GoPro HEVC keyframes land every ~1s, precise enough for both 10s coarse
    # and 3s fine intervals. Not needed for LRV (already fast H.264).
    if lrv is None:
        cmd.append("-skip_frame")
        cmd.append("nokey")

    if start is not None:
        cmd.extend(["-ss", str(start)])

    cmd.extend(["-i", str(source)])

    if duration is not None:
        cmd.extend(["-t", str(duration)])

    cmd.extend([
        "-vf", f"fps=1/{interval},scale={width}:-2",
        "-q:v", "3", "-threads", "1", "-v", "quiet",
        output_pattern,
    ])

    sem = _get_ffmpeg_semaphore()
    sem.acquire()
    try:
        subprocess.run(cmd, capture_output=True)
    finally:
        sem.release()


def _extract_sparse_frames(
    video_path: Path, interval: float, width: int, tmpdir: Path,
    clip_duration: float | None = None,
) -> list[tuple[float, Path]]:
    """Extract one frame every `interval` seconds, return [(video_secs, path)]."""
    _ffmpeg_extract(
        video_path, interval, width,
        str(tmpdir / "frame_%05d.jpg"),
    )

    frames = []
    for p in sorted(tmpdir.glob("frame_*.jpg")):
        num = int(p.stem.split("_")[1])
        video_secs = (num - 1) * interval
        frames.append((video_secs, p))
    return frames


def _extract_region_frames(
    video_path: Path, region: tuple[float, float],
    interval: float, width: int, tmpdir: Path, region_idx: int,
) -> list[tuple[float, Path]]:
    """Extract frames from a single hot region using keyframe seek."""
    start, end = region
    duration = end - start
    if duration <= 0:
        return []

    prefix = f"r{region_idx:03d}"
    _ffmpeg_extract(
        video_path, interval, width,
        str(tmpdir / f"{prefix}_frame_%05d.jpg"),
        start=start, duration=duration,
    )

    frames = []
    for p in sorted(tmpdir.glob(f"{prefix}_frame_*.jpg")):
        num = int(p.stem.split("_")[-1])
        video_secs = start + (num - 1) * interval
        frames.append((video_secs, p))
    return frames


def _extract_regions_parallel(
    video_path: Path, regions: list[tuple[float, float]],
    interval: float, width: int, tmpdir: Path,
) -> list[tuple[float, Path]]:
    """Extract frames from multiple hot regions.

    Submits all regions to a thread pool but the global _FFMPEG_SEMAPHORE
    ensures at most _MAX_FFMPEG_JOBS run simultaneously across the entire
    scan (all clips, all passes).
    """
    if not regions:
        return []

    all_frames: list[tuple[float, Path]] = []

    # Use a generous thread pool — the semaphore does the real throttling
    workers = min(4, len(regions))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _extract_region_frames, video_path, region,
                interval, width, tmpdir, i,
            ): i
            for i, region in enumerate(regions)
        }
        for future in as_completed(futures):
            try:
                all_frames.extend(future.result())
            except Exception as e:
                idx = futures[future]
                print(f"    Region {idx}: extraction error: {e}")

    all_frames.sort(key=lambda f: f[0])
    return all_frames


# ─── Telemetry context builder ───────────────────────────────

def _build_telemetry_block(
    batch: list[tuple[float, Path]], sc, ride, interval: float,
) -> str:
    """Build per-frame telemetry context for a batch."""
    has_power = ride.has_power
    lines = []
    speeds, powers = [], []
    for i, (video_secs, _path) in enumerate(batch):
        point = sc.ride_point_at_video_time(video_secs)
        if point and point.speed is not None:
            speed_mph = point.speed * MS_TO_MPH
            hr = point.heart_rate or 0
            speeds.append(speed_mph)
            gradient = 0.0
            prev = sc.ride_point_at_video_time(max(0, video_secs - 5))
            if (prev and prev.altitude is not None and point.altitude is not None
                    and prev.distance is not None and point.distance is not None):
                gradient = compute_gradient(
                    prev.altitude, prev.distance, point.altitude, point.distance,
                )
            if has_power:
                power = point.power or 0
                powers.append(power)
                lines.append(f"Frame {i}: {speed_mph:.0f}mph {power}W {hr}bpm {gradient:+.0f}%")
            else:
                lines.append(f"Frame {i}: {speed_mph:.0f}mph {hr}bpm {gradient:+.0f}%")
        else:
            lines.append(f"Frame {i}: no data")

    summary = ""
    if speeds:
        parts = [f"avg {sum(speeds)/len(speeds):.0f}mph"]
        if powers:
            parts.append(f"{sum(powers)/len(powers):.0f}W")
        summary = f"Baseline: {' '.join(parts)}\n"

    return summary + "\n".join(lines)


def _build_clip_telemetry(sc, ride, start: float, end: float) -> str:
    """Summarize telemetry across a clip's window for the rubric prompt."""
    has_power = ride.has_power
    speeds: list[float] = []
    powers: list[int] = []
    hrs: list[int] = []
    grads: list[float] = []

    n_samples = 8
    step = max(0.5, (end - start) / n_samples)
    t = start
    while t <= end + 1e-3:
        point = sc.ride_point_at_video_time(t)
        if point and point.speed is not None:
            speeds.append(point.speed * MS_TO_MPH)
            if has_power and point.power is not None:
                powers.append(point.power)
            if point.heart_rate is not None:
                hrs.append(point.heart_rate)
            prev = sc.ride_point_at_video_time(max(0, t - 5))
            if (prev and prev.altitude is not None and point.altitude is not None
                    and prev.distance is not None and point.distance is not None):
                grads.append(compute_gradient(
                    prev.altitude, prev.distance, point.altitude, point.distance,
                ))
        t += step

    parts: list[str] = []
    if speeds:
        parts.append(
            f"speed avg={sum(speeds)/len(speeds):.0f}mph peak={max(speeds):.0f}mph"
        )
    if powers:
        parts.append(f"power avg={sum(powers)/len(powers):.0f}W peak={max(powers)}W")
    if hrs:
        parts.append(f"HR avg={sum(hrs)/len(hrs):.0f}bpm")
    if grads:
        parts.append(f"gradient avg={sum(grads)/len(grads):+.1f}% peak={max(grads, key=abs):+.0f}%")

    return "Telemetry: " + " | ".join(parts) if parts else "Telemetry: no data"


def _build_context_strings(has_power: bool, ftp: int) -> tuple[str, str, str]:
    """Build (power_zones, telemetry_fields, telemetry_examples) strings."""
    if has_power:
        power_zones = (
            f"Athlete power zones (FTP={ftp}W):\n"
            f"  Z1 Active Recovery <{int(ftp * 0.55)}W | Z2 Endurance <{int(ftp * 0.75)}W | "
            f"Z3 Tempo <{int(ftp * 0.90)}W\n"
            f"  Z4 Threshold <{int(ftp * 1.05)}W | Z5 VO2max <{int(ftp * 1.20)}W | "
            f"Z6 Anaerobic <{int(ftp * 1.50)}W | Z7 Neuromuscular {int(ftp * 1.50)}W+\n"
            "Only Z5+ qualifies as genuinely hard effort. Z3-Z4 is moderate. Z1-Z2 is easy."
        )
        telemetry_fields = "power, HR, "
        telemetry_examples = (
            "a visually plain frame at 350W on a 10% grade\n"
            "is a hard effort worth including; the same frame at 120W on flat is not."
        )
    else:
        power_zones = ""
        telemetry_fields = "HR, "
        telemetry_examples = (
            "a visually plain frame with high HR on a 10% grade\n"
            "is a hard effort worth including; the same frame with low HR on flat is not."
        )
    return power_zones, telemetry_fields, telemetry_examples


# ─── Gemini API call (with retry) ────────────────────────────


class _ScanErrors:
    """Thread-safe tally of hard Gemini API failures during a scan.

    A scan that fails every call (billing lapse, quota exhaustion,
    project denial) must NOT be indistinguishable from a scan that
    genuinely found nothing interesting. Call sites record hard failures
    here; ``_scan_clip`` consults the count to decide whether caching the
    (empty) result is safe, and ``scan_ride`` uses the aggregate to emit
    a loud warning instead of silently degrading to telemetry-only.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.count = 0
        self.samples: list[str] = []

    def record(self, msg: str) -> None:
        with self._lock:
            self.count += 1
            if len(self.samples) < 3:
                self.samples.append(msg)

    def merge(self, other: "_ScanErrors") -> None:
        with self._lock:
            self.count += other.count
            for s in other.samples:
                if len(self.samples) < 3:
                    self.samples.append(s)


def _read_images(frame_paths: list[Path]) -> list[bytes]:
    images: list[bytes] = []
    for fp in frame_paths:
        with open(fp, "rb") as f:
            images.append(f.read())
    return images


def _coarse_prompts(
    telemetry_text: str, n_frames: int, interval: float, has_power: bool,
) -> tuple[str, str]:
    """Build system + user text for the coarse (v5) pass."""
    from .config import get_settings as _get_settings
    _s = _get_settings()
    power_zones, telemetry_fields, telemetry_examples = _build_context_strings(
        has_power, _s.ftp,
    )
    system_text = _COARSE_SYSTEM_INSTRUCTION.format(
        min_visual=_COARSE_MIN_VISUAL, min_action=_COARSE_MIN_ACTION,
        power_zones=power_zones,
        telemetry_fields=telemetry_fields,
        telemetry_examples=telemetry_examples,
    )
    user_text = _BATCH_TEMPLATE.format(
        n_frames=n_frames, interval=interval, telemetry=telemetry_text,
    )
    return system_text, user_text


def _fine_prompts(
    telemetry_text: str, duration: float, n_frames: int, has_power: bool,
) -> tuple[str, str]:
    """Build system + user text for the fine rubric pass."""
    from .config import get_settings as _get_settings
    _s = _get_settings()
    power_zones, telemetry_fields, telemetry_examples = _build_context_strings(
        has_power, _s.ftp,
    )
    system_text = _SYSTEM_INSTRUCTION.format(
        n_frames=n_frames,
        power_zones=power_zones,
        telemetry_fields=telemetry_fields,
        telemetry_examples=telemetry_examples,
    )
    user_text = _CLIP_TEMPLATE.format(
        duration=duration, n_frames=n_frames, telemetry=telemetry_text,
    )
    return system_text, user_text


def _call_batch(
    adapter: ModelAdapter,
    frame_paths: list[Path],
    telemetry_text: str,
    n_frames: int,
    interval: float,
    has_power: bool = True,
) -> list[dict]:
    """Coarse pass — score N frames individually using v5 frame-rating prompt."""
    system_text, user_text = _coarse_prompts(
        telemetry_text, n_frames, interval, has_power,
    )
    return adapter.score_frames(
        images=_read_images(frame_paths),
        system=system_text,
        user=user_text,
    )


def _call_clip_rubric(
    adapter: ModelAdapter,
    frame_paths: list[Path],
    telemetry_text: str,
    duration: float,
    has_power: bool = True,
) -> dict | None:
    """Clip pass — rate one clip on the 5-dim rubric."""
    system_text, user_text = _fine_prompts(
        telemetry_text, duration, len(frame_paths), has_power,
    )
    return adapter.score_clip_rubric(
        images=_read_images(frame_paths),
        system=system_text,
        user=user_text,
    )


def _call_with_retry(
    adapter: ModelAdapter, frame_paths, telemetry_text, n_frames, interval,
    label: str = "", has_power: bool = True, errors: _ScanErrors | None = None,
) -> list[dict] | None:
    """Call the vision adapter with exponential backoff on transient errors.

    Returns the parsed result on success (including a legitimately empty
    list). Returns None only when the call HARD-FAILS — a non-retryable
    error or exhausted retries — and records that failure in ``errors``
    so the caller can tell "API down" apart from "found nothing" and
    avoid caching an empty result.
    """
    last_err = ""
    for attempt in range(5):
        try:
            return _call_batch(
                adapter, frame_paths, telemetry_text, n_frames, interval,
                has_power=has_power,
            )
        except Exception as e:
            last_err = str(e)
            if adapter.is_transient_error(e):
                wait = 2 ** attempt * 5
                print(f"    {label}: {last_err[:80]}... retrying in {wait}s")
                time.sleep(wait)
            else:
                print(f"    {label}: failed: {e}")
                if errors is not None:
                    errors.record(f"{label}: {last_err[:160]}")
                return None
    if errors is not None:
        errors.record(f"{label}: exhausted retries — {last_err[:160]}")
    return None


# ─── Per-clip caching ────────────────────────────────────────

def _label_fingerprint(forced_fine_times: list[float] | None) -> str:
    """Short stable digest of label-forced timestamps for cache keying.

    Empty / None ↔ no fingerprint (so unlabeled scans share a cache key).
    Timestamps are rounded to whole seconds before hashing, so a 0.4s
    label drift does not invalidate the cache. Adding or removing a
    label changes the digest and forces a re-scan.
    """
    if not forced_fine_times:
        return ""
    sig = ",".join(f"{t:.0f}" for t in sorted(forced_fine_times))
    return hashlib.sha1(sig.encode()).hexdigest()[:8]


def _model_fingerprint(model_id: str) -> str:
    return hashlib.sha1(model_id.encode()).hexdigest()[:8]


def _clip_cache_key(
    clip_name: str,
    forced_fine_times: list[float] | None = None,
    *,
    provider: str = "gemini",
    model_id: str = "",
) -> str:
    """Stable filename-safe cache key for a clip + scan version + labels.

    Gemini keeps legacy filenames (no model fingerprint). Other providers
    include ``_m{model_fp}`` so changing OPENAI_MODEL cannot reuse ratings.
    """
    stem = Path(clip_name).stem
    parts = [stem, _CACHE_VERSION]
    if provider != "gemini" and model_id:
        parts.append(f"m{_model_fingerprint(model_id)}")
    fp = _label_fingerprint(forced_fine_times)
    if fp:
        parts.append(f"l{fp}")
    return "_".join(parts) + ".json"


def _load_clip_cache(
    video_dir: Path, clip_name: str,
    forced_fine_times: list[float] | None = None,
    *,
    provider: str = "gemini",
    model_id: str = "",
) -> list[dict] | None:
    """Load cached hits for a single clip + label set + provider/model."""
    cache_dir = video_dir / cache_dir_name(provider)
    cache_file = cache_dir / _clip_cache_key(
        clip_name, forced_fine_times, provider=provider, model_id=model_id,
    )
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text())
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _save_clip_cache(
    video_dir: Path, clip_name: str, hits: list[dict],
    forced_fine_times: list[float] | None = None,
    *,
    provider: str = "gemini",
    model_id: str = "",
) -> None:
    """Save raw hits for a single clip + label set + provider/model."""
    cache_dir = video_dir / cache_dir_name(provider)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / _clip_cache_key(
        clip_name, forced_fine_times, provider=provider, model_id=model_id,
    )
    cache_file.write_text(json.dumps(hits, indent=2))


# ─── Two-pass scan per clip ──────────────────────────────────

def _scan_clip(
    sc, ride, settings, video_dir: Path,
    forced_fine_times: list[float] | None = None,
    agg_errors: _ScanErrors | None = None,
) -> list[dict]:
    """Two-pass scan of a single clip. Returns clip-rating dicts.

    Pass 1 (coarse): 1 frame/10s, batched. Finds hot timestamps using
        the v5 frame-rating prompt.
    Pass 2 (clip rubric): each hot region becomes a candidate clip
        (~10s window). Sample N frames evenly, rate the clip as a
        whole on the 5-dim rubric.

    forced_fine_times: video_secs positions that must be rated regardless
        of coarse scores. Used to ensure the model sees every labeled moment.
    """
    clip = sc.clip
    clip_name = clip.path.name

    # Per-clip adapter so nested thread pools do not share one client.
    adapter = get_model_adapter(settings)
    provider = adapter.name
    model_id = adapter.model_id

    # Cache key folds in the label fingerprint so adding or editing
    # manual labels invalidates a cached scan and the model gets to score
    # the newly forced regions on the next run.
    cached = _load_clip_cache(
        video_dir, clip_name, forced_fine_times,
        provider=provider, model_id=model_id,
    )
    if cached is not None:
        print(f"  {clip_name}: {len(cached)} clips (cached)")
        return cached

    # Per-clip failure tally — gates caching so a transient API outage
    # is never persisted as "scanned, found nothing."
    errors = _ScanErrors()

    with tempfile.TemporaryDirectory(prefix="vlm_coarse_") as tmpdir:
        # ── Pass 1: coarse scan ───────────────────────────────
        print(f"  {clip_name}: coarse pass ({clip.duration_secs:.0f}s, 1 frame/{_COARSE_INTERVAL}s)...")
        coarse_frames = _extract_sparse_frames(
            clip.path, _COARSE_INTERVAL, _SCAN_WIDTH, Path(tmpdir),
            clip_duration=clip.duration_secs,
        )
        if not coarse_frames:
            _save_clip_cache(
                video_dir, clip_name, [], forced_fine_times,
                provider=provider, model_id=model_id,
            )
            return []

        coarse_hits = _run_batches_parallel(
            adapter, coarse_frames, sc, ride, settings,
            batch_size=_COARSE_BATCH, interval=_COARSE_INTERVAL,
            label_prefix=f"{clip_name} coarse",
            errors=errors,
        )

        # Identify hot timestamps (frames that scored above the coarse threshold)
        hot_times: list[float] = []
        for h in coarse_hits:
            visual = h.get("visual", 0)
            action = h.get("action", 0)
            if visual >= _COARSE_MIN_SCORE or action >= _COARSE_MIN_SCORE:
                hot_times.append(h["video_secs"])

        # Inject forced times from labels, telemetry peaks, and coverage
        # samples. Model scores blind (no label/source text), but every
        # forced moment gets a fair rubric rating.
        if forced_fine_times:
            n_added = 0
            for t in forced_fine_times:
                if not any(abs(t - ht) < _CLIP_PAD for ht in hot_times):
                    hot_times.append(t)
                    n_added += 1
            if n_added > 0:
                print(f"  {clip_name}: {n_added} forced clips added")

        if not hot_times:
            if errors.count:
                print(f"  ⚠ {clip_name}: coarse pass had {errors.count} "
                      f"failure(s) — NOT caching empty result (will retry next run)")
                if agg_errors is not None:
                    agg_errors.merge(errors)
                return []
            print(f"  {clip_name}: no interesting regions found in coarse pass")
            _save_clip_cache(
                video_dir, clip_name, [], forced_fine_times,
                provider=provider, model_id=model_id,
            )
            return []

        # Build candidate clips (±_CLIP_PAD around each hot timestamp) and
        # merge overlapping windows.
        candidate_clips = _merge_regions(
            hot_times, _CLIP_PAD, clip.duration_secs,
        )
        coverage = sum(end - start for start, end in candidate_clips)
        print(f"  {clip_name}: {len(candidate_clips)} candidate clips "
              f"({coverage:.0f}s / {clip.duration_secs:.0f}s = "
              f"{coverage / clip.duration_secs * 100:.0f}%)")

    # ── Pass 2: clip rubric rating ────────────────────────────
    clip_results = _rate_candidate_clips(
        adapter, clip.path, clip_name, candidate_clips,
        sc, ride, settings, errors=errors,
    )
    print(f"  {clip_name}: {len(clip_results)} clips rated by rubric")

    if errors.count:
        print(f"  ⚠ {clip_name}: {errors.count} failure(s) during scan — "
              f"NOT caching {len(clip_results)} partial result(s) (will retry next run)")
        if agg_errors is not None:
            agg_errors.merge(errors)
        return clip_results

    _save_clip_cache(
        video_dir, clip_name, clip_results, forced_fine_times,
        provider=provider, model_id=model_id,
    )
    return clip_results


def _rate_candidate_clips(
    adapter: ModelAdapter, video_path: Path, clip_name: str,
    candidate_clips: list[tuple[float, float]],
    sc, ride, settings, errors: _ScanErrors | None = None,
) -> list[dict]:
    """Rate each candidate clip with the 5-dim rubric. Returns list of dicts."""
    if not candidate_clips:
        return []

    has_power = ride.has_power
    n_total = len(candidate_clips)

    def process_one(idx: int, region: tuple[float, float]) -> dict | None:
        start, end = region
        duration = end - start
        if duration <= 0:
            return None
        with tempfile.TemporaryDirectory(prefix="vlm_clip_") as tmpdir:
            # Sample N evenly-spaced frames from this clip
            interval = duration / max(1, _FRAMES_PER_CLIP)
            samples = _extract_region_frames(
                video_path, region, interval, _SCAN_WIDTH, Path(tmpdir),
                region_idx=idx,
            )
            if not samples:
                return None
            # Cap at _FRAMES_PER_CLIP in case ffmpeg returns one extra
            samples = samples[:_FRAMES_PER_CLIP]
            telemetry = _build_clip_telemetry(sc, ride, start, end)
            label = f"{clip_name} clip {idx + 1}/{n_total}"
            result = _call_clip_with_retry(
                adapter, [p for _, p in samples], telemetry,
                duration, label=label, has_power=has_power,
                errors=errors,
            )
            if not result:
                return None

            # Parse rubric. Low-quality clips are still useful calibration
            # data, so keep them and let ranking/review decide their fate.
            rubric_keys = ("light", "composition", "motion", "scenery", "subject")
            rubric = {k: int(result.get(k, 0)) for k in rubric_keys}
            raw_sum = sum(rubric.values())

            # Model-picked best 3-second window inside the clip.
            # peak_offset 0.0-1.0 maps to start..end. Default to 0.5
            # (midpoint) if the model didn't return one. Clamp to [0,1].
            peak_offset = float(result.get("peak_offset", 0.5))
            peak_offset = max(0.0, min(1.0, peak_offset))
            anchor = start + peak_offset * (end - start)

            return {
                "clip_name": clip_name,
                "video_start": start,
                "video_end": end,
                "anchor_video_secs": anchor,
                "peak_offset": peak_offset,
                "rubric": rubric,
                "raw_sum": raw_sum,
                "clip_type": result.get("clip_type", ""),
                "crop_x": int(result.get("crop_x", 50)),
                "reason": result.get("reason", ""),
            }

    print(f"  {clip_name}: rating {n_total} clips on rubric...")
    results: list[dict] = []
    workers = min(_MAX_WORKERS, n_total)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_one, i, region): i
            for i, region in enumerate(candidate_clips)
        }
        for future in as_completed(futures):
            try:
                r = future.result()
                if r is not None:
                    results.append(r)
            except Exception as e:
                idx = futures[future]
                print(f"    {clip_name} clip {idx + 1}/{n_total}: error: {e}")

    results.sort(key=lambda r: r["video_start"])
    return results


def _call_clip_with_retry(
    adapter: ModelAdapter, frame_paths, telemetry_text, duration,
    label: str = "", has_power: bool = True, errors: _ScanErrors | None = None,
) -> dict | None:
    """Call clip rubric with exponential backoff on transient errors.

    A returned None means the model gave no usable rubric (a normal,
    cacheable outcome). A recorded entry in ``errors`` means the API
    itself hard-failed — the signal the caller uses to avoid caching an
    empty result as if the clip were genuinely uninteresting.
    """
    last_err = ""
    for attempt in range(5):
        try:
            return _call_clip_rubric(
                adapter, frame_paths, telemetry_text, duration,
                has_power=has_power,
            )
        except Exception as e:
            last_err = str(e)
            if adapter.is_transient_error(e):
                wait = 2 ** attempt * 5
                print(f"    {label}: {last_err[:80]}... retrying in {wait}s")
                time.sleep(wait)
            else:
                print(f"    {label}: failed: {e}")
                if errors is not None:
                    errors.record(f"{label}: {last_err[:160]}")
                return None
    if errors is not None:
        errors.record(f"{label}: exhausted retries — {last_err[:160]}")
    return None


def _merge_regions(
    times: list[float], pad: float, max_duration: float,
) -> list[tuple[float, float]]:
    """Merge nearby timestamps into (start, end) regions."""
    if not times:
        return []
    times = sorted(times)
    regions: list[tuple[float, float]] = []
    start = max(0, times[0] - pad)
    end = min(times[0] + pad, max_duration)

    for t in times[1:]:
        t_start = max(0, t - pad)
        t_end = min(t + pad, max_duration)
        if t_start <= end:
            end = t_end
        else:
            regions.append((start, end))
            start, end = t_start, t_end
    regions.append((start, end))
    return regions




# ─── Parallel batch runner ───────────────────────────────────

def _run_batches_parallel(
    adapter: ModelAdapter,
    frames: list[tuple[float, Path]],
    sc, ride, settings,
    batch_size: int,
    interval: float,
    label_prefix: str,
    errors: _ScanErrors | None = None,
) -> list[dict]:
    """Run batches in parallel with ThreadPoolExecutor. Returns raw hit dicts."""
    batches = []
    for i in range(0, len(frames), batch_size):
        batches.append(frames[i:i + batch_size])

    n_batches = len(batches)
    if n_batches == 0:
        return []

    has_power = ride.has_power

    def process_batch(batch_num: int, batch: list[tuple[float, Path]]) -> list[dict]:
        batch_paths = [p for _, p in batch]
        telemetry = _build_telemetry_block(batch, sc, ride, interval)
        label = f"{label_prefix} {batch_num + 1}/{n_batches}"

        results = _call_with_retry(
            adapter, batch_paths, telemetry, len(batch), interval,
            label=label, has_power=has_power, errors=errors,
        )
        if results is None:
            return []

        hits = []
        for r in results:
            idx = r.get("frame_index", 0)
            visual = r.get("visual", 0)
            action = r.get("action", 0)
            reason = r.get("reason", "")
            clip_type = r.get("clip_type", "")

            if idx >= len(batch):
                continue
            if visual < _COARSE_MIN_VISUAL and action < _COARSE_MIN_ACTION:
                continue

            hits.append({
                "clip_name": sc.clip.path.name,
                "video_secs": batch[idx][0],
                "visual": visual,
                "action": action,
                "clip_type": clip_type,
                "crop_x": r.get("crop_x", 50),
                "reason": reason,
            })

        print(f"    {label}: {len(hits)} hits")
        return hits

    all_hits: list[dict] = []
    workers = min(_MAX_WORKERS, n_batches)

    if workers <= 1:
        # Sequential for single batch
        for i, batch in enumerate(batches):
            all_hits.extend(process_batch(i, batch))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(process_batch, i, batch): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                try:
                    all_hits.extend(future.result())
                except Exception as e:
                    batch_num = futures[future]
                    print(f"    {label_prefix} {batch_num + 1}/{n_batches}: error: {e}")

    return all_hits


# ─── Clip result → Segment conversion ───────────────────────

def _hits_to_segments(
    hits: list[dict], synced_clips: list, ride, config,
    *,
    provider: str = "gemini",
) -> list:
    """Convert clip rubric results to Segment candidates.

    New format (v6): each hit is one rated clip with rubric dict and
    video_start/video_end already set. score = raw_sum (5-25) → 0-10
    via /2.5 scaling.
    """
    import datetime as dt
    from .composer import Segment
    from .sync import normalize_tz

    clip_by_name = {sc.clip.path.name: sc for sc in synced_clips}
    candidates = []
    vision_type = f"{provider}_vision"

    for h in hits:
        clip_name = h["clip_name"]
        sc = clip_by_name.get(clip_name)
        if sc is None:
            continue

        video_start = h.get("video_start", 0.0)
        video_end = h.get("video_end", 0.0)
        anchor = h.get("anchor_video_secs", (video_start + video_end) / 2)
        rubric = h.get("rubric", {})
        raw_sum = h.get("raw_sum", sum(rubric.values()) if rubric else 0)
        reason = h.get("reason", "")
        clip_type = h.get("clip_type", "")

        # Anchor → wall time → ride seconds
        wall = sc.clip.creation_time + dt.timedelta(seconds=anchor)
        if ride.start_time:
            wall = normalize_tz(wall, ride.start_time)
            ride_secs = (wall - ride.start_time).total_seconds() + sc.offset_secs
        else:
            ride_secs = anchor

        # User-facing 1-10 score = raw_sum / 5 (range 5-50 → 1-10)
        score = raw_sum / 5.0

        # Portrait crop bias — same logic as before
        raw_crop_x = h.get("crop_x", 50)
        raw_bias = (raw_crop_x - 50) / 50.0
        _CENTER_TYPES = {"scenery", "landmark", "transition", "urban"}
        if clip_type in _CENTER_TYPES:
            crop_bias = 0.0
        else:
            # Use motion confidence (1-10 scale) to scale the crop shift
            motion = rubric.get("motion", 5)
            motion_confidence = max(0, (motion - 4)) / 6.0  # 0.0-1.0
            shift = abs(raw_bias)
            dampening = 1.0 if shift <= 0.3 else 0.5 + 0.5 * motion_confidence
            crop_bias = raw_bias * dampening

        candidates.append(Segment(
            clip_name=clip_name,
            video_start=video_start,
            video_end=video_end,
            ride_time_secs=ride_secs,
            anchor_video_secs=anchor,
            score=score,
            source=provider,
            rubric=rubric,
            portrait_crop_bias=crop_bias,
            label={
                "ride_time_secs": ride_secs,
                "type": vision_type,
                "notes": reason,
                "clip_name": clip_name,
                "video_secs": anchor,
                "rubric": rubric,
                "raw_sum": raw_sum,
                "clip_type": clip_type,
                "crop_x": raw_crop_x,
            },
        ))

    return candidates


def _inject_telemetry_peaks(
    forced_by_clip: dict[str, list[float]],
    ride,
    synced_clips: list,
) -> int:
    """Append telemetry highlight peak_times to forced_by_clip in place.

    Mirrors the geometry of composer._ride_time_to_video. For each
    detected highlight, find the chapter that contains its peak_time
    in real-world wall clock (after applying the synced offset) and
    push the corresponding video_secs into the chapter's forced list.

    Returns the number of peaks added.
    """
    from .highlights import detect_highlights, HighlightConfig
    from .sync import normalize_tz
    import datetime as dt

    if ride.start_time is None:
        return 0
    try:
        highlights = detect_highlights(
            ride, HighlightConfig(padding_before_secs=0, padding_after_secs=0),
        )
    except Exception:
        return 0

    n_added = 0
    for hl in highlights:
        peak_time = hl.peak_time or hl.start_time
        if peak_time is None:
            continue
        for sc in synced_clips:
            clip = sc.clip
            clip_start = normalize_tz(clip.creation_time, peak_time)
            clip_end = clip_start + dt.timedelta(seconds=clip.duration_secs)
            # peak_time is FIT-timeline; subtract sc.offset_secs to get
            # the GoPro-timeline wall time, then check chapter bounds.
            adjusted = peak_time - dt.timedelta(seconds=sc.offset_secs)
            if clip_start <= adjusted <= clip_end:
                video_secs = (adjusted - clip_start).total_seconds()
                forced_by_clip.setdefault(clip.path.name, []).append(video_secs)
                n_added += 1
                break
    return n_added


def _inject_coverage_samples(
    forced_by_clip: dict[str, list[float]],
    synced_clips: list,
    interval_secs: float = _COVERAGE_SAMPLE_INTERVAL,
) -> int:
    """Append periodic timestamps so Gemini rates the ride distribution.

    The coarse pass is intentionally sparse and thresholded. Sampling one
    rubric window per minute gives the reviewer a calibrated spread of visual
    scores, including ordinary sections, without sending every second of
    footage to Gemini.
    """
    n_added = 0
    for sc in synced_clips:
        duration = getattr(sc.clip, "duration_secs", 0.0) or 0.0
        if duration <= 0:
            continue
        lo = min(_COVERAGE_EDGE_PAD, duration / 2)
        hi = max(lo, duration - _COVERAGE_EDGE_PAD)
        existing = forced_by_clip.setdefault(sc.clip.path.name, [])
        t = lo
        while t <= hi + 1e-3:
            if not any(abs(t - other) < _CLIP_PAD for other in existing):
                existing.append(t)
                n_added += 1
            t += interval_secs
    return n_added


# ─── Main entry point ────────────────────────────────────────

def scan_ride(
    video_dir: Path,
    synced_clips: list,
    ride,
    config,
    labels: list[dict] | None = None,
) -> list:
    """Scan entire ride with the configured vision model, return Segment candidates.

    Two-pass strategy:
      1. Coarse: 1 frame/10s → find interesting regions
      2. Fine: rate hot regions, telemetry/label peaks, and coverage samples

    Per-clip caching means only new/changed clips get scanned.

    If labels are provided, their timestamps are injected as forced
    fine-pass regions so the model always scores labeled moments (blind —
    no label text in the prompt).
    """
    from .config import get_settings

    video_dir = Path(video_dir)

    settings = get_settings()
    provider = (settings.model_provider or "gemini").lower()
    if not provider_api_key(settings):
        print(f"  Vision scan ({provider}): no API key configured, skipping")
        return []

    # Resolve model_id for cache probes without constructing a client when
    # we only need the fingerprint for cache-hit counting below.
    model_id = provider_model_id(settings)

    # Build per-clip forced fine-pass times from labels + telemetry peaks.
    # The coarse pass at 1 frame / 10s misses fast events — a corner taken
    # at speed, a brief sprint, a 3-second descent — because the sample
    # frames don't land on the action. Telemetry knows exactly when the
    # speed/power/HR/climb peaks were; we hand those video timestamps to
    # the model directly so the rubric pass scores them no matter what the
    # coarse pass thought.
    forced_by_clip: dict[str, list[float]] = {}

    if labels:
        for lab in labels:
            clip_name = lab.get("clip_name", "")
            video_secs = lab.get("video_secs")
            if clip_name and video_secs is not None:
                forced_by_clip.setdefault(clip_name, []).append(video_secs)
        n_labels = sum(len(v) for v in forced_by_clip.values())
        if n_labels:
            print(f"  Vision scan ({provider}): {n_labels} label timestamps "
                  f"forced into fine pass")

    # Telemetry peak injection: detect highlights and map each peak_time
    # back to a (clip_name, video_secs) inside one of the synced clips.
    # Uses each SyncedClip's timezone-normalized clip-start to invert.
    n_telemetry = _inject_telemetry_peaks(forced_by_clip, ride, synced_clips)
    if n_telemetry:
        print(f"  Vision scan ({provider}): {n_telemetry} telemetry peaks "
              f"forced into fine pass")

    n_coverage = _inject_coverage_samples(forced_by_clip, synced_clips)
    if n_coverage:
        print(f"  Vision scan ({provider}): {n_coverage} coverage samples "
              f"forced into fine pass")

    # Scan clips in parallel — each clip gets its own coarse+fine pass
    # and its own adapter/client instance.
    # scan_errors aggregates hard API failures across every clip so a
    # total outage triggers a loud warning instead of silently returning
    # an empty (and therefore telemetry-only) candidate set.
    scan_errors = _ScanErrors()
    all_hits: list[dict] = []
    if len(synced_clips) <= 1:
        for sc in synced_clips:
            forced = forced_by_clip.get(sc.clip.path.name)
            hits = _scan_clip(
                sc, ride, settings, video_dir,
                forced_fine_times=forced, agg_errors=scan_errors,
            )
            all_hits.extend(hits)
    else:
        print(f"  Scanning {len(synced_clips)} clips in parallel...")
        # Clip-level threads overlap API waits with ffmpeg extraction.
        # Actual ffmpeg concurrency is capped by _FFMPEG_SEMAPHORE (default 2).
        with ThreadPoolExecutor(max_workers=len(synced_clips)) as pool:
            futures = {
                pool.submit(
                    _scan_clip, sc, ride, settings, video_dir,
                    forced_by_clip.get(sc.clip.path.name), scan_errors,
                ): sc.clip.path.name
                for sc in synced_clips
            }
            for future in as_completed(futures):
                clip_name = futures[future]
                try:
                    all_hits.extend(future.result())
                except Exception as e:
                    print(f"  {clip_name}: scan failed: {e}")

    total_cached = sum(
        1 for sc in synced_clips
        if _load_clip_cache(
            video_dir, sc.clip.path.name,
            forced_by_clip.get(sc.clip.path.name),
            provider=provider, model_id=model_id,
        ) is not None
    )
    print(f"  Vision scan ({provider}): {len(all_hits)} total hits across "
          f"{len(synced_clips)} clips ({total_cached} from cache)")

    if scan_errors.count:
        print(
            "\n  " + "!" * 64 + "\n"
            f"  ⚠️  VISION SCAN DEGRADED ({provider}) — "
            f"{scan_errors.count} API call(s) hard-failed.\n"
            "      Visual candidates are INCOMPLETE; affected clips fall back to\n"
            "      telemetry/filler only. Failed clips were NOT cached, so a\n"
            "      re-run will retry them once the API is healthy.\n"
            f"      First failure: {scan_errors.samples[0] if scan_errors.samples else 'n/a'}\n"
            "  " + "!" * 64 + "\n"
        )

    candidates = _hits_to_segments(
        all_hits, synced_clips, ride, config, provider=provider,
    )
    print(f"  Vision scan ({provider}): {len(candidates)} interesting moments")
    return candidates
