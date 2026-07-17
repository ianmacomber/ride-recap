"""GPMF-based clock sync between GoPro footage and Garmin FIT.

GoPro's reported `creation_time` (read by ffprobe) is what the camera's
RTC believed when recording started. If the camera hasn't been Quik-
synced recently and GPS lock was slow, that RTC drifts. The GoPro's
embedded GPMF metadata stream is a much better truth source: GPS time
is sub-millisecond accurate.

This module reads the first locked GPS sample out of a chapter's GPMF
stream and returns the offset that maps GoPro `creation_time` to the
actual UTC moment the recording started. That offset is the same
quantity the manual `--offset` flag has always supplied.

Hero 13 records GPS in the GPS9 format (vs older cameras' GPS5). The
upstream `gpmf` package only knows GPS5 (and drags in pandas/geopandas
for the privilege), so this module has no GPMF dependency at all: KLV
is a simple length-prefixed tree, walked here with `struct` directly.

Example:
    offset_secs, confidence = detect_offset(Path("data/raw/2026-04-28"))
    # offset_secs ≈ -9.0 means GoPro creation_time is 9s ahead of actual.
"""

from __future__ import annotations

import datetime as dt
import json
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

# 9-element GPS9 record: 7 int32 + 2 uint16 = 32 bytes per sample
_GPS9_FMT = ">lllllllHH"
_GPS9_ROW_BYTES = 32
_GPS_EPOCH = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)


# ─── GPMF extraction ──────────────────────────────────────────────

def _find_gpmd_stream(video_path: Path) -> int | None:
    """Return the ffmpeg stream index of the GPMF (gpmd) data stream."""
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", str(video_path)],
        capture_output=True, text=True, check=True,
    ).stdout
    streams = json.loads(probe).get("streams", [])
    for s in streams:
        if s.get("codec_tag_string") == "gpmd":
            return s["index"]
    return None


def _dump_gpmf_bytes(video_path: Path, stream_index: int) -> bytes:
    """Pipe the gpmd stream's raw bytes via ffmpeg."""
    return subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-map", f"0:{stream_index}", "-codec", "copy",
         "-f", "rawvideo", "pipe:"],
        capture_output=True, check=True,
    ).stdout


def _iter_klv(buf: bytes):
    """Walk one level of a GPMF KLV buffer.

    Yields ``(fourcc, type_char, size, repeat, payload)``. Containers
    (type 0x00) carry nested KLV in their payload — recurse with another
    _iter_klv call. Payloads are 4-byte aligned on the wire.
    """
    pos = 0
    while pos + 8 <= len(buf):
        fourcc = buf[pos:pos + 4].decode("latin1")
        type_char, size, repeat = struct.unpack(">BBH", buf[pos + 4:pos + 8])
        pos += 8
        length = size * repeat
        yield fourcc, type_char, size, repeat, buf[pos:pos + length]
        pos += (length + 3) & ~3


@dataclass
class GpsSample:
    timestamp: dt.datetime  # UTC
    lat: float
    lon: float
    alt: float
    speed_mps: float
    dop: float
    fix: int


def extract_gps_track(video_path: Path) -> list[GpsSample]:
    """Decode the full GPS9 track from a Hero 13 chapter.

    Returns samples at the camera's GPS rate (~10 Hz). Finds STRM
    containers holding a GPS9 block plus its SCAL divisors and decodes
    the 32-byte rows with `struct`.
    """
    idx = _find_gpmd_stream(video_path)
    if idx is None:
        return []
    raw = _dump_gpmf_bytes(video_path, idx)

    samples: list[GpsSample] = []
    for key, typ, _, _, devc in _iter_klv(raw):
        if key != "DEVC" or typ != 0:
            continue
        for skey, styp, _, _, strm in _iter_klv(devc):
            if skey != "STRM" or styp != 0:
                continue
            gps9 = None
            n = 0
            scale: list[int] = []
            for ckey, ctyp, csize, crepeat, payload in _iter_klv(strm):
                if ckey == "GPS9":
                    gps9, n = payload, crepeat
                elif ckey == "SCAL":
                    # SCAL element width varies by stream: 'l' int32 for
                    # GPS, 's' int16 elsewhere. Decode by declared type.
                    fmt = {ord("l"): ">l", ord("s"): ">h"}.get(ctyp)
                    if fmt is not None:
                        scale = [v[0] for v in struct.iter_unpack(fmt, payload)]
            if gps9 is None or len(scale) < 8:
                continue
            for i in range(n):
                row = struct.unpack(_GPS9_FMT,
                                    gps9[i * _GPS9_ROW_BYTES:(i + 1) * _GPS9_ROW_BYTES])
                lat, lon, alt, sp2, sp3, days, secs_ms, dop, fix = row
                ts = _GPS_EPOCH + dt.timedelta(
                    days=int(days), seconds=secs_ms / scale[6],
                )
                samples.append(GpsSample(
                    timestamp=ts,
                    lat=lat / scale[0],
                    lon=lon / scale[1],
                    alt=alt / scale[2],
                    speed_mps=sp2 / scale[3],
                    dop=dop / scale[7],
                    fix=int(fix),
                ))
    return samples


def first_locked_sample(samples: list[GpsSample], min_fix: int = 2) -> GpsSample | None:
    """Earliest sample with at least a 2D fix."""
    for s in samples:
        if s.fix >= min_fix:
            return s
    return None


# ─── Offset inference ─────────────────────────────────────────────

@dataclass
class OffsetReport:
    """Result of auto-sync inference for a ride.

    offset_secs is the same quantity as the manual `--offset` flag:
    add it to GoPro `creation_time` (or wall-clock derived from it) to
    get FIT-relative wall time.

    confidence is a heuristic 0-1 score:
      - 1.0  multiple chapters agree within 0.5s
      - 0.7  single locked GPS sample available
      - 0.0  no GPS lock found
    method describes which path produced the offset.
    """
    offset_secs: float
    confidence: float
    method: str
    per_chapter: dict[str, float]    # chapter name → offset
    notes: str = ""


def detect_offset(ride_dir: Path) -> OffsetReport:
    """Estimate the GoPro→FIT clock offset for an entire ride folder.

    Each GoPro recording (separate stop/start) gets its own per-clip
    offset because:

      - When you start a recording before GPS lock, the "first locked
        GPS sample" lives several seconds INTO the video. Subtracting
        ffprobe creation_time from that sample's timestamp measures
        GPS-lock latency, NOT camera RTC drift — so that clip's offset
        is a spurious large positive (e.g. +38s on a delayed lock).
      - Later recordings in the same ride may have a totally different
        RTC drift if the camera auto-corrected its clock from GPS
        between recordings.

    Averaging these into a single global offset is what produces the
    "overlay drifts 9s ahead during the sprint" bug. We instead emit
    per-clip offsets and an outlier-rejected median as the default.
    The compose path applies the per-clip value when burning each clip.
    """
    from .gopro_meta import extract_metadata

    ride_dir = Path(ride_dir)
    chapters = sorted(ride_dir.glob("GX*.MP4"))
    if not chapters:
        return OffsetReport(0.0, 0.0, "no-chapters", {}, "no GoPro chapters found")

    per_chap: dict[str, float] = {}
    failures: list[str] = []

    for chap in chapters:
        try:
            samples = extract_gps_track(chap)
        except Exception as exc:
            failures.append(f"{chap.name}: GPMF read failed ({exc})")
            continue
        clip = extract_metadata(chap)
        ct = clip.creation_time
        if ct.tzinfo is None:
            ct = ct.replace(tzinfo=dt.timezone.utc)
        offset = _stable_drift_from_samples(samples, ct)
        if offset is None:
            failures.append(f"{chap.name}: insufficient GPS lock")
            continue
        per_chap[chap.name] = offset

    if not per_chap:
        notes = "; ".join(failures) or "no offsets computable"
        return OffsetReport(0.0, 0.0, "no-fix", {}, notes)

    # Outlier rejection: drop any clip whose offset is >5s from the median.
    # On most rides the per-clip values agree within ~0.5s; large deviation
    # almost always means GPS-lock-latency masquerading as drift (see
    # docstring), and using that clip's value as a global default produces
    # noticeable overlay drift.
    offsets = list(per_chap.values())
    med = _median(offsets)
    kept = {n: o for n, o in per_chap.items() if abs(o - med) <= 5.0}
    dropped = sorted(set(per_chap) - set(kept))

    if not kept:
        # Median is itself the only useful value
        return OffsetReport(
            offset_secs=med, confidence=0.4,
            method="median (all clips drift-flagged)",
            per_chapter=per_chap,
            notes="; ".join(failures + [f"dropped: {', '.join(dropped)}"]),
        )

    kept_vals = list(kept.values())
    default = _median(kept_vals)
    spread = max(kept_vals) - min(kept_vals) if len(kept_vals) > 1 else 0.0

    if spread <= 1.0:
        method = f"per-clip (n={len(kept)}, spread={spread:.2f}s)"
        confidence = 1.0 if len(kept) > 1 else 0.85
    else:
        method = f"per-clip median (spread={spread:.2f}s)"
        confidence = 0.7

    notes_parts = list(failures)
    if dropped:
        notes_parts.append(
            "outlier offsets dropped (>5s from median): "
            + ", ".join(f"{n}={per_chap[n]:+.2f}s" for n in dropped)
        )
    return OffsetReport(
        offset_secs=default,
        confidence=confidence,
        method=method,
        per_chapter=per_chap,
        notes="; ".join(notes_parts),
    )


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _stable_drift_from_samples(
    samples: list[GpsSample],
    creation_time: dt.datetime,
    sample_rate_hz: float = 10.0,
    min_locked: int = 30,
) -> float | None:
    """Return the camera RTC drift (UTC - camera_clock) in seconds.

    Instead of "first locked sample minus creation_time" (which conflates
    GPS-lock latency with RTC drift — e.g. GPS lock 38s into a chapter
    produces a spurious +38s drift), this:

      1. Finds the first sample with a stable lock (fix>=2 and the
         following ``min_locked`` samples also locked).
      2. For each sample in the stable window, computes
         ``drift = sample.timestamp - (creation_time + N/rate)``
         where N is the sample index. A real RTC drift is constant
         within a recording (single quartz crystal), so all values
         in the window agree to within a sample period.
      3. Returns the median, robust against any single weird sample.

    Returns ``None`` if fewer than ``min_locked`` stable samples exist.
    """
    if not samples:
        return None
    start = None
    for i, s in enumerate(samples):
        if s.fix < 2:
            continue
        if all(samples[j].fix >= 2
               for j in range(i, min(i + min_locked, len(samples)))):
            start = i
            break
    if start is None or start + min_locked > len(samples):
        return None

    drifts: list[float] = []
    for i in range(start, min(start + 200, len(samples))):
        s = samples[i]
        if s.fix < 2:
            continue
        video_secs = i / sample_rate_hz
        expected_utc = creation_time + dt.timedelta(seconds=video_secs)
        drifts.append((s.timestamp - expected_utc).total_seconds())

    return _median(drifts) if drifts else None


# ─── Persistence ──────────────────────────────────────────────────

_SYNC_FILENAME = "sync.json"


def save_sync_meta(ride_dir: Path, report: OffsetReport) -> Path:
    """Persist an OffsetReport to <ride_dir>/sync.json.

    Stored as plain JSON so any tool can read it without importing the
    pipeline. Future runs of compose-selected, review-ride, etc. pick
    this up automatically as the default offset.
    """
    ride_dir = Path(ride_dir)
    out = ride_dir / _SYNC_FILENAME
    payload = {
        "offset_secs": report.offset_secs,
        "confidence": report.confidence,
        "method": report.method,
        "per_chapter": report.per_chapter,
        "notes": report.notes,
        "detected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    out.write_text(json.dumps(payload, indent=2))
    return out


def load_sync_meta(ride_dir: Path) -> dict | None:
    """Load <ride_dir>/sync.json or None if missing."""
    p = Path(ride_dir) / _SYNC_FILENAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def resolve_offsets(
    ride_dir: Path,
    manual_offset: float = 0.0,
    auto: bool = True,
) -> tuple[dict[str, float], float, str]:
    """Resolve per-clip and default FIT-vs-GoPro offsets.

    Returns ``(per_clip, default, source)``. ``per_clip`` maps GoPro
    filename → its own offset; ``default`` is the value to use for any
    clip not in the map (and for legacy callers that want one number).

    Priority: manual override → sync.json cache → live GPMF detection
    → 0.0 with a note. Multi-recording rides whose chapters drift
    differently now apply each clip's own offset instead of a global
    average, which fixed the "overlay reads N seconds ahead during
    sprints" class of bug.
    """
    if manual_offset:
        return {}, manual_offset, f"manual --offset {manual_offset:+.2f}s"

    cached = load_sync_meta(ride_dir)
    if cached is not None and "offset_secs" in cached:
        per_clip = {str(k): float(v) for k, v in (cached.get("per_chapter") or {}).items()}
        # Drop cached outliers (>5s from median) so they don't get used
        # per-clip. The default is already the median of kept values.
        if per_clip:
            med = _median(list(per_clip.values()))
            per_clip = {n: o for n, o in per_clip.items() if abs(o - med) <= 5.0}
        return per_clip, float(cached["offset_secs"]), (
            f"sync.json cache "
            f"({cached.get('method', '?')}, "
            f"conf={cached.get('confidence', 0.0):.2f})"
        )

    if not auto:
        return {}, 0.0, "auto-sync disabled"

    report = detect_offset(ride_dir)
    if report.confidence > 0:
        save_sync_meta(ride_dir, report)
        per_clip = dict(report.per_chapter)
        if per_clip:
            med = _median(list(per_clip.values()))
            per_clip = {n: o for n, o in per_clip.items() if abs(o - med) <= 5.0}
        return per_clip, report.offset_secs, (
            f"auto-detected ({report.method}, "
            f"conf={report.confidence:.2f})"
        )

    return {}, 0.0, f"auto-sync failed: {report.notes}"


def resolve_offset(
    ride_dir: Path,
    manual_offset: float = 0.0,
    auto: bool = True,
) -> tuple[float, str]:
    """Resolve the FIT-vs-GoPro offset for a ride.

    Priority:
      1. manual_offset, if non-zero — explicit user override always wins.
      2. <ride_dir>/sync.json, if present — already-detected from a
         prior run, no need to re-extract GPMF.
      3. live GPMF detection, if auto=True.
      4. 0.0, with a note explaining why.

    Returns (offset_secs, source) where source is a short description
    suitable for logging. Same resolution as :func:`resolve_offsets`,
    minus the per-clip map.
    """
    _, default, source = resolve_offsets(ride_dir, manual_offset, auto)
    return default, source
