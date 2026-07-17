"""CLI entry point for the GoPro + Garmin pipeline."""

from __future__ import annotations

import re
import shutil
from datetime import date, timedelta
from pathlib import Path

import click

from .fit_parser import parse_fit
from .gopro_meta import extract_all
from .highlights import HighlightConfig, detect_highlights


_CREW_CACHE_PATH = Path.home() / ".ride_recap_cache" / "last_crew.txt"


def _find_fits(folder: Path) -> list[Path]:
    """All FIT files in a ride folder, sorted so the first pick is deterministic."""
    return sorted(list(folder.glob("*.fit")) + list(folder.glob("*.FIT")))


def _find_mp4s(folder: Path) -> list[Path]:
    """All GoPro .MP4 chapters in a ride folder, either filename case."""
    return list(folder.glob("*.MP4")) + list(folder.glob("*.mp4"))


def _read_last_crew() -> str:
    """Return the crew label used on the previous ride, or 'SOLO' if none."""
    try:
        return _CREW_CACHE_PATH.read_text().strip() or "SOLO"
    except OSError:
        return "SOLO"


def _write_last_crew(crew: str) -> None:
    try:
        _CREW_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CREW_CACHE_PATH.write_text(crew + "\n")
    except OSError:
        pass


def resolve_crew(explicit: str | None) -> str:
    """Pick the crew label for this ride.

    Resolution order: explicit CLI value → interactive prompt (TTY only,
    default = last-used) → cached last-used (non-TTY fallback). The
    chosen value is cached so the next ride pre-fills with it.
    """
    import sys

    if explicit:
        _write_last_crew(explicit)
        return explicit

    default = _read_last_crew()
    if not sys.stdin.isatty():
        return default

    crew = click.prompt("Crew", default=default, show_default=True)
    crew = (crew or default).strip() or default
    _write_last_crew(crew)
    return crew


def prompt_recap_fields(
    fit_path: Path,
    *,
    origin: str | None = None,
    destination: str | None = None,
    road: str | None = None,
    subtitle: str | None = None,
    crew: str | None = None,
) -> dict[str, str]:
    """Resolve the five recap-card fields by prompting the user.

    Defaults come from compute_route_metadata (start place, farthest
    place, dominant named road) and the design tokens (subtitle).
    Explicit CLI overrides bypass the prompt for that field. Non-TTY
    runs accept all derived defaults silently. The chosen ``crew`` is
    cached for the next ride.

    Returns ``{"origin": ..., "destination": ..., "road": ...,
    "subtitle": ..., "crew": ...}`` — empty string means "no value,
    let downstream defaults apply" (e.g. local-loop with no end place).
    """
    import sys
    from .fit_parser import parse_fit
    from .route_metadata import compute_route_metadata
    from .design.tokens import (
        LOCKUP_ORIGIN as _DEF_ORIGIN,
        LOCKUP_ROAD as _DEF_ROAD,
    )
    from .intro_outro import _DEFAULT_OUTRO_SUBTITLE

    click.echo("Reading FIT + deriving route metadata…")
    ride = parse_fit(fit_path)
    try:
        meta = compute_route_metadata(ride)
    except Exception as exc:
        click.echo(f"  Route metadata failed ({exc.__class__.__name__}); using defaults.")
        meta = None

    def _meta_or(value: str | None, fallback: str) -> str:
        if value and not value.startswith("ERR:") and value != "UNKNOWN":
            return value
        return fallback

    derived_origin = _meta_or(meta.origin if meta else None, _DEF_ORIGIN)
    derived_far = _meta_or(meta.destination if meta else None, "")
    derived_road = _meta_or(meta.road if meta else None, _DEF_ROAD)

    if meta:
        click.echo(
            f"  GPS: {meta.ride_type}, {meta.distance_mi:.1f}mi, "
            f"max {meta.max_dist_from_start_mi:.1f}mi from start"
        )

    interactive = sys.stdin.isatty()

    def _resolve(label: str, explicit: str | None, default: str,
                 allow_empty: bool = False) -> str:
        if explicit is not None:
            return explicit
        if not interactive:
            return default
        # show_default puts the default in brackets; default_is_missing
        # means an Enter keystroke returns the default verbatim.
        return click.prompt(label, default=default, show_default=True,
                            type=str)

    origin_str = _resolve("Start ", origin, derived_origin)

    # If the farthest point falls in the same place as the start (e.g. a
    # park loop where the park is both origin and the farthest reach),
    # default the prompt to empty so
    # the recap card doesn't read "CENTRAL PARK → CENTRAL PARK".
    if derived_far and derived_far.strip().upper() == origin_str.strip().upper():
        far_default = ""
    else:
        far_default = derived_far
    far_str = _resolve("Far   ", destination, far_default, allow_empty=True)

    road_str = _resolve("Road  ", road, derived_road)
    subtitle_str = _resolve("Saying", subtitle, _DEFAULT_OUTRO_SUBTITLE)
    crew_str = resolve_crew(crew)

    return {
        "origin": origin_str.strip(),
        "destination": far_str.strip(),
        "road": road_str.strip(),
        "subtitle": subtitle_str.strip(),
        "crew": crew_str.strip(),
    }


_FFMPEG_TOOLS = ("ffmpeg", "ffprobe")


def require_ffmpeg() -> None:
    """Fail fast, and legibly, when ffmpeg/ffprobe are missing.

    Every video path in this project shells out to ffmpeg. Without this the
    first failure is a bare FileNotFoundError from deep inside a worker — and
    inside `process` it surfaces even worse, swallowed by the scan's broad
    exception handler and reported as "Gemini scan failed", which sends you
    debugging the wrong system entirely.
    """
    missing = [t for t in _FFMPEG_TOOLS if shutil.which(t) is None]
    if not missing:
        return
    raise click.ClickException(
        f"{' and '.join(missing)} not found on PATH.\n\n"
        "This project shells out to ffmpeg for all video work.\n"
        "  macOS:         brew install ffmpeg\n"
        "  Debian/Ubuntu: sudo apt install ffmpeg\n"
        "  Fedora:        sudo dnf install ffmpeg\n"
        "  Windows:       https://ffmpeg.org/download.html"
    )


@click.group()
def main():
    """GoPro + Garmin cycling video pipeline."""


@main.command()
@click.argument("fit_file", type=click.Path(exists=True, path_type=Path))
def inspect_fit(fit_file: Path):
    """Print a summary of a FIT file."""
    ride = parse_fit(fit_file)
    click.echo(f"Start:    {ride.start_time}")
    click.echo(f"End:      {ride.end_time}")
    click.echo(f"Duration: {ride.duration}")
    click.echo(f"Points:   {len(ride.points)}")

    if ride.points:
        powers = [p.power for p in ride.points if p.power is not None]
        hrs = [p.heart_rate for p in ride.points if p.heart_rate is not None]
        speeds = [p.speed for p in ride.points if p.speed is not None]
        cadences = [p.cadence for p in ride.points if p.cadence is not None]
        if powers:
            click.echo(f"Power:    avg={sum(powers)/len(powers):.0f}W  max={max(powers)}W")
        if hrs:
            click.echo(f"HR:       avg={sum(hrs)/len(hrs):.0f}bpm  max={max(hrs)}bpm")
        if speeds:
            avg_mph = sum(speeds) / len(speeds) * 2.23694
            max_mph = max(speeds) * 2.23694
            click.echo(f"Speed:    avg={avg_mph:.1f}mph  max={max_mph:.1f}mph")
        if cadences:
            click.echo(f"Cadence:  avg={sum(cadences)/len(cadences):.0f}rpm  max={max(cadences)}rpm")

        sensors = []
        if not ride.has_power:
            sensors.append("power")
        if not ride.has_cadence:
            sensors.append("cadence")
        if sensors:
            click.echo(f"Sensors:  no {', '.join(sensors)} data (overlay/highlights will adapt)")


@main.command()
@click.argument("video_dir", type=click.Path(exists=True, path_type=Path))
def inspect_video(video_dir: Path):
    """Print metadata for GoPro videos in a directory."""
    clips = extract_all(video_dir)
    for clip in clips:
        click.echo(f"{clip.path.name}:")
        click.echo(f"  Created:    {clip.creation_time}")
        click.echo(f"  Duration:   {clip.duration_secs:.1f}s")
        click.echo(f"  Resolution: {clip.width}x{clip.height}")
        click.echo(f"  FPS:        {clip.fps:.2f}")


@main.command()
@click.argument("fit_file", type=click.Path(exists=True, path_type=Path))
@click.option("--power-threshold", default=None, type=float,
              help="Power spike threshold (watts). Default: ~1.45x your FTP.")
@click.option("--speed-threshold", default=None, type=float,
              help="Speed spike threshold (m/s). Default: 12.0 (~27 mph).")
@click.option("--hr-threshold", default=None, type=float,
              help="HR spike threshold (bpm). Default: ~87% of your max HR.")
def find_highlights(fit_file: Path, power_threshold: float, speed_threshold: float, hr_threshold: float):
    """Detect interesting segments in a FIT file."""
    ride = parse_fit(fit_file)
    config = HighlightConfig(**{
        k: v for k, v in {
            "power_spike_threshold": power_threshold,
            "speed_spike_threshold": speed_threshold,
            "hr_spike_threshold": hr_threshold,
        }.items() if v is not None
    })
    highlights = detect_highlights(ride, config)

    click.echo(f"Found {len(highlights)} highlights:\n")
    for i, hl in enumerate(highlights):
        duration = hl.duration.total_seconds()
        click.echo(
            f"  [{i}] {hl.reason.value:12s}  "
            f"{hl.start_time.strftime('%H:%M:%S')} - {hl.end_time.strftime('%H:%M:%S')}  "
            f"({duration:.0f}s)  score={hl.score:.2f}  peak={hl.peak_value:.0f}"
        )


@main.command("extract-frames")
@click.argument("fit_file", type=click.Path(exists=True, path_type=Path))
@click.argument("video_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=None,
              help="Output directory (default: <video_dir>/frames)")
@click.option("--interval", default=1, help="Seconds between frames (default 1)")
@click.option("--width", default=480, help="Frame width in pixels (default 480)")
@click.option("--offset", default=0.0, help="Time offset in seconds (FIT - GoPro)")
@click.option("--workers", default=8, help="Parallel ffmpeg workers (default 8). Lower if RAM is tight.")
def extract_frames(fit_file: Path, video_dir: Path, output_dir: Path | None,
                   interval: int, width: int, offset: float, workers: int):
    """Extract downsampled frames from all clips, indexed by ride time.

    Saves JPEGs named frame_NNNNNN.jpg (ride second) + manifest.json.
    Run this once before using the labeler.
    """
    require_ffmpeg()
    from .extract_frames import extract_ride_frames

    if output_dir is None:
        output_dir = video_dir / "frames"

    extract_ride_frames(str(fit_file), str(video_dir), str(output_dir),
                        interval=interval, width=width, offset=offset,
                        workers=workers)


@main.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=Path))
@click.argument("fit_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None, help="Output .mp4 path")
@click.option("--offset", default=0.0, help="Time offset in seconds (FIT - GoPro)")
@click.option("--portrait", is_flag=True, help="Render in portrait (9:16) layout")
@click.option("--start", "start_time", default=None, help="Start time as MM:SS (e.g. 47:00)")
@click.option("--duration", "duration", default=0.0, type=float, help="Duration in seconds to render")
@click.option("--master", is_flag=True, help="Use master encoding (libx264, high quality) instead of preview")
@click.option("--intro", "intro_secs", default=0.0, type=float,
              help="Opening blur→clear title card with date + time-of-day (seconds, e.g. 3)")
def burn(video_path: Path, fit_file: Path, output: Path | None, offset: float,
         portrait: bool, start_time: str | None, duration: float, master: bool,
         intro_secs: float):
    """Burn telemetry overlay permanently onto a video.

    VIDEO_PATH: Path to a GoPro .mp4 file
    FIT_FILE: Path to the Garmin .fit file
    """
    require_ffmpeg()
    from .burn_overlay import ENCODE_MASTER, ENCODE_PREVIEW, burn_overlay

    layout = "portrait" if portrait else "landscape"

    start_secs = 0.0
    if start_time:
        parts = start_time.split(":")
        if len(parts) == 2:
            start_secs = int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            start_secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

    if output is None:
        suffix = "_portrait" if portrait else "_overlay"
        output = video_path.with_name(video_path.stem + suffix + ".mp4")

    burn_overlay(str(video_path), str(fit_file), str(output), offset,
                 layout=layout, start_offset=start_secs, trim_duration=duration,
                 encode_preset=ENCODE_MASTER if master else ENCODE_PREVIEW,
                 intro_secs=intro_secs)


@main.command()
@click.argument("video_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("fit_file", type=click.Path(exists=True, path_type=Path))
@click.option("--labels", "labels_file", type=click.Path(exists=True, path_type=Path), default=None,
              help="Optional labels JSON file for additional candidates")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=None,
              help="Output directory (default: <video_dir>/highlights)")
@click.option("--offset", default=0.0, help="Time offset in seconds (FIT - GoPro). "
              "If 0 (default), the pipeline auto-detects offset from GoPro GPMF GPS.")
@click.option("--no-auto-sync", is_flag=True,
              help="Disable GPMF auto-sync (use --offset value as-is, even 0).")
@click.option("--landscape-duration", default=60.0, help="Target landscape video duration (seconds)")
@click.option("--portrait-duration", default=30.0, help="Target portrait video duration (seconds)")
@click.option("--segment-duration", default=3.0, help="Duration of each clip segment (seconds)")
@click.option("--landscape-only", is_flag=True, help="Only produce landscape video")
@click.option("--strava-activity", type=int, default=None,
              help="Strava activity ID — injects popular segments as candidates")
@click.option("--preview", is_flag=True, help="Use fast preview encoding (VideoToolbox) instead of master (libx264)")
@click.option("--skip-gemini", is_flag=True, help="Skip Gemini vision scan")
@click.option("--skip-narrative", is_flag=True, help="Skip Gemini narrative selection — fall back to greedy ranking")
@click.option("--no-upload", is_flag=True, help="Skip 1080p compress + Drive upload after compose")
@click.option("--origin", default=None,
              help="Title + start-pin label on the recap card (e.g. 'CENTRAL PARK'). "
              "Auto-derived from GPS if omitted.")
@click.option("--destination", default=None,
              help="Destination label, e.g. 'PIERMONT'. Title becomes "
              "'{origin} → {destination}'.")
@click.option("--subtitle", default=None,
              help="Italic tagline under the title (e.g. 'the long way home.').")
@click.option("--road", default=None,
              help="Left side of the footer lockup (e.g. 'PARK LOOP'). "
              "Auto-derived from GPS if omitted.")
@click.option("--crew", default=None,
              help="Right side of the footer lockup (e.g. 'SOLO'). If omitted, "
              "prompts on a TTY using the last-used value as the default.")
@click.option("--lockup", default=None,
              help="Full in-segment bottom-band string. Overrides origin/road for "
              "the per-clip HUD only; recap card still uses --origin/--road/--crew.")
def compose(video_dir: Path, fit_file: Path, labels_file: Path | None, output_dir: Path | None,
            offset: float, no_auto_sync: bool,
            landscape_duration: float, portrait_duration: float,
            segment_duration: float,
            landscape_only: bool, strava_activity: int | None,
            preview: bool, skip_gemini: bool, skip_narrative: bool, no_upload: bool,
            origin: str | None, destination: str | None, subtitle: str | None,
            road: str | None, crew: str | None, lockup: str | None):
    """Compose highlight videos from ride telemetry + optional labels.

    VIDEO_DIR: Directory containing GoPro .mp4 files
    FIT_FILE: Path to the Garmin .fit file
    """
    require_ffmpeg()
    from .burn_overlay import ENCODE_MASTER, ENCODE_PREVIEW
    from .composer import ComposerConfig, compose_highlight
    from .gpmf_sync import resolve_offsets
    from .utils import keep_system_awake

    if output_dir is None:
        output_dir = video_dir / "highlights"

    # Auto-detect ride_labels.json if --labels wasn't passed.
    # Mirrors `process` so users don't have to pass --labels for the
    # common case of "compose this date folder I just labeled."
    if labels_file is None:
        candidate = video_dir / "ride_labels.json"
        if candidate.exists():
            labels_file = candidate
            click.echo(f"Labels:   {candidate.name} (auto-detected)")

    per_clip_offsets, resolved_offset, source = resolve_offsets(
        video_dir, manual_offset=offset, auto=not no_auto_sync,
    )
    click.echo(f"Sync:     {resolved_offset:+.2f}s  ({source})")
    if per_clip_offsets:
        click.echo("Per-clip offsets:")
        for name in sorted(per_clip_offsets):
            click.echo(f"  {name}: {per_clip_offsets[name]:+.2f}s")

    # Prompt for recap-card fields up-front so the long-running burn
    # doesn't block mid-stream. Each flag passed on the CLI skips that
    # field's prompt; everything else defaults to the GPS-derived value
    # (origin / far / road) or the design-token default (subtitle).
    fields = prompt_recap_fields(
        fit_file, origin=origin, destination=destination,
        road=road, subtitle=subtitle, crew=crew,
    )

    config = ComposerConfig(
        landscape_duration=landscape_duration,
        portrait_duration=portrait_duration,
        segment_duration=segment_duration,
        offset=resolved_offset,
        per_clip_offsets=per_clip_offsets or None,
        landscape_only=landscape_only,
        strava_activity_id=strava_activity,
        skip_gemini=skip_gemini,
        skip_narrative=skip_narrative,
        origin=fields["origin"],
        destination=fields["destination"],
        subtitle=fields["subtitle"],
        road=fields["road"],
        crew=fields["crew"],
        lockup=lockup,
    )
    with keep_system_awake() as awake:
        if awake:
            click.echo("System sleep paused (caffeinate -di) for the duration.")
        results = compose_highlight(
            video_dir, fit_file, output_dir, config,
            labels_path=labels_file,
            encode_preset=ENCODE_PREVIEW if preview else ENCODE_MASTER,
        )
    click.echo("\nDone!")
    for fmt, path in results.items():
        click.echo(f"  {fmt}: {path}")

    if not no_upload:
        from .share import share_outputs
        share_outputs(video_dir.parent if video_dir.name == "highlights" else video_dir, results)


@main.command()
@click.argument("date_folder", type=click.Path(exists=True, path_type=Path))
@click.option("--offset", default=0.0, help="Manual FIT-vs-GoPro offset in seconds. "
              "If 0 (default), the pipeline auto-detects offset from GoPro GPMF GPS.")
@click.option("--no-auto-sync", is_flag=True,
              help="Disable GPMF auto-sync (use --offset value as-is, even 0).")
@click.option("--strava-activity", type=int, default=None,
              help="Strava activity ID for segment candidates")
@click.option("--landscape-duration", default=60.0, help="Landscape video duration (seconds)")
@click.option("--portrait-duration", default=30.0, help="Portrait video duration (seconds)")
@click.option("--segment-duration", default=3.0, help="Clip segment duration (seconds)")
@click.option("--skip-gemini", is_flag=True, help="Skip Gemini vision scan")
@click.option("--skip-review", is_flag=True, help="Skip candidate review, auto-select best")
@click.option("--no-upload", is_flag=True, help="Skip 1080p compress + Drive upload after compose")
@click.option("--port", default=8502, help="Port for candidate reviewer")
@click.option("--origin", default=None,
              help="Title + start-pin label on the recap card. Auto-derived from GPS "
              "if omitted (e.g. 'HUDSON YARDS', 'CENTRAL PARK').")
@click.option("--destination", default=None,
              help="Destination label for out-and-back rides. Auto-derived from GPS "
              "(farthest point from start) if omitted.")
@click.option("--subtitle", default=None,
              help="Italic tagline under the title. Defaults to design token.")
@click.option("--road", default=None,
              help="Left side of the footer lockup. Auto-derived from GPS "
              "(most-traveled named road or route ref) if omitted.")
@click.option("--crew", default=None,
              help="Right side of the footer lockup. If omitted, prompts on a TTY "
              "using the last-used value as the default.")
@click.option("--lockup", default=None,
              help="Full in-segment bottom-band string. Overrides origin/road for "
              "the per-clip HUD only; recap card still uses --origin/--road/--crew.")
@click.option("--far-pin", is_flag=True,
              help="Anchor the recap end pin at the farthest point reached (the ride "
              "apex/turnaround) instead of where the ride ended. Use when you trained "
              "or drove home from a different town than the real destination.")
def process(date_folder: Path, offset: float, no_auto_sync: bool,
            strava_activity: int | None,
            landscape_duration: float, portrait_duration: float,
            segment_duration: float,
            skip_gemini: bool, skip_review: bool, no_upload: bool, port: int,
            origin: str | None, destination: str | None, subtitle: str | None,
            road: str | None, crew: str | None, lockup: str | None,
            far_pin: bool = False):
    """Full pipeline: detect → review → compose highlight videos.

    DATE_FOLDER: Date folder (e.g. data/raw/2026-04-11/) containing
    GoPro .MP4 files and a Garmin .fit file.

    Steps:
      0. Auto-sync the GoPro and FIT clocks via GPMF GPS (unless
         --offset is explicitly set or --no-auto-sync is passed).
      1. Discover GoPro videos + FIT file in the folder
      2. Generate candidates from telemetry + Strava + Gemini + labels
      3. Launch candidate reviewer (or auto-select if --skip-review)
      4. Compose landscape + portrait videos from selections
    """
    require_ffmpeg()
    from .utils import keep_system_awake

    # Preflight: discover FIT + MP4s before the interactive prompts and
    # caffeinate so a missing file fails fast instead of after the user
    # has answered five recap questions. (FIT is also needed up-front so
    # the recap prompt has GPS context.)
    fit_files = _find_fits(date_folder)
    if not fit_files:
        raise click.ClickException("No .fit file found in the date folder.")
    if len(fit_files) > 1:
        click.echo(
            f"Warning: {len(fit_files)} FIT files found "
            f"({', '.join(f.name for f in fit_files)}) — "
            f"using {fit_files[0].name} (first in sorted order)."
        )
    fit_path = fit_files[0]

    mp4_files = _find_mp4s(date_folder)
    if not mp4_files:
        raise click.ClickException("No .MP4 files found in the date folder.")

    # Prompt for recap-card fields before caffeinate / heavy work so the
    # user can walk away after answering. Each --flag skips its prompt.
    fields = prompt_recap_fields(
        fit_path, origin=origin, destination=destination,
        road=road, subtitle=subtitle, crew=crew,
    )

    # Hold the system awake for the duration. 4K libx264 burns get
    # devastated by App Nap if the display sleeps mid-pipeline.
    with keep_system_awake() as awake:
        if awake:
            click.echo("System sleep paused (caffeinate -di) for the duration.")
        _process_body(
            date_folder, offset, no_auto_sync, strava_activity,
            landscape_duration, portrait_duration, segment_duration,
            skip_gemini, skip_review, no_upload, port,
            origin=fields["origin"], destination=fields["destination"],
            subtitle=fields["subtitle"], road=fields["road"],
            crew=fields["crew"], lockup=lockup, far_pin=far_pin,
        )


def _process_body(date_folder: Path, offset: float, no_auto_sync: bool,
                  strava_activity: int | None,
                  landscape_duration: float, portrait_duration: float,
                  segment_duration: float,
                  skip_gemini: bool, skip_review: bool, no_upload: bool, port: int,
                  *,
                  origin: str | None = None, destination: str | None = None,
                  subtitle: str | None = None, road: str | None = None,
                  crew: str | None = None, lockup: str | None = None,
                  far_pin: bool = False):
    """Body of `process` — extracted so caffeinate wraps the whole run."""
    from .composer import ComposerConfig, generate_all_candidates, compose_highlight
    from .gpmf_sync import resolve_offset

    # Discover files (`process` already preflighted and warned about
    # multiple FITs; sorted-first keeps the pick deterministic here too)
    fit_files = _find_fits(date_folder)
    mp4_files = _find_mp4s(date_folder)

    if not fit_files:
        raise click.ClickException("No .fit file found in the date folder.")
    if not mp4_files:
        raise click.ClickException("No .MP4 files found in the date folder.")

    fit_path = fit_files[0]
    click.echo(f"FIT file: {fit_path.name}")
    click.echo(f"Videos:   {len(mp4_files)} GoPro files")

    # Check for LRV proxy files (low-res 480p H.264 — 10-20x faster to decode)
    lrv_files = list(date_folder.glob("*.LRV")) + list(date_folder.glob("*.lrv"))
    if lrv_files:
        click.echo(f"Proxies:  {len(lrv_files)} LRV files")
    else:
        click.echo("Proxies:  none (tip: copy .LRV files from GoPro SD card for 10x faster scans)")

    # Check for labels
    labels_path = date_folder / "ride_labels.json"
    if not labels_path.exists():
        labels_path = None
    else:
        click.echo(f"Labels:   {labels_path.name}")

    # Auto-detect FIT/GoPro offset before generating candidates so every
    # downstream consumer sees the corrected timing baked into anchors.
    resolved_offset, source = resolve_offset(
        date_folder, manual_offset=offset, auto=not no_auto_sync,
    )
    click.echo(f"Sync:     {resolved_offset:+.2f}s  ({source})")

    config = ComposerConfig(
        landscape_duration=landscape_duration,
        portrait_duration=portrait_duration,
        segment_duration=segment_duration,
        offset=resolved_offset,
        strava_activity_id=strava_activity,
        skip_gemini=skip_gemini,
        origin=origin,
        destination=destination,
        subtitle=subtitle,
        road=road,
        crew=crew,
        lockup=lockup,
        far_pin=far_pin,
    )

    # Step 1: Generate candidates
    click.echo("\n=== Step 1: Generating candidates ===")
    candidates, ride, synced_clips = generate_all_candidates(
        date_folder, fit_path, config, labels_path=labels_path,
    )

    if not candidates:
        click.echo("No candidates found. Check your FIT file and video files.")
        return

    click.echo(f"\n{len(candidates)} candidates ready for review:")
    for i, c in enumerate(candidates[:10]):
        notes = c.label.get("notes", "")[:50]
        click.echo(f"  {c.ride_time_secs:>7.0f}s  score={c.score:5.1f}  [{c.source}] {notes}")
    if len(candidates) > 10:
        click.echo(f"  ... and {len(candidates) - 10} more")

    # Always write moments.json — the canonical artifact. compose-selected
    # reads it. Status starts as `pending`; --skip-review marks the
    # auto-selected subset as `auto` after composition.
    from .proposals import (
        proposals_from_segments, save_proposals,
        STATUS_AUTO,
    )
    proposals = proposals_from_segments(candidates)
    moments_file = date_folder / "moments.json"
    save_proposals(proposals, moments_file)
    click.echo(f"\nSaved {len(proposals)} moments to {moments_file}")

    if skip_review:
        # Auto-select: compose directly from top candidates
        click.echo("\n=== Step 2: Auto-selecting (--skip-review) ===")
        output_dir = date_folder / "highlights"

        # Run selection here so we know which moments to mark as `auto`.
        # compose_highlight will re-select internally with the same logic
        # — same inputs, deterministic output (modulo Gemini narrative
        # selection, which is allowed a small amount of nondeterminism).
        from .composer import select_segments
        landscape_segs = select_segments(
            config.landscape_duration, config,
            precomputed_candidates=candidates, layout="landscape",
        )
        portrait_segs = []
        if not config.landscape_only:
            portrait_segs = select_segments(
                config.portrait_duration, config,
                precomputed_candidates=candidates, layout="portrait",
            )
        selected_ids = {
            s.stable_id for s in (*landscape_segs, *portrait_segs)
        }
        for p in proposals:
            if p.stable_id in selected_ids:
                p.status = STATUS_AUTO
        save_proposals(proposals, moments_file)
        click.echo(
            f"  Marked {len(selected_ids)} moments as auto-selected "
            f"({len(landscape_segs)} landscape + "
            f"{len(portrait_segs)} portrait)"
        )

        results = compose_highlight(
            date_folder, fit_path, output_dir, config,
            labels_path=labels_path,
            precomputed_candidates=candidates,
        )
        click.echo("\nDone!")
        for fmt, path in results.items():
            click.echo(f"  {fmt}: {path}")
        if not no_upload:
            from .share import share_outputs
            share_outputs(date_folder, results)
    else:
        # Reviewer input — sorted by score descending; a thin projection
        # of the same MomentProposals in the shape the Streamlit
        # reviewer consumes.
        import json
        candidates_json = [
            {
                "stable_id": c.stable_id,
                "clip_name": c.clip_name,
                "video_start": c.video_start,
                "video_end": c.video_end,
                "anchor_video_secs": c.anchor_video_secs,
                "ride_time_secs": c.ride_time_secs,
                "score": c.score,
                "source": c.source,
                "sources": c.sources,
                "rubric": c.rubric,
                "portrait_crop_bias": c.portrait_crop_bias,
                "notes": c.label.get("notes", ""),
                "type": c.label.get("type", ""),
                "star_count": c.label.get("star_count", 0),
            }
            for c in candidates
        ]
        candidates_file = date_folder / "candidates.json"
        candidates_file.write_text(json.dumps(candidates_json, indent=2))
        click.echo(f"Saved {len(candidates)} candidates to {candidates_file}")

        # Launch reviewer
        click.echo("\n=== Step 2: Launching candidate reviewer ===")
        import subprocess as sp
        import sys

        app_path = Path(__file__).parent / "candidate_review.py"
        cmd = [
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", str(port),
            "--",
            "--video-dir", str(date_folder),
            "--fit", str(fit_path),
            "--offset", str(resolved_offset),
        ]
        if labels_path:
            cmd.extend(["--labels", str(labels_path)])
        if strava_activity:
            cmd.extend(["--strava-activity", str(strava_activity)])

        click.echo(f"Opening reviewer at http://localhost:{port}")
        click.echo("After reviewing, run: gopro-garmin compose-selected "
                    f"{date_folder}/selected_candidates.json {date_folder} {fit_path}")
        sp.run(cmd)


@main.command("review-candidates")
@click.argument("video_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("fit_file", type=click.Path(exists=True, path_type=Path))
@click.option("--labels", "labels_file", type=click.Path(exists=True, path_type=Path), default=None,
              help="Optional labels JSON file")
@click.option("--offset", default=0.0, help="Time offset in seconds (FIT - GoPro)")
@click.option("--strava-activity", type=int, default=None, help="Strava activity ID")
@click.option("--port", default=8502, help="Port for Streamlit server")
def review_candidates(video_dir: Path, fit_file: Path, labels_file: Path | None,
                      offset: float, strava_activity: int | None, port: int):
    """Browse, pick, and rate highlight candidates in a visual UI.

    Generates candidates from FIT telemetry highlights + optional Strava segments
    + optional labels. Pick which to include, rate 1-5, then compose.
    """
    require_ffmpeg()
    import subprocess
    import sys

    app_path = Path(__file__).parent / "candidate_review.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(port),
        "--",
        "--video-dir", str(video_dir),
        "--fit", str(fit_file),
        "--offset", str(offset),
    ]
    if labels_file:
        cmd.extend(["--labels", str(labels_file)])
    if strava_activity:
        cmd.extend(["--strava-activity", str(strava_activity)])

    click.echo(f"Opening candidate reviewer at http://localhost:{port}")
    subprocess.run(cmd)


@main.command("compose-selected")
@click.argument("selections_file", type=click.Path(exists=True, path_type=Path))
@click.argument("video_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("fit_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--offset", default=0.0, help="Manual FIT-vs-GoPro offset (seconds). "
              "Default 0 ↔ read from <video_dir>/sync.json if present, else auto-detect.")
@click.option("--no-auto-sync", is_flag=True,
              help="Disable GPMF auto-sync. Use --offset value as-is even if 0.")
@click.option("--layout", type=click.Choice(["landscape", "portrait"]), default="landscape")
@click.option("--preview", is_flag=True, help="Use fast preview encoding instead of master")
@click.option("--trim", default=3.0, type=float,
              help="Final clip width in seconds, centered on each candidate's anchor (default 3.0). "
                   "Set to 0 to use the full review-span window from candidates.json.")
@click.option("--origin", default=None,
              help="Recap card title + start pin. Auto-derived from GPS if omitted.")
@click.option("--destination", default=None,
              help="Recap card destination. Auto-derived from GPS (farthest point) "
              "for out-and-back rides if omitted.")
@click.option("--subtitle", default=None, help="Italic tagline under the title.")
@click.option("--road", default=None,
              help="Lockup left side. Auto-derived from GPS if omitted.")
@click.option("--crew", default=None,
              help="Lockup right side. If omitted, prompts on a TTY using the "
              "last-used value as the default.")
@click.option("--lockup", default=None,
              help="Full in-segment bottom-band string. Overrides per-clip HUD only.")
@click.option("--far-pin", is_flag=True,
              help="Anchor the recap end pin at the farthest point reached (the ride "
              "apex/turnaround) instead of where the ride ended. Use when you trained "
              "or drove home from a different town than the real destination.")
def compose_selected(selections_file: Path, video_dir: Path, fit_file: Path,
                     output_dir: Path | None, offset: float, no_auto_sync: bool,
                     layout: str, preview: bool, trim: float,
                     origin: str | None, destination: str | None, subtitle: str | None,
                     road: str | None, crew: str | None, lockup: str | None,
                     far_pin: bool = False):
    """Compose a highlight video from hand-picked candidate selections.

    SELECTIONS_FILE: moments.json or selected_candidates.json
    """
    require_ffmpeg()
    from .burn_overlay import ENCODE_MASTER, ENCODE_PREVIEW
    from .composer import compose_from_selections
    from .gpmf_sync import resolve_offset
    from .utils import keep_system_awake

    if output_dir is None:
        output_dir = video_dir / "highlights"

    resolved_offset, source = resolve_offset(
        video_dir, manual_offset=offset, auto=not no_auto_sync,
    )
    click.echo(f"Sync: {resolved_offset:+.2f}s  ({source})")

    # Prompt for recap-card fields up-front so we don't block partway
    # through the burn. --flags skip individual prompts.
    fields = prompt_recap_fields(
        fit_file, origin=origin, destination=destination,
        road=road, subtitle=subtitle, crew=crew,
    )

    # Hold the system awake through the burn — same reason as `process`.
    with keep_system_awake() as awake:
        if awake:
            click.echo("System sleep paused (caffeinate -di) for the duration.")
        result = compose_from_selections(
            selections_file, video_dir, fit_file, output_dir,
            offset=resolved_offset, layout=layout,
            encode_preset=ENCODE_PREVIEW if preview else ENCODE_MASTER,
            trim_secs=trim,
            origin=fields["origin"], destination=fields["destination"],
            subtitle=fields["subtitle"], road=fields["road"],
            crew=fields["crew"], lockup=lockup, far_pin=far_pin,
        )
    click.echo(f"\nDone! {result}")


@main.command("label")
@click.argument("fit_file", type=click.Path(exists=True, path_type=Path))
@click.argument("video_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--offset", default=0.0, help="Time offset in seconds (FIT - GoPro)")
@click.option("--labels-file", type=click.Path(path_type=Path), default=None, help="Path to labels JSON file")
@click.option("--port", default=8501, help="Port for Streamlit server")
def label(fit_file: Path, video_dir: Path, offset: float, labels_file: Path | None, port: int):
    """Launch the ride labeler to scrub through video + telemetry and flag moments.

    FIT_FILE: Path to the Garmin .fit file
    VIDEO_DIR: Directory containing GoPro .mp4 files
    """
    require_ffmpeg()
    import subprocess
    import sys

    labeler_path = Path(__file__).parent / "labeler.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(labeler_path),
        "--server.port", str(port),
        "--",
        "--fit", str(fit_file),
        "--video-dir", str(video_dir),
        "--offset", str(offset),
    ]
    if labels_file:
        cmd.extend(["--labels-file", str(labels_file)])

    click.echo(f"Launching labeler at http://localhost:{port}")
    subprocess.run(cmd)


@main.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=Path))
@click.argument("fit_file", type=click.Path(exists=True, path_type=Path))
@click.option("--offset", default=0.0, help="Time offset in seconds (FIT - GoPro)")
@click.option("--port", default=5555, help="Port for the web server")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
def review(video_path: Path, fit_file: Path, offset: float, port: int, host: str):
    """Launch web-based video review with real-time Garmin overlays.

    VIDEO_PATH: Path to a GoPro .mp4 file
    FIT_FILE: Path to the Garmin .fit file
    """
    from .web.app import create_app
    import webbrowser

    click.echo("Parsing ride data...")
    app = create_app(video_path=str(video_path), fit_path=str(fit_file), offset=offset)
    click.echo(f"Starting review server at http://{host}:{port}")
    click.echo("Open in Safari for HEVC playback support.")
    webbrowser.open(f"http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


@main.command()
@click.option("--count", default=10, help="Number of recent activities to show")
def garmin_list(count: int):
    """List recent activities from Garmin Connect."""
    from .garmin_connect import list_activities

    click.echo("Connecting to Garmin Connect...")
    activities = list_activities(count)
    click.echo(f"\n{'ID':<15} {'Date':<22} {'Type':<15} {'Name'}")
    click.echo("-" * 70)
    for a in activities:
        click.echo(
            f"{a['activityId']:<15} {a.get('startTimeLocal', 'N/A'):<22} "
            f"{a.get('activityType', {}).get('typeKey', 'N/A'):<15} "
            f"{a.get('activityName', 'N/A')}"
        )


@main.command()
@click.option("--activity-id", type=int, help="Specific activity ID to download")
@click.option("--latest", is_flag=True, help="Download the most recent activity")
@click.option("--date", "target_date", type=click.DateTime(formats=["%Y-%m-%d"]), help="Download all cycling activities for a date (YYYY-MM-DD)")
@click.option("--output-dir", type=click.Path(path_type=Path), help="Output directory for FIT files")
def garmin_download(activity_id: int | None, latest: bool, target_date, output_dir: Path | None):
    """Download activity FIT files from Garmin Connect."""
    from .garmin_connect import download_activity, download_activities_for_date, download_latest

    if activity_id:
        click.echo(f"Downloading activity {activity_id}...")
        path = download_activity(activity_id, output_dir)
        click.echo(f"Saved to {path}")
    elif target_date:
        click.echo(f"Downloading cycling activities for {target_date.date()}...")
        paths = download_activities_for_date(target_date.date(), output_dir)
        if not paths:
            click.echo("No cycling activities found for that date.")
        for p in paths:
            click.echo(f"Saved to {p}")
    elif latest:
        click.echo("Downloading most recent activity...")
        path = download_latest(output_dir)
        click.echo(f"Saved to {path}")
    else:
        click.echo("Specify --activity-id, --latest, or --date. Use garmin-list to see activities.")


@main.command()
@click.argument("date_folder", type=click.Path(exists=True, path_type=Path))
@click.option("--offset", default=0.0, help="Time offset in seconds (FIT - GoPro)")
def compare(date_folder: Path, offset: float):
    """Compare human labels vs Gemini scan output for a ride.

    DATE_FOLDER: Date folder containing ride_labels.json and .gemini_cache/

    Produces a structured comparison report showing:
      - Matched moments (both human and Gemini agree)
      - Label-only moments (Gemini missed)
      - Gemini-only moments (user didn't flag)
      - Systematic pattern analysis
    """
    from .prompt_eval import compare_ride, enrich_gemini_hits_with_ride_time

    # Enrich Gemini cache with ride-relative timestamps for alignment
    click.echo("Enriching Gemini cache with ride timestamps...")
    enrich_gemini_hits_with_ride_time(date_folder, offset)

    click.echo("\nComparing labels vs Gemini scan...")
    report = compare_ride(date_folder)
    report.print_summary()


@main.command("eval-prompt")
@click.argument("date_folders", nargs=-1, type=click.Path(exists=True, path_type=Path))
def eval_prompt_cmd(date_folders: tuple[Path, ...]):
    """Analyze label/Gemini comparison and suggest prompt improvements.

    DATE_FOLDERS: One or more ride folders with prompt_eval.json files.
    Run 'compare' first to generate comparison reports.

    Uses Gemini to analyze systematic patterns in what the scan prompt
    gets right vs wrong, and suggests specific prompt changes.
    """
    from .prompt_eval import eval_prompt

    if not date_folders:
        click.echo("Provide one or more date folders with comparison reports.")
        return

    click.echo(f"Analyzing {len(date_folders)} ride(s) for prompt improvement...")
    result = eval_prompt(list(date_folders))

    if not result:
        return

    analysis = result.get("analysis", "")
    if analysis:
        click.echo(f"\n{'='*60}")
        click.echo("Analysis:")
        click.echo(f"  {analysis}")

    changes = result.get("prompt_changes", [])
    if changes:
        click.echo(f"\nSuggested prompt changes ({len(changes)}):")
        for i, change in enumerate(changes, 1):
            click.echo(f"\n  [{i}] {change.get('type', '').upper()} — {change.get('section', '')}")
            if change.get("current_text"):
                click.echo(f"      Current: {change['current_text'][:80]}")
            if change.get("suggested_text"):
                click.echo(f"      Suggest: {change['suggested_text'][:80]}")
            click.echo(f"      Why: {change.get('rationale', '')}")

    adj = result.get("suggested_score_adjustments", {})
    if adj:
        click.echo(f"\nScore adjustments: {adj.get('description', '')}")


@main.command("strava-auth")
def strava_auth():
    """Authorize with Strava (opens browser for OAuth)."""
    from .strava import authorize_interactive

    authorize_interactive()
    click.echo("Strava tokens saved to ~/.strava/tokens.json")


@main.command("strava-list")
@click.option("--count", default=10, help="Number of recent activities to show")
def strava_list(count: int):
    """List recent Strava activities."""
    from .strava import list_activities

    activities = list_activities(count)
    click.echo(f"\n{'ID':<15} {'Date':<22} {'Type':<15} {'Name'}")
    click.echo("-" * 70)
    for a in activities:
        click.echo(
            f"{a['id']:<15} {a.get('start_date_local', 'N/A')[:19]:<22} "
            f"{a.get('type', 'N/A'):<15} "
            f"{a.get('name', 'N/A')}"
        )


@main.command("strava-segments")
@click.argument("activity_id", type=int)
def strava_segments(activity_id: int):
    """Show segment efforts and star counts for a Strava activity."""
    from .strava import get_segment_efforts

    efforts = get_segment_efforts(activity_id)
    if not efforts:
        click.echo("No segment efforts found.")
        return

    click.echo(f"\n{'Start':>8}  {'Duration':>8}  {'Stars':>6}  Name")
    click.echo("-" * 70)
    for e in efforts:
        m, s = divmod(int(e.start_time_secs), 60)
        h, m = divmod(m, 60)
        start = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
        click.echo(
            f"{start:>8}  {e.elapsed_time_secs:>7.0f}s  {e.star_count:>6}  {e.name}"
        )


@main.command("ranker-status")
def ranker_status():
    """Show learned ranker status — training data and model weights."""
    from .learned_ranker import status as ranker_status_fn
    click.echo(ranker_status_fn())


@main.command("ranker-train")
def ranker_train():
    """Train the learned ranker from accumulated selection data."""
    from .learned_ranker import train
    model = train()
    if model is None:
        click.echo("\nNot enough data to train yet. Keep composing rides!")


@main.command("ranker-collect")
@click.argument("date_folder", type=click.Path(exists=True))
@click.argument("selection_file", type=click.Path(exists=True))
def ranker_collect(date_folder, selection_file):
    """Manually collect training data from a selection file.

    DATE_FOLDER is the ride folder (e.g. data/raw/2026-04-14).
    SELECTION_FILE is a selection_*.json or selected_candidates.json.
    """
    import json
    from pathlib import Path
    from .learned_ranker import collect_training_data

    date_folder = Path(date_folder)
    ride_id = date_folder.name

    # Load selected candidates
    selected = json.loads(Path(selection_file).read_text())
    selected_times = {s["ride_time_secs"] for s in selected}

    # Load or generate full candidate pool
    candidates_file = date_folder / "candidates.json"
    if candidates_file.exists():
        all_cands = json.loads(candidates_file.read_text())
    else:
        click.echo("No candidates.json found. Run 'gopro-garmin process' first.")
        return

    selected_indices = set()
    for i, c in enumerate(all_cands):
        if c.get("ride_time_secs") in selected_times:
            selected_indices.add(i)

    collect_training_data(all_cands, selected_indices, ride_id)
    n_pos = len(selected_indices)
    n_neg = len(all_cands) - n_pos
    click.echo(f"Collected {n_pos} positive + {n_neg} negative examples from {ride_id}")


@main.command("compare-features")
@click.argument("date_folder", type=click.Path(exists=True))
@click.option("--strava-activity", type=int, default=None, help="Strava activity ID")
@click.option("--skip-gemini", is_flag=True, help="Skip Gemini vision scan")
def compare_features(date_folder, strava_activity, skip_gemini):
    """Run selection two ways for A/B comparison: greedy vs Gemini narrative.

    Generates candidates once, then runs selection both with and without
    Gemini narrative selection. Saves each variant's selection to
    <date_folder>/selection_variant_<name>.json for side-by-side review.
    Does NOT burn overlays — use compose-selected on the variant you prefer.
    """
    require_ffmpeg()
    import json
    from .composer import ComposerConfig, generate_all_candidates, select_segments

    date_folder = Path(date_folder)
    fit_files = _find_fits(date_folder)
    if not fit_files:
        click.echo("No FIT file found.")
        return

    click.echo("=== Generating candidates ===")
    config_full = ComposerConfig(
        strava_activity_id=strava_activity,
        skip_gemini=skip_gemini,
    )
    all_candidates, _ride, _synced = generate_all_candidates(
        date_folder, fit_files[0], config_full,
    )

    variants = {
        "greedy": {"skip_narrative": True},
        "narrative": {"skip_narrative": False},
    }

    for name, flags in variants.items():
        click.echo(f"\n{'='*60}")
        click.echo(f"=== Variant: {name} ===")
        click.echo(f"{'='*60}")

        cfg = ComposerConfig(
            strava_activity_id=strava_activity,
            skip_gemini=skip_gemini,
            skip_narrative=flags["skip_narrative"],
        )

        segs = select_segments(
            cfg.landscape_duration, cfg,
            precomputed_candidates=all_candidates,
            layout="landscape",
        )

        click.echo(f"\n  Selected {len(segs)} clips:")
        for seg in segs:
            t = seg.ride_time_secs
            notes = seg.label.get("notes", "")[:45]
            src = "+".join(seg.sources) if seg.sources else seg.source
            click.echo(f"    {int(t//60):>3}:{int(t%60):02d}  "
                        f"score={seg.score:5.1f}  [{src}]  {notes}")

        out = []
        for seg in segs:
            out.append({
                "clip_name": seg.clip_name,
                "video_start": seg.video_start,
                "video_end": seg.video_end,
                "ride_time_secs": seg.ride_time_secs,
                "score": seg.score,
                "sources": seg.sources,
                "notes": seg.label.get("notes", ""),
                "type": seg.label.get("clip_type", seg.label.get("type", "")),
            })
        out_path = date_folder / f"selection_variant_{name}.json"
        out_path.write_text(json.dumps(out, indent=2))
        click.echo(f"  Saved → {out_path.name}")

    click.echo(f"\n{'='*60}")
    click.echo("Done! Compare the selection_variant_*.json files, then burn your favorite:")
    click.echo(f"  gopro-garmin compose-selected <selection_file> {date_folder} {fit_files[0]}")


_RAW_FILE_SUFFIXES = {".mp4", ".lrv", ".thm"}
_RAW_DIR_NAMES = {"frames", "_work"}
_RIDE_FOLDER_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def _parse_ride_date(folder_name: str) -> date | None:
    m = _RIDE_FOLDER_RE.match(folder_name)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _has_highlights(ride_dir: Path) -> bool:
    hl = ride_dir / "highlights"
    return hl.is_dir() and any(hl.glob("*.mp4"))


def _raw_targets(ride_dir: Path) -> tuple[list[Path], list[Path]]:
    files = [
        p for p in ride_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _RAW_FILE_SUFFIXES
    ]
    dirs = [ride_dir / name for name in _RAW_DIR_NAMES if (ride_dir / name).is_dir()]
    return files, dirs


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


@main.command("cleanup-raw")
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=Path("data/raw"), show_default=True,
              help="Root directory containing YYYY-MM-DD ride folders.")
@click.option("--older-than", "older_than_days", type=int, default=7, show_default=True,
              help="Only delete from rides at least this many days old.")
@click.option("--require-highlights/--no-require-highlights", default=True, show_default=True,
              help="Only delete from rides that have a highlights/ folder with mp4 clips.")
@click.option("--dry-run/--no-dry-run", default=False, show_default=True,
              help="Print what would be deleted without removing anything.")
def cleanup_raw(data_dir: Path, older_than_days: int, require_highlights: bool, dry_run: bool):
    """Delete raw video assets (.MP4/.LRV/.THM, frames/, _work/) from old, processed rides.

    Keeps highlights/, .fit, and JSON metadata. Folder age is parsed from the
    YYYY-MM-DD folder name, not mtime.
    """
    cutoff = date.today() - timedelta(days=older_than_days)
    ride_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())

    total_freed = 0
    deleted_count = 0
    skipped: list[tuple[str, str]] = []

    click.echo(f"Cutoff: rides on or before {cutoff} ({older_than_days}+ days old)")
    click.echo(f"Mode:   {'DRY RUN — no deletions' if dry_run else 'DELETING'}\n")

    for ride_dir in ride_dirs:
        ride_date = _parse_ride_date(ride_dir.name)
        if ride_date is None:
            skipped.append((ride_dir.name, "not a YYYY-MM-DD folder"))
            continue
        if ride_date > cutoff:
            skipped.append((ride_dir.name, f"too recent ({ride_date})"))
            continue
        if require_highlights and not _has_highlights(ride_dir):
            skipped.append((ride_dir.name, "no highlights/ folder"))
            continue

        files, dirs = _raw_targets(ride_dir)
        if not files and not dirs:
            skipped.append((ride_dir.name, "already clean"))
            continue

        ride_freed = 0
        for f in files:
            try:
                ride_freed += f.stat().st_size
            except OSError:
                pass
        for d in dirs:
            ride_freed += _dir_size(d)

        click.echo(f"  {ride_dir.name}: {len(files)} files + {len(dirs)} dirs  ({_human_bytes(ride_freed)})")

        if not dry_run:
            for f in files:
                f.unlink(missing_ok=True)
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

        total_freed += ride_freed
        deleted_count += 1

    click.echo("")
    if skipped:
        click.echo("Skipped:")
        for name, reason in skipped:
            click.echo(f"  {name}: {reason}")
        click.echo("")

    verb = "Would free" if dry_run else "Freed"
    click.echo(f"{verb} {_human_bytes(total_freed)} across {deleted_count} ride(s).")


if __name__ == "__main__":
    main()
