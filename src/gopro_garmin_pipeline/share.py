"""Compress highlight outputs to 1080p HEVC and upload to Google Drive via rclone.

Best-effort throughout: a missing encoder, missing rclone, or an unconfigured
remote is logged and skipped, never raised — the full-resolution highlights are
already burned and on disk by the time any of this runs, so nothing here is
worth failing a ten-minute pipeline over.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from pathlib import Path

import click

RCLONE_REMOTE = "gdrive"
DRIVE_FOLDER = "GoPro_Highlights"

_SHARE_HEIGHT = {"landscape": 1080, "portrait": 1920}
_SHARE_BITRATE = {"landscape": "12M", "portrait": "8M"}

# Hardware HEVC first (VideoToolbox on macOS, NVENC on Linux), then a
# software fallback. Picked by probing ffmpeg rather than by platform sniffing,
# since a given build may lack any of them. hevc_vaapi is deliberately absent:
# most Linux ffmpeg builds list it, but it fails without a -vaapi_device and a
# nv12/vaapi filter chain, neither of which we pass.
_ENCODER_PREFERENCE = [
    "hevc_videotoolbox",
    "hevc_nvenc",
    "libx265",
    "libx264",
]


@functools.lru_cache(maxsize=1)
def _pick_encoder() -> str | None:
    """First available HEVC/H.264 encoder in this ffmpeg build, or None."""
    if shutil.which("ffmpeg") is None:
        return None
    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True,
    )
    if probe.returncode != 0:
        return None
    for enc in _ENCODER_PREFERENCE:
        if enc in probe.stdout:
            return enc
    return None


def _compress(src: Path, layout: str, out_dir: Path) -> Path | None:
    """Downscale to a shareable 1080p. Returns None if no encoder is available."""
    encoder = _pick_encoder()
    if encoder is None:
        click.echo("  no usable ffmpeg HEVC/H.264 encoder found — skipping compression")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{layout}_1080p.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale=-2:{_SHARE_HEIGHT[layout]}",
        "-c:v", encoder,
    ]
    # x264/x265 take a quality target rather than a hardware bitrate.
    if encoder in ("libx265", "libx264"):
        cmd += ["-crf", "23", "-preset", "medium"]
    else:
        cmd += ["-b:v", _SHARE_BITRATE[layout]]
    if encoder != "libx264":
        cmd += ["-tag:v", "hvc1"]  # QuickTime/iOS need this to play HEVC
    cmd += ["-c:a", "aac", "-b:a", "128k", str(out)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        click.echo(f"  compression failed ({encoder}) — skipping:\n{tail}")
        return None
    return out


def _rclone_ready(remote: str) -> bool:
    if shutil.which("rclone") is None:
        return False
    probe = subprocess.run(
        ["rclone", "lsd", f"{remote}:"], capture_output=True, text=True,
    )
    return probe.returncode == 0


def _upload(file: Path, remote: str, folder: str) -> str | None:
    dest = f"{remote}:{folder}/{file.name}"
    cp = subprocess.run(
        ["rclone", "copyto", str(file), dest], capture_output=True, text=True,
    )
    if cp.returncode != 0:
        return None
    link = subprocess.run(
        ["rclone", "link", dest], capture_output=True, text=True,
    )
    if link.returncode != 0:
        return None
    return link.stdout.strip()


def share_outputs(
    date_folder: Path,
    compose_results: dict[str, Path],
    remote: str = RCLONE_REMOTE,
    drive_folder: str = DRIVE_FOLDER,
) -> dict[str, str]:
    """Compress + upload highlight outputs. Returns share links by layout."""
    eligible = {k: v for k, v in compose_results.items() if k in _SHARE_HEIGHT}
    if not eligible:
        return {}

    share_dir = date_folder / "highlights" / "share"
    click.echo("\n=== Sharing: compressing to 1080p ===")
    compressed: dict[str, Path] = {}
    for layout, src in eligible.items():
        out = _compress(Path(src), layout, share_dir)
        if out is None:
            continue
        size_mb = out.stat().st_size / (1024 * 1024)
        click.echo(f"  {layout}: {out.name} ({size_mb:.0f} MB)")
        compressed[layout] = out

    if not compressed:
        click.echo(f"nothing compressed — full-resolution highlights remain in "
                   f"{date_folder / 'highlights'}/")
        return {}

    if not _rclone_ready(remote):
        click.echo(
            f"\nrclone remote '{remote}:' not configured — skipping upload. "
            f"Compressed files at {share_dir}/"
        )
        return {}

    folder = f"{drive_folder}/{date_folder.name}"
    subprocess.run(
        ["rclone", "mkdir", f"{remote}:{folder}"],
        capture_output=True, check=False,
    )
    click.echo(f"\n=== Sharing: uploading to {remote}:{folder}/ ===")
    links: dict[str, str] = {}
    for layout, file in compressed.items():
        link = _upload(file, remote, folder)
        if link:
            links[layout] = link
            click.echo(f"  {layout}: {link}")
        else:
            click.echo(f"  {layout}: upload failed")
    return links
