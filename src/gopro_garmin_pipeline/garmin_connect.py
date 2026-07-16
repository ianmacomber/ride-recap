"""Download activity FIT files from Garmin Connect."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

from garminconnect import Garmin

from .config import get_settings

TOKEN_DIR = Path.home() / ".garminconnect"


def _get_client() -> Garmin:
    """Authenticate with Garmin Connect, reusing saved tokens when possible."""
    settings = get_settings()
    tokenstore = str(TOKEN_DIR)

    # Try restoring from saved tokens first
    if TOKEN_DIR.exists():
        try:
            client = Garmin()
            client.login(tokenstore)
            return client
        except Exception:
            pass

    if not settings.garmin_email or not settings.garmin_password:
        raise ValueError(
            "Set GARMIN_EMAIL and GARMIN_PASSWORD in your .env file or environment."
        )

    client = Garmin(settings.garmin_email, settings.garmin_password)
    client.login(tokenstore)

    return client


def list_activities(count: int = 10) -> list[dict]:
    """Return the most recent activities as dicts."""
    client = _get_client()
    return client.get_activities(0, count)


def download_activity(activity_id: int, output_dir: Path | None = None) -> Path:
    """Download a single activity as a FIT file.

    Returns the path to the downloaded .fit file.
    """
    settings = get_settings()
    out = output_dir or settings.raw_dir / "garmin"
    out.mkdir(parents=True, exist_ok=True)

    client = _get_client()
    fit_data = client.download_activity(
        activity_id, dl_fmt=client.ActivityDownloadFormat.ORIGINAL
    )

    # Garmin wraps FIT files in a ZIP archive
    fit_path = out / f"{activity_id}.fit"
    if zipfile.is_zipfile(io.BytesIO(fit_data)):
        with zipfile.ZipFile(io.BytesIO(fit_data)) as zf:
            fit_names = [n for n in zf.namelist() if n.endswith(".fit")]
            if fit_names:
                fit_path.write_bytes(zf.read(fit_names[0]))
            else:
                fit_path.write_bytes(fit_data)
    else:
        fit_path.write_bytes(fit_data)
    return fit_path


def download_latest(output_dir: Path | None = None) -> Path:
    """Download the most recent activity as a FIT file."""
    activities = list_activities(1)
    if not activities:
        raise RuntimeError("No activities found on Garmin Connect.")
    return download_activity(activities[0]["activityId"], output_dir)


def download_activities_for_date(
    target_date: date, output_dir: Path | None = None
) -> list[Path]:
    """Download all activities for a specific date."""
    client = _get_client()
    activities = client.get_activities_by_date(
        target_date.isoformat(), target_date.isoformat(), "cycling"
    )
    paths = []
    for activity in activities:
        path = download_activity(activity["activityId"], output_dir)
        paths.append(path)
    return paths
