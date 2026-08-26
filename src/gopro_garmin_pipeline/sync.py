"""Synchronize FIT ride data with GoPro video timestamps."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from .fit_parser import RideData, RidePoint
from .gopro_meta import GoProClip


def get_ride_timezone(timezone_name: str | None = None) -> ZoneInfo:
    """Return the configured ride timezone, with a backward-compatible default."""
    return ZoneInfo(timezone_name or "America/New_York")

def normalize_tz(a: dt.datetime, ref: dt.datetime) -> dt.datetime:
    """Strip or attach timezone info on *a* so it matches *ref*.

    Both GoPro (ffprobe) and Garmin (FIT) use UTC-ish GPS time, but
    one may carry tzinfo and the other may not. This function makes
    them comparable without silently converting across zones.
    """
    if a.tzinfo is not None and ref.tzinfo is None:
        return a.replace(tzinfo=None)
    if a.tzinfo is None and ref.tzinfo is not None:
        return a.replace(tzinfo=ref.tzinfo)
    return a


@dataclass
class SyncedClip:
    """A GoPro clip with its corresponding FIT data aligned."""

    clip: GoProClip
    ride: RideData
    offset_secs: float  # added to video time to get FIT time
    ride_timezone: ZoneInfo | None = None

    def _adjust(self, video_secs: float) -> dt.datetime:
        """Convert video time to FIT-aligned wall time with tz normalization."""
        wall = self.clip.video_time_to_wall_time(video_secs)
        adjusted = wall + dt.timedelta(seconds=self.offset_secs)
        if self.ride.start_time:
            adjusted = normalize_tz(adjusted, self.ride.start_time)
        return adjusted

    def ride_point_at_video_time(self, video_secs: float) -> RidePoint | None:
        """Get the FIT data point closest to a position in the video."""
        return self.ride.point_at(self._adjust(video_secs))

    def ride_points_for_video_range(
        self, start_secs: float, end_secs: float
    ) -> list[RidePoint]:
        """Get FIT data points for a range of video time."""
        return self.ride.points_in_range(
            self._adjust(start_secs), self._adjust(end_secs),
        )


def auto_sync(
    clip: GoProClip,
    ride: RideData,
    offset_secs: float = 0.0,
    ride_timezone: str | None = None,
) -> SyncedClip:
    """Create a synced clip with a manual time offset.

    Both the GoPro and Garmin Edge 540 use GPS time, so they should be
    closely aligned out of the box. The offset parameter allows fine-tuning
    if there's drift (positive = FIT data is ahead of video).

    Args:
        clip: GoPro video metadata.
        ride: Parsed FIT ride data.
        offset_secs: Manual offset in seconds (FIT time - GoPro time).

    Returns:
        SyncedClip ready for overlay rendering.
    """
    return SyncedClip(
        clip=clip,
        ride=ride,
        offset_secs=offset_secs,
        ride_timezone=get_ride_timezone(ride_timezone),
    )


def sync_all(
    clips: list[GoProClip],
    ride: RideData,
    offset_secs: float = 0.0,
    per_clip_offsets: dict[str, float] | None = None,
    ride_timezone: str | None = None,
) -> list[SyncedClip]:
    """Sync multiple GoPro clips to a single ride.

    When ``per_clip_offsets`` is given, each clip uses its own value
    (keyed by filename) and falls back to ``offset_secs``. This is the
    multi-recording fix — different GX*.MP4 files can have different
    RTC drift, and applying a global average misaligns later chapters.
    """
    per_clip = per_clip_offsets or {}
    return [
        auto_sync(
            clip, ride, per_clip.get(clip.path.name, offset_secs), ride_timezone
        )
        for clip in clips
    ]
