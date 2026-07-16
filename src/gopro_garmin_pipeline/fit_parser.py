"""Parse Garmin FIT files into structured ride data."""

from __future__ import annotations

import bisect
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

from fitparse import FitFile


@dataclass
class RidePoint:
    """A single data point from the FIT file."""

    timestamp: dt.datetime
    lat: float | None = None
    lon: float | None = None
    power: int | None = None
    heart_rate: int | None = None
    speed: float | None = None  # m/s
    cadence: int | None = None
    altitude: float | None = None
    temperature: int | None = None
    distance: float | None = None  # cumulative meters


@dataclass
class RideData:
    """Parsed ride data from a FIT file."""

    points: list[RidePoint] = field(default_factory=list)
    start_time: dt.datetime | None = None
    end_time: dt.datetime | None = None
    # Garmin session totals — authoritative for the recap card. The
    # session block reports moving (timer) time and elapsed time
    # separately, plus distance and avg/max speed in m/s. None when
    # the FIT file lacks a session message (rare).
    total_elapsed_secs: float | None = None
    total_timer_secs: float | None = None
    session_distance_m: float | None = None
    session_avg_speed_ms: float | None = None
    session_max_speed_ms: float | None = None
    _timestamps: list[float] = field(default_factory=list, repr=False)

    @property
    def duration(self) -> dt.timedelta | None:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def has_power(self) -> bool:
        """True if any point recorded power data (power meter connected)."""
        return any(p.power is not None for p in self.points)

    @property
    def has_cadence(self) -> bool:
        """True if any point recorded cadence data (cadence sensor connected)."""
        return any(p.cadence is not None for p in self.points)

    def _build_index(self):
        """Build a sorted timestamp index for bisect-based lookups."""
        if self.points and not self._timestamps:
            epoch = self.points[0].timestamp
            self._timestamps = [
                (p.timestamp - epoch).total_seconds() for p in self.points
            ]

    def points_in_range(
        self, start: dt.datetime, end: dt.datetime
    ) -> list[RidePoint]:
        """Return points within a time range (bisect-accelerated)."""
        if not self.points:
            return []
        self._build_index()
        epoch = self.points[0].timestamp
        s = (start - epoch).total_seconds()
        e = (end - epoch).total_seconds()
        lo = bisect.bisect_left(self._timestamps, s)
        hi = bisect.bisect_right(self._timestamps, e)
        return self.points[lo:hi]

    def point_at(self, timestamp: dt.datetime) -> RidePoint | None:
        """Return the closest point to a given timestamp (bisect-accelerated)."""
        if not self.points:
            return None
        self._build_index()
        epoch = self.points[0].timestamp
        t = (timestamp - epoch).total_seconds()
        i = bisect.bisect_left(self._timestamps, t)
        # Check neighbors to find the closest
        best = None
        best_dist = float("inf")
        for j in (i - 1, i):
            if 0 <= j < len(self._timestamps):
                d = abs(self._timestamps[j] - t)
                if d < best_dist:
                    best_dist = d
                    best = j
        return self.points[best] if best is not None else None

    def point_index(self, point: RidePoint) -> int:
        """Return the index of a point using bisect (O(log n) instead of O(n))."""
        if not self.points:
            return -1
        self._build_index()
        epoch = self.points[0].timestamp
        t = (point.timestamp - epoch).total_seconds()
        i = bisect.bisect_left(self._timestamps, t)
        if i < len(self.points) and self.points[i] is point:
            return i
        # Fallback: check neighbors (handles float precision)
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(self.points) and self.points[j] is point:
                return j
        return -1


def _semicircles_to_degrees(semicircles: int) -> float:
    """Convert Garmin semicircle units to decimal degrees."""
    return semicircles * (180.0 / 2**31)


def parse_fit(path: str | Path) -> RideData:
    """Parse a FIT file and return structured ride data.

    Args:
        path: Path to a .fit file (typically from a Garmin Edge 540).

    Returns:
        RideData with all recorded data points.
    """
    fit = FitFile(str(path))
    ride = RideData()

    for record in fit.get_messages("record"):
        values = {f.name: f.value for f in record.fields}

        timestamp = values.get("timestamp")
        if timestamp is None:
            continue

        # Garmin stores coordinates in semicircles
        lat_raw = values.get("position_lat")
        lon_raw = values.get("position_long")
        lat = _semicircles_to_degrees(lat_raw) if lat_raw is not None else None
        lon = _semicircles_to_degrees(lon_raw) if lon_raw is not None else None

        point = RidePoint(
            timestamp=timestamp,
            lat=lat,
            lon=lon,
            power=values.get("power"),
            heart_rate=values.get("heart_rate"),
            speed=values.get("enhanced_speed") or values.get("speed"),
            cadence=values.get("cadence"),
            altitude=values.get("enhanced_altitude") or values.get("altitude"),
            temperature=values.get("temperature"),
            distance=values.get("distance"),
        )
        ride.points.append(point)

    if ride.points:
        ride.start_time = ride.points[0].timestamp
        ride.end_time = ride.points[-1].timestamp

    for session in fit.get_messages("session"):
        v = {f.name: f.value for f in session.fields}
        ride.total_elapsed_secs = v.get("total_elapsed_time")
        ride.total_timer_secs = v.get("total_timer_time")
        ride.session_distance_m = v.get("total_distance")
        ride.session_avg_speed_ms = (
            v.get("enhanced_avg_speed") or v.get("avg_speed")
        )
        ride.session_max_speed_ms = (
            v.get("enhanced_max_speed") or v.get("max_speed")
        )
        break

    return ride
