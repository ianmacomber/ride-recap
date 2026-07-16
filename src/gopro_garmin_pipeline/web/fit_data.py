"""Pre-process FIT + sync data into JSON-serializable format for the web frontend."""

from __future__ import annotations

import datetime as dt

from ..sync import SyncedClip, normalize_tz
from ..utils import M_TO_FT, M_TO_MILES, MS_TO_MPH, compute_gradient


def prepare_ride_json(synced: SyncedClip) -> dict:
    """Convert a SyncedClip into a JSON-serializable dict for the frontend.

    Timestamps are converted to video-relative seconds so the frontend
    can do a simple binary search on video.currentTime.
    """
    clip = synced.clip
    ride = synced.ride

    # Build the full GPS route for the mini-map (all ride points, not just this clip)
    route = []
    for p in ride.points:
        if p.lat is not None and p.lon is not None:
            route.append([p.lat, p.lon])

    # Compute route bounding box
    if route:
        lats = [r[0] for r in route]
        lons = [r[1] for r in route]
        route_bounds = {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lon": min(lons),
            "max_lon": max(lons),
        }
    else:
        route_bounds = {"min_lat": 0, "max_lat": 0, "min_lon": 0, "max_lon": 0}

    # Total ride distance
    distances = [p.distance for p in ride.points if p.distance is not None]
    total_distance_miles = max(distances) * M_TO_MILES if distances else 0

    # Pre-compute gradient for each point (% grade from altitude/distance deltas)
    gradients: dict[int, float] = {}
    for i in range(1, len(ride.points)):
        p0, p1 = ride.points[i - 1], ride.points[i]
        if (p0.altitude is not None and p1.altitude is not None
                and p0.distance is not None and p1.distance is not None):
            gradients[i] = compute_gradient(p0.altitude, p0.distance, p1.altitude, p1.distance)

    # Build elevation profile for the full ride
    elevation_profile = []
    for p in ride.points:
        if p.distance is not None and p.altitude is not None:
            elevation_profile.append([
                round(p.distance * M_TO_MILES, 3),
                round(p.altitude * M_TO_FT, 1),
            ])

    # Build per-point data array, filtered to the video's time range
    points = []
    for i, p in enumerate(ride.points):
        video_secs = _timestamp_to_video_secs(p.timestamp, synced)
        if video_secs is None or video_secs < -1 or video_secs > clip.duration_secs + 1:
            continue
        points.append({
            "t": round(video_secs, 2),
            "speed": round(p.speed * MS_TO_MPH, 1) if p.speed is not None else None,
            "hr": p.heart_rate,
            "power": p.power,
            "cadence": p.cadence,
            "distance": round(p.distance * M_TO_MILES, 2) if p.distance is not None else None,
            "lat": p.lat,
            "lon": p.lon,
            "altitude": round(p.altitude, 1) if p.altitude is not None else None,
            "altitude_ft": round(p.altitude * M_TO_FT) if p.altitude is not None else None,
            "gradient": round(gradients.get(i, 0), 1),
        })

    points.sort(key=lambda p: p["t"])

    return {
        "clip": {
            "filename": clip.path.name,
            "duration_secs": clip.duration_secs,
            "width": clip.width,
            "height": clip.height,
            "fps": clip.fps,
        },
        "total_distance_miles": round(total_distance_miles, 2),
        "route": route,
        "route_bounds": route_bounds,
        "elevation_profile": elevation_profile,
        "points": points,
    }


def _timestamp_to_video_secs(
    timestamp: dt.datetime, synced: SyncedClip
) -> float | None:
    """Convert a FIT timestamp to video-relative seconds."""
    adjusted = timestamp - dt.timedelta(seconds=synced.offset_secs)
    creation = synced.clip.creation_time
    adjusted = normalize_tz(adjusted, creation)

    try:
        return (adjusted - creation).total_seconds()
    except Exception:
        return None
