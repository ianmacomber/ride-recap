"""Derive per-ride lockup strings (origin, destination, road) from GPS.

Three signals come from the FIT track alone:

  1. ``classify_ride`` — local-loop vs out-and-back vs point-to-point,
     based on max distance from start, bbox vs total distance, and how
     far the ride's end is from its start. Local-loop rides report no
     destination; out-and-back rides take their destination from the
     farthest point; point-to-point rides take it from the actual
     ride terminus.
  2. ``reverse_geocode`` — Nominatim lookup for a (lat, lon). Results are
     cached on disk so the smoke-test / re-runs are free.
  3. ``match_road`` — for each named ``highway=*`` way returned by an
     Overpass query over the ride's bbox, count how many GPS points fall
     within MATCH_RADIUS_M of the polyline; pick the highest-coverage name.

The composer reads :func:`compute_route_metadata` and uses its values for
the lockup / recap card. Manual CLI overrides (--origin / --road / --crew)
always win.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .config import get_settings
from .fit_parser import RideData


_CACHE_DIR = Path.home() / ".ride_recap_cache"
_NOMINATIM_DIR = _CACHE_DIR / "nominatim"
_OVERPASS_DIR = _CACHE_DIR / "overpass"
for _d in (_NOMINATIM_DIR, _OVERPASS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _user_agent() -> str:
    """Build the OSM User-Agent.

    Nominatim and Overpass both require a real contact address so they can
    reach the operator of a misbehaving client. Set ``OSM_CONTACT_EMAIL`` in
    ``.env``; :func:`osm_enabled` gates the lookups when it is missing.
    """
    return f"ride-recap/0.1 ({get_settings().osm_contact_email})"


def osm_enabled() -> bool:
    """True when a contact address is configured and OSM may be queried."""
    return bool(get_settings().osm_contact_email.strip())


# A ride whose max-distance-from-start is below this is treated as a
# local loop (park laps, neighborhood ride) with no meaningful destination.
LOCAL_LOOP_MAX_DIST_MI = 5.0

# A ride whose start-to-end distance exceeds this is treated as
# point-to-point (one-way ride that terminates somewhere other than
# the start). Below this, the ride is an out-and-back loop and the
# destination pin should sit at the farthest point reached.
POINT_TO_POINT_END_DIST_MI = 5.0

# Point-to-way match threshold. 25 m is wide enough to swallow GPS jitter
# plus the typical OSM way centerline offset from the actual riding
# lane, but tight enough that adjacent named streets don't both claim
# the same point.
MATCH_RADIUS_M = 25.0


@dataclass
class RouteMetadata:
    origin: str
    destination: str  # "" for local-loop rides
    road: str
    ride_type: str  # "local_loop" | "out_and_back" | "point_to_point" | "unknown"
    distance_mi: float
    max_dist_from_start_mi: float
    end_to_start_mi: float
    bbox_diag_mi: float
    farthest_point: tuple[float, float] | None
    start_point: tuple[float, float] | None
    end_point: tuple[float, float] | None
    destination_point: tuple[float, float] | None  # where the end pin should land
    road_coverage_pct: float  # fraction of points matched to the chosen road


# ─── Geometry ─────────────────────────────────────────────────

def haversine_mi(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _latlon_to_xy(lat: float, lon: float, lat0: float) -> tuple[float, float]:
    """Equirectangular projection in meters, centered on lat0.

    Good to ~0.1% over a city-scale bbox; way faster than haversine for
    bulk distance work. We only use it for point-to-segment matching
    where absolute accuracy doesn't matter — relative ordering does.
    """
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    return (lon * m_per_deg_lon, lat * m_per_deg_lat)


def _point_to_segment_m(px, py, ax, ay, bx, by) -> float:
    """Distance from point P to segment AB, in the same units as inputs."""
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


# ─── Ride classification ──────────────────────────────────────

def classify_ride(ride: RideData) -> dict:
    """Compute distance summaries and decide local_loop / out_and_back / point_to_point."""
    pts = [(p.lat, p.lon) for p in ride.points if p.lat is not None and p.lon is not None]
    if not pts:
        return {
            "ride_type": "unknown",
            "max_dist_from_start_mi": 0.0,
            "end_to_start_mi": 0.0,
            "bbox_diag_mi": 0.0,
            "distance_mi": 0.0,
            "start_point": None,
            "farthest_point": None,
            "end_point": None,
        }

    start_lat, start_lon = pts[0]
    end_lat, end_lon = pts[-1]
    max_d = 0.0
    farthest = pts[0]
    for lat, lon in pts:
        d = haversine_mi(start_lat, start_lon, lat, lon)
        if d > max_d:
            max_d, farthest = d, (lat, lon)

    end_to_start = haversine_mi(start_lat, start_lon, end_lat, end_lon)

    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    bbox_diag = haversine_mi(min(lats), min(lons), max(lats), max(lons))

    # Total distance from FIT, if available
    dists = [p.distance for p in ride.points if p.distance is not None]
    from .utils import M_TO_MILES
    total_mi = (max(dists) * M_TO_MILES) if dists else 0.0

    if max_d < LOCAL_LOOP_MAX_DIST_MI:
        ride_type = "local_loop"
    elif end_to_start >= POINT_TO_POINT_END_DIST_MI:
        # End sits far from start → the ride terminated somewhere else,
        # not a loop back home. The destination pin should be the
        # terminus, not the apex.
        ride_type = "point_to_point"
    else:
        ride_type = "out_and_back"

    return {
        "ride_type": ride_type,
        "max_dist_from_start_mi": max_d,
        "end_to_start_mi": end_to_start,
        "bbox_diag_mi": bbox_diag,
        "distance_mi": total_mi,
        "start_point": (start_lat, start_lon),
        "farthest_point": farthest,
        "end_point": (end_lat, end_lon),
    }


# ─── Reverse geocoding (Nominatim) ────────────────────────────

def _nominatim_cache_key(lat: float, lon: float) -> Path:
    # 4 decimals ≈ 11 m — caches share between nearby points
    key = hashlib.md5(f"{lat:.4f},{lon:.4f}".encode()).hexdigest()
    return _NOMINATIM_DIR / f"{key}.json"


def reverse_geocode(lat: float, lon: float, *, zoom: int = 16) -> dict:
    cf = _nominatim_cache_key(lat, lon)
    if cf.exists():
        return json.loads(cf.read_text())

    r = requests.get(
        "https://nominatim.openstreetmap.org/reverse",
        params={"lat": lat, "lon": lon, "format": "json", "zoom": zoom},
        headers={"User-Agent": _user_agent()},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    cf.write_text(json.dumps(data))
    # Nominatim's usage policy asks for ≤1 req/s
    time.sleep(1.05)
    return data


def _is_admin_pseudonym(value: str) -> bool:
    """True for Nominatim 'neighborhood' values that are actually admin codes.

    NYC's OSM data tags ``neighbourhood = "Manhattan Community Board 7"``
    everywhere in Manhattan, which is useless as a lockup label. Same for
    "Council District N", "Ward N", etc. When we see one of these, fall
    through to the next address field (usually ``quarter``).
    """
    if not value:
        return True
    bad_patterns = ("community board", "council district",
                    "ward ", "election district", "precinct ",
                    "historic district")
    v = value.lower()
    return any(p in v for p in bad_patterns)


def best_locality(geo: dict) -> str:
    """Pick a human-readable label from a Nominatim address dict.

    Prefers fine-grained neighborhood names; falls back upward. Returns
    UPPERCASE for the lockup style. Skips admin pseudonyms like
    "Manhattan Community Board 7" so we fall through to "Upper West Side".
    """
    addr = geo.get("address", {}) or {}
    for key in (
        "quarter", "neighbourhood", "suburb", "city_district",
        "hamlet", "village", "town", "city",
    ):
        v = addr.get(key)
        if v and not _is_admin_pseudonym(v):
            return v.upper()
    # Last resort: first chunk of the display_name
    display = geo.get("display_name", "")
    return display.split(",")[0].strip().upper() if display else "UNKNOWN"


# ─── Road detection (Overpass) ────────────────────────────────

_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def _overpass_cache_key(bbox: tuple[float, float, float, float]) -> Path:
    # bbox = (south, west, north, east), rounded for cache stability
    s = ",".join(f"{v:.4f}" for v in bbox)
    return _OVERPASS_DIR / f"{hashlib.md5(s.encode()).hexdigest()}.json"


def overpass_named_ways(bbox: tuple[float, float, float, float]) -> list[dict]:
    """Return all named highway-tagged ways inside ``bbox``.

    bbox = (south, west, north, east). Cached on disk by rounded-bbox key.
    Each way is ``{"name": str, "highway": str, "geom": [(lat, lon), ...]}``.
    """
    cf = _overpass_cache_key(bbox)
    if cf.exists():
        return json.loads(cf.read_text())

    south, west, north, east = bbox
    # Exclude non-rideable / decorative ways. Keep bike-relevant paths.
    query = f"""
    [out:json][timeout:60];
    (
      way["highway"]["name"]["highway"!~"^(footway|steps|elevator|construction|proposed)$"]
        ({south:.6f},{west:.6f},{north:.6f},{east:.6f});
    );
    out geom tags;
    """
    last_err: Exception | None = None
    for url in _OVERPASS_ENDPOINTS:
        try:
            r = requests.post(url, data={"data": query},
                              headers={"User-Agent": _user_agent()}, timeout=90)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as exc:
            last_err = exc
            continue
    else:
        raise RuntimeError(f"Overpass failed on all endpoints: {last_err}")

    ways = []
    for el in data.get("elements", []):
        if el.get("type") != "way":
            continue
        tags = el.get("tags", {}) or {}
        name = tags.get("name")
        if not name:
            continue
        geom = [(g["lat"], g["lon"]) for g in el.get("geometry", []) or []]
        if len(geom) < 2:
            continue
        ways.append({
            "name": name,
            "highway": tags.get("highway", ""),
            "ref": tags.get("ref", ""),
            "geom": geom,
        })

    cf.write_text(json.dumps(ways))
    return ways


_PLACE_RANK = {  # smaller = preferred when distances are close
    "neighbourhood": 1,
    "quarter": 1,
    "hamlet": 2,
    "village": 2,
    "suburb": 3,
    "town": 3,
    "city": 4,
}


def nearest_place_name(lat: float, lon: float, radius_m: int = 2500) -> str:
    """Return the nearest OSM ``place=*`` feature's name.

    Nominatim's reverse-geocode hides finer-grained neighborhoods in
    cities like NYC where it returns admin pseudonyms like "Manhattan
    Community Board 4" instead of "Hudson Yards". Hitting OSM directly
    via Overpass for ``place=neighbourhood|quarter|hamlet|village`` nodes
    fixes this — Hudson Yards / Chelsea / Hell's Kitchen are all there
    as tagged nodes.

    Falls back to '' if nothing usable is in range.
    """
    cf = _OVERPASS_DIR / f"place_{hashlib.md5(f'{lat:.4f},{lon:.4f},{radius_m}'.encode()).hexdigest()}.json"
    if cf.exists():
        data = json.loads(cf.read_text())
    else:
        q = f"""
        [out:json][timeout:30];
        nwr[place~"^(neighbourhood|quarter|hamlet|village|town|suburb|city)$"][name]
          (around:{radius_m},{lat},{lon});
        out tags center;
        """
        last_err: Exception | None = None
        data = None
        for url in _OVERPASS_ENDPOINTS:
            try:
                r = requests.post(url, data={"data": q},
                                  headers={"User-Agent": _user_agent()}, timeout=60)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as exc:
                last_err = exc
                continue
        if data is None:
            print(f"  nearest_place lookup failed: {last_err}")
            return ""
        cf.write_text(json.dumps(data))

    # Compute (distance, rank) for each element; lowest rank then lowest
    # distance wins. Rank prevents a far "neighbourhood" losing to a near
    # "city" — for the NYC start, "Manhattan" (suburb) is closer in some
    # cases than "Hudson Yards" (neighbourhood), but the user wants the
    # neighborhood.
    best: tuple[int, float, str] | None = None
    for el in data.get("elements", []):
        tags = el.get("tags", {}) or {}
        name = tags.get("name")
        place = tags.get("place")
        if not name or place not in _PLACE_RANK:
            continue
        if _is_admin_pseudonym(name):
            continue  # skips "Frederick Douglass Square Historic District" etc.
        elat = el.get("lat") or (el.get("center") or {}).get("lat")
        elon = el.get("lon") or (el.get("center") or {}).get("lon")
        if elat is None or elon is None:
            continue
        d_m = haversine_mi(lat, lon, elat, elon) * 1609.344
        rank = _PLACE_RANK[place]
        if best is None or (rank, d_m) < (best[0], best[1]):
            best = (rank, d_m, name)

    return best[2] if best else ""


def _strip_locality_prefix(name: str) -> str:
    """Trim "Village of " / "Town of " / "City of " prefixes."""
    if not name:
        return name
    for pre in ("Village of ", "Town of ", "City of ", "Township of "):
        if name.startswith(pre):
            return name[len(pre):]
    return name


def overpass_named_parks(bbox: tuple[float, float, float, float]) -> list[dict]:
    """Return named ``leisure=park`` ways inside ``bbox``, with polygon geometry.

    We only fetch the way representation (relations are skipped). Central
    Park, Prospect Park, the Boston Common, and Charles River Esplanade
    all have way-form polygons that are sufficient for point-in-polygon
    coverage tests.
    """
    cf = _OVERPASS_DIR / f"parks_{hashlib.md5(str(bbox).encode()).hexdigest()}.json"
    if cf.exists():
        return json.loads(cf.read_text())

    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:60];
    (
      way["leisure"="park"]["name"]({south:.6f},{west:.6f},{north:.6f},{east:.6f});
    );
    out geom tags;
    """
    last_err: Exception | None = None
    for url in _OVERPASS_ENDPOINTS:
        try:
            r = requests.post(url, data={"data": query},
                              headers={"User-Agent": _user_agent()}, timeout=90)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as exc:
            last_err = exc
            continue
    else:
        raise RuntimeError(f"Overpass parks failed: {last_err}")

    parks = []
    for el in data.get("elements", []):
        if el.get("type") != "way":
            continue
        tags = el.get("tags", {}) or {}
        name = tags.get("name")
        geom = [(g["lat"], g["lon"]) for g in el.get("geometry", []) or []]
        if not name or len(geom) < 4:
            continue
        # Only closed-loop polygons
        if geom[0] != geom[-1]:
            continue
        parks.append({"name": name, "geom": geom})

    cf.write_text(json.dumps(parks))
    return parks


def _point_in_polygon(lat: float, lon: float, poly: list[tuple[float, float]]) -> bool:
    """Ray-casting test in raw lat/lon space (works fine at city scale)."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        yi, xi = poly[i]
        yj, xj = poly[j]
        if ((yi > lat) != (yj > lat)) and \
           (lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def park_coverage(ride: RideData, parks: list[dict]) -> tuple[str, float]:
    """Find the park containing the most GPS points. Returns (name, fraction)."""
    pts = [(p.lat, p.lon) for p in ride.points if p.lat is not None and p.lon is not None]
    if not pts or not parks:
        return "", 0.0

    # Cheap bbox prefilter per park
    counts: dict[str, int] = {}
    for park in parks:
        ys = [g[0] for g in park["geom"]]
        xs = [g[1] for g in park["geom"]]
        ymin, ymax = min(ys), max(ys)
        xmin, xmax = min(xs), max(xs)
        n = 0
        for lat, lon in pts:
            if lat < ymin or lat > ymax or lon < xmin or lon > xmax:
                continue
            if _point_in_polygon(lat, lon, park["geom"]):
                n += 1
        if n > 0:
            counts[park["name"]] = counts.get(park["name"], 0) + n

    if not counts:
        return "", 0.0
    top = max(counts, key=counts.get)
    return top, counts[top] / len(pts)


def match_road(ride: RideData, ways: list[dict]) -> tuple[str, float, dict]:
    """Attribute each GPS point to its nearest named way (within MATCH_RADIUS_M).

    Returns (top_name, coverage_pct, per_way_count). ``top_name`` is the
    most-attributed name; coverage is the fraction of total ride points
    that got matched to *some* way. Per-way counts are returned for
    debugging / display.
    """
    pts = [(p.lat, p.lon) for p in ride.points if p.lat is not None and p.lon is not None]
    if not pts or not ways:
        return "", 0.0, {}

    lat0 = sum(p[0] for p in pts) / len(pts)

    # Project all points + way vertices into a flat meter plane
    px = [_latlon_to_xy(lat, lon, lat0) for lat, lon in pts]

    way_xy: list[tuple[dict, list[tuple[float, float]], tuple[float, float, float, float]]] = []
    for w in ways:
        coords = [_latlon_to_xy(lat, lon, lat0) for lat, lon in w["geom"]]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        bbox = (min(xs) - MATCH_RADIUS_M, min(ys) - MATCH_RADIUS_M,
                max(xs) + MATCH_RADIUS_M, max(ys) + MATCH_RADIUS_M)
        way_xy.append((w, coords, bbox))

    name_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    matched = 0
    for x, y in px:
        best_d = MATCH_RADIUS_M
        best_way: dict | None = None
        for w, coords, (bx0, by0, bx1, by1) in way_xy:
            # Cheap bbox reject
            if x < bx0 or x > bx1 or y < by0 or y > by1:
                continue
            for i in range(len(coords) - 1):
                ax, ay = coords[i]
                bx, by = coords[i + 1]
                d = _point_to_segment_m(x, y, ax, ay, bx, by)
                if d < best_d:
                    best_d = d
                    best_way = w
                    if d < 3.0:  # very close — stop early
                        break
            if best_d < 3.0:
                break
        if best_way is not None:
            name_counts[best_way["name"]] = name_counts.get(best_way["name"], 0) + 1
            ref = best_way.get("ref") or ""
            if ref:
                # Multi-ref ways tag like "NJ 63;CR 501" — count each component
                for r in ref.split(";"):
                    r = r.strip()
                    if r:
                        ref_counts[r] = ref_counts.get(r, 0) + 1
            matched += 1

    if not name_counts:
        return "", 0.0, {}

    coverage = matched / len(px)

    # Prefer a route ref (e.g. "US 9W") when its hit count beats the
    # dominant single road name. Long rides span many named ways but
    # only a few route corridors — aggregating by ref captures the
    # rider's mental model ("a 9W ride") better than the single
    # longest road segment ("Riverside Drive").
    top_name = max(name_counts, key=name_counts.get)
    if ref_counts:
        top_ref = max(ref_counts, key=ref_counts.get)
        if (ref_counts[top_ref] >= 100
                and ref_counts[top_ref] >= name_counts[top_name]):
            return (_clean_route_ref(top_ref), coverage,
                    {**name_counts, "_ref:" + top_ref: ref_counts[top_ref]})

    return top_name, coverage, name_counts


def _clean_route_ref(ref: str) -> str:
    """Strip country/state prefix from a route ref. 'US 9W' → '9W', 'I 87' → 'I 87'.

    Local users say "9W" not "US 9W"; "I 87" stays as "I 87" because the
    interstate prefix is the colloquial form. Heuristic: strip country
    (US) and 2-letter state codes; keep highway-class prefixes like 'I'.
    """
    parts = ref.split()
    if len(parts) == 2:
        head, tail = parts
        if head in ("US",) or (len(head) == 2 and head.isupper() and head != "I"):
            return tail
    return ref


# ─── Orchestrator ─────────────────────────────────────────────

def _bbox_with_pad(pts: list[tuple[float, float]], pad_mi: float = 0.25):
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    # 1 deg lat ≈ 69 mi; 1 deg lon ≈ 69 * cos(lat) mi
    lat_pad = pad_mi / 69.0
    lat_mid = (min(lats) + max(lats)) / 2
    lon_pad = pad_mi / (69.0 * max(0.01, math.cos(math.radians(lat_mid))))
    return (min(lats) - lat_pad, min(lons) - lon_pad,
            max(lats) + lat_pad, max(lons) + lon_pad)


def compute_route_metadata(
    ride: RideData,
    *,
    skip_road: bool = False,
    skip_geocode: bool = False,
) -> RouteMetadata:
    """End-to-end: classify ride → geocode origin/destination → match road.

    Both OSM lookups are skipped when no contact address is configured; the
    ride still classifies from GPS alone and the lockup strings fall back to
    CLI overrides or design-token defaults.
    """
    if not osm_enabled():
        print("  OSM_CONTACT_EMAIL unset — skipping OSM lookups "
              "(set it in .env to auto-derive origin/destination/road)")
        skip_road = True
        skip_geocode = True

    cls = classify_ride(ride)
    pts = [(p.lat, p.lon) for p in ride.points if p.lat is not None and p.lon is not None]

    origin = "UNKNOWN"
    destination = ""
    park_name = ""
    park_pct = 0.0

    # Park-polygon test runs first for local-loops: if the ride spends
    # most of its time inside a named park, the park's name beats any
    # Nominatim neighborhood label.
    if pts and not skip_road and cls["ride_type"] == "local_loop":
        try:
            bbox = _bbox_with_pad(pts, pad_mi=0.25)
            parks = overpass_named_parks(bbox)
            park_name, park_pct = park_coverage(ride, parks)
        except Exception as exc:
            park_name = ""
            print(f"  park lookup failed: {exc}")

    if pts and not skip_geocode:
        if park_name and park_pct >= 0.5:
            origin = park_name.upper()
        else:
            # Prefer OSM place=* nearest-feature (catches Hudson Yards
            # which Nominatim hides behind "Manhattan Community Board 4");
            # fall back to Nominatim reverse-geocode if the OSM lookup
            # comes up empty.
            slat, slon = cls["start_point"]
            origin = nearest_place_name(slat, slon)
            if not origin:
                try:
                    geo = reverse_geocode(slat, slon)
                    origin = best_locality(geo)
                except Exception as exc:
                    origin = f"ERR: {exc.__class__.__name__}"
            origin = _strip_locality_prefix(origin).upper()

        if cls["ride_type"] in ("out_and_back", "point_to_point"):
            # out_and_back: the farthest-from-start point names the ride.
            # point_to_point: the actual ride terminus names the ride.
            if cls["ride_type"] == "point_to_point":
                dlat, dlon = cls["end_point"]
            else:
                dlat, dlon = cls["farthest_point"]
            destination = nearest_place_name(dlat, dlon)
            if not destination:
                try:
                    dest_geo = reverse_geocode(dlat, dlon)
                    destination = best_locality(dest_geo)
                except Exception as exc:
                    destination = f"ERR: {exc.__class__.__name__}"
            destination = _strip_locality_prefix(destination).upper()

    road = ""
    coverage = 0.0
    if pts and not skip_road:
        try:
            bbox = _bbox_with_pad(pts, pad_mi=0.25)
            ways = overpass_named_ways(bbox)
            road_name, coverage, _ = match_road(ride, ways)
            road = road_name.upper() if road_name else ""
        except Exception as exc:
            road = f"ERR: {exc.__class__.__name__}"

    if cls["ride_type"] == "point_to_point":
        destination_point = cls["end_point"]
    elif cls["ride_type"] == "out_and_back":
        destination_point = cls["farthest_point"]
    else:
        destination_point = None

    return RouteMetadata(
        origin=origin,
        destination=destination,
        road=road,
        ride_type=cls["ride_type"],
        distance_mi=cls["distance_mi"],
        max_dist_from_start_mi=cls["max_dist_from_start_mi"],
        end_to_start_mi=cls["end_to_start_mi"],
        bbox_diag_mi=cls["bbox_diag_mi"],
        farthest_point=cls["farthest_point"],
        start_point=cls["start_point"],
        end_point=cls["end_point"],
        destination_point=destination_point,
        road_coverage_pct=coverage,
    )
