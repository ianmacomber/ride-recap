"""Strava API integration — fetch activity segment efforts and segment popularity.

OAuth2 flow:
  1. Register an app at https://www.strava.com/settings/api
  2. Set STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET in .env
  3. Run `gopro-garmin strava-auth` to authorize and save tokens
  4. Tokens persist to ~/.strava/tokens.json and auto-refresh

Usage in the pipeline:
  efforts = get_segment_efforts(activity_id)
  # Each effort has: segment_id, name, start_time, elapsed_time, star_count
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import get_settings

try:
    from timezonefinder import TimezoneFinder
    _TZF = TimezoneFinder()
except ImportError:
    _TZF = None

_API_BASE = "https://www.strava.com/api/v3"
_TOKEN_URL = "https://www.strava.com/oauth/token"
_AUTH_URL = "https://www.strava.com/oauth/authorize"
_TOKEN_DIR = Path.home() / ".strava"
_TOKEN_FILE = _TOKEN_DIR / "tokens.json"


@dataclass
class SegmentEffort:
    """A segment effort from a Strava activity."""

    segment_id: int
    name: str
    start_time_secs: float  # seconds from activity start
    elapsed_time_secs: float
    star_count: int
    distance_m: float


# ─── Token management ─────────────────────────────────────────

def _save_tokens(tokens: dict):
    _TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    _TOKEN_FILE.write_text(json.dumps(tokens, indent=2))


def _load_tokens() -> dict | None:
    if _TOKEN_FILE.exists():
        return json.loads(_TOKEN_FILE.read_text())
    return None


def _api_request(path: str, token: str, retries: int = 5) -> dict | list:
    url = f"{_API_BASE}{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries:
                # Strava rate limit: 100 reqs/15min. Wait escalating.
                wait = min(120 * (attempt + 1), 900)
                print(f"    Rate limited, waiting {wait}s (attempt {attempt + 1})...")
                time.sleep(wait)
                continue
            raise


def _refresh_token(tokens: dict) -> dict:
    """Refresh an expired access token."""
    settings = get_settings()
    data = urllib.parse.urlencode({
        "client_id": settings.strava_client_id,
        "client_secret": settings.strava_client_secret,
        "grant_type": "refresh_token",
        "refresh_token": tokens["refresh_token"],
    }).encode()
    req = urllib.request.Request(_TOKEN_URL, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        new_tokens = json.loads(resp.read())
    # Merge — keep refresh_token if not returned
    tokens["access_token"] = new_tokens["access_token"]
    tokens["expires_at"] = new_tokens["expires_at"]
    if "refresh_token" in new_tokens:
        tokens["refresh_token"] = new_tokens["refresh_token"]
    _save_tokens(tokens)
    return tokens


def get_access_token() -> str:
    """Get a valid access token, refreshing if expired."""
    tokens = _load_tokens()
    if tokens is None:
        raise RuntimeError(
            "No Strava tokens found. Run `gopro-garmin strava-auth` first."
        )
    if tokens.get("expires_at", 0) < time.time() + 60:
        tokens = _refresh_token(tokens)
    return tokens["access_token"]


def get_auth_url() -> str:
    """Return the OAuth authorization URL for the user to visit."""
    settings = get_settings()
    params = urllib.parse.urlencode({
        "client_id": settings.strava_client_id,
        "response_type": "code",
        "redirect_uri": "http://localhost:8888/callback",
        "scope": "read,activity:read_all",
        "approval_prompt": "auto",
    })
    return f"{_AUTH_URL}?{params}"


def exchange_code(code: str) -> dict:
    """Exchange an authorization code for tokens."""
    settings = get_settings()
    data = urllib.parse.urlencode({
        "client_id": settings.strava_client_id,
        "client_secret": settings.strava_client_secret,
        "code": code,
        "grant_type": "authorization_code",
    }).encode()
    req = urllib.request.Request(_TOKEN_URL, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        tokens = json.loads(resp.read())
    _save_tokens(tokens)
    return tokens


def authorize_interactive():
    """Run the full OAuth flow with a local callback server."""
    import http.server
    import threading
    import webbrowser

    auth_code = None

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal auth_code
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            auth_code = params.get("code", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h2>Authorized! You can close this tab.</h2>")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 8888), Handler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    url = get_auth_url()
    print("Opening browser for Strava authorization...")
    print(f"  {url}")
    webbrowser.open(url)

    thread.join(timeout=120)
    server.server_close()

    if auth_code is None:
        raise RuntimeError("Authorization timed out or was denied.")

    tokens = exchange_code(auth_code)
    athlete = tokens.get("athlete", {})
    print(f"Authorized as {athlete.get('firstname', '')} {athlete.get('lastname', '')}")
    return tokens


# ─── Activity + segment data ──────────────────────────────────

def list_activities(count: int = 10) -> list[dict]:
    """List recent activities."""
    token = get_access_token()
    return _api_request(f"/athlete/activities?per_page={count}", token)


def get_activity(activity_id: int) -> dict:
    """Get full activity detail including segment efforts."""
    token = get_access_token()
    return _api_request(f"/activities/{activity_id}", token)


def get_segment(segment_id: int) -> dict:
    """Get segment detail including star_count."""
    token = get_access_token()
    return _api_request(f"/segments/{segment_id}", token)


def get_segment_efforts(activity_id: int) -> list[SegmentEffort]:
    """Get segment efforts for an activity using starred status.

    Uses the `starred` boolean from the activity response (whether the
    authenticated athlete starred the segment) as the popularity signal.
    This requires only 1 API call — no per-segment fetches needed.

    For starred segments, optionally enriches with star_count from cache
    or a small number of detail fetches.

    Returns efforts sorted by start time.
    """
    import datetime as dt

    print(f"Fetching Strava activity {activity_id}...")
    activity = get_activity(activity_id)

    efforts_raw = activity.get("segment_efforts", [])
    if not efforts_raw:
        print("  No segment efforts found.")
        return []

    # Parse activity start time for computing effort offsets
    start_str = activity.get("start_date", "")
    if start_str:
        activity_start = dt.datetime.fromisoformat(start_str.replace("Z", "+00:00"))
    else:
        activity_start = None

    # Filter to starred segments only (or all if none starred)
    starred_efforts = [e for e in efforts_raw if e.get("segment", {}).get("starred")]
    if starred_efforts:
        print(f"  {len(efforts_raw)} segment efforts, {len(starred_efforts)} on starred segments")
        source = starred_efforts
    else:
        print(f"  {len(efforts_raw)} segment efforts, none starred — using all")
        source = efforts_raw

    # Fetch star_count for starred segments (small number, usually <20)
    star_cache = _load_star_cache()
    starred_ids = {e["segment"]["id"] for e in source if e.get("segment", {}).get("starred")}
    uncached = [sid for sid in starred_ids if sid not in star_cache]

    if uncached:
        print(f"  Fetching star counts for {len(uncached)} starred segments...")
        for sid in uncached:
            try:
                seg = get_segment(sid)
                star_cache[sid] = seg.get("star_count", 0)
            except Exception:
                star_cache[sid] = 0
        _save_star_cache(star_cache)

    # Build efforts with correct start time offsets
    efforts = []
    for e in source:
        seg = e["segment"]
        sid = seg["id"]
        starred = seg.get("starred", False)

        # Compute start offset from activity start
        start_secs = 0.0
        effort_start_str = e.get("start_date", "")
        if effort_start_str and activity_start:
            effort_start = dt.datetime.fromisoformat(
                effort_start_str.replace("Z", "+00:00"))
            start_secs = (effort_start - activity_start).total_seconds()

        effort = SegmentEffort(
            segment_id=sid,
            name=seg.get("name", f"Segment {sid}"),
            start_time_secs=start_secs,
            elapsed_time_secs=e.get("elapsed_time", 0),
            star_count=star_cache.get(sid, 100 if starred else 0),
            distance_m=seg.get("distance", 0),
        )
        efforts.append(effort)

    efforts.sort(key=lambda e: e.start_time_secs)

    print(f"  {len(efforts)} segments:")
    for eff in efforts:
        h, rem = divmod(int(eff.start_time_secs), 3600)
        m, s = divmod(rem, 60)
        ts = f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        print(f"    {ts:>8}  {eff.star_count:>4} stars  {eff.name[:50]}")

    return efforts


def get_activity_timezone(activity_id: int) -> str | None:
    """Detect an activity timezone from its Strava start coordinates."""
    if _TZF is None:
        return None
    try:
        activity = get_activity(activity_id)
        start_latlng = activity.get("start_latlng")
        if start_latlng and len(start_latlng) >= 2:
            return _TZF.timezone_at(
                lat=float(start_latlng[0]), lng=float(start_latlng[1])
            )
    except Exception as exc:
        print(f"  Warning: Could not detect timezone from activity {activity_id}: {exc}")
    return None


# ─── Star count cache (avoid re-fetching) ─────────────────────

_STAR_CACHE_FILE = _TOKEN_DIR / "star_cache.json"


def _load_star_cache() -> dict[int, int]:
    if _STAR_CACHE_FILE.exists():
        raw = json.loads(_STAR_CACHE_FILE.read_text())
        return {int(k): v for k, v in raw.items()}
    return {}


def _save_star_cache(cache: dict[int, int]):
    _TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    _STAR_CACHE_FILE.write_text(json.dumps(cache, indent=2))
