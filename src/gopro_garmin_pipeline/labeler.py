"""Ride labeler — scrub through an entire ride's video + telemetry and flag moments of interest.

Prerequisites:
    gopro-garmin extract-frames <fit> <video-dir>

Launch:
    .venv/bin/python -m streamlit run src/gopro_garmin_pipeline/labeler.py -- \
        --fit data/raw/2026-04-11/ride.fit \
        --video-dir data/raw/2026-04-11 \
        --offset <seconds>

Note: --offset defaults to 0 and is NOT auto-synced here (unlike `process`),
so pass the real GoPro↔FIT offset or the telemetry will not line up with the
frames you are labeling.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import streamlit as st

try:
    from .fit_parser import parse_fit
    from .gopro_meta import extract_all
    from .sync import normalize_tz
    from .utils import (
        LABEL_SCALE_VERSION,
        M_TO_FT,
        MS_TO_MPH,
        compute_gradient,
        format_time,
        normalize_label_scale,
    )
except ImportError:
    from gopro_garmin_pipeline.fit_parser import parse_fit
    from gopro_garmin_pipeline.gopro_meta import extract_all
    from gopro_garmin_pipeline.sync import normalize_tz
    from gopro_garmin_pipeline.utils import (
        LABEL_SCALE_VERSION,
        M_TO_FT,
        MS_TO_MPH,
        compute_gradient,
        format_time,
        normalize_label_scale,
    )

st.set_page_config(page_title="Ride Labeler", layout="wide")

# The frame server is bound to an OS-assigned ephemeral port and tracked
# per (process, directory). A previous fixed-port design silently shared
# a server across labeler sessions: a second instance would catch the
# "address in use" error, assume "already running", and serve frames
# from the *first* instance's ride. Per-directory binding avoids that.
_frame_server_ports: dict[str, int] = {}


def _start_frame_server(directory: str) -> int:
    """Start (or reuse) a background HTTP server for this frame directory.

    Returns the port the server is listening on. Subsequent calls for
    the same directory reuse the existing server.
    """
    if directory in _frame_server_ports:
        return _frame_server_ports[directory]

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _frame_server_ports[directory] = port
    return port

_LABELS_FILE = "ride_labels.json"
_SCRUBBER_STATE_KEY = "ride_scrubber_event"
_SCRUBBER_COMMIT_KEY = "ride_scrubber_last_commit"


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", required=True, help="Path to .fit file")
    parser.add_argument("--video-dir", required=True, help="Directory of GoPro .MP4 files")
    parser.add_argument("--frames-dir", default=None, help="Directory of pre-extracted frames (default: <video-dir>/frames)")
    parser.add_argument("--offset", type=float, default=0.0, help="Sync offset (seconds)")
    parser.add_argument("--labels-file", default=None, help="Path to labels JSON file")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Parsing FIT file...")
def load_ride(fit_path: str) -> dict:
    ride = parse_fit(fit_path)
    points = []
    for i, p in enumerate(ride.points):
        gradient = 0.0
        if i > 0:
            p0 = ride.points[i - 1]
            if (p0.altitude is not None and p.altitude is not None
                    and p0.distance is not None and p.distance is not None):
                gradient = compute_gradient(p0.altitude, p0.distance, p.altitude, p.distance)
        points.append({
            "elapsed_secs": (p.timestamp - ride.start_time).total_seconds() if ride.start_time else 0,
            "power": p.power,
            "hr": p.heart_rate,
            "speed_mph": round(p.speed * MS_TO_MPH, 1) if p.speed else None,
            "cadence": p.cadence,
            "altitude_ft": round(p.altitude * M_TO_FT, 1) if p.altitude is not None else None,
            "gradient": round(gradient, 1),
            "lat": p.lat,
            "lon": p.lon,
        })
    return {
        "start_time": ride.start_time.isoformat() if ride.start_time else None,
        "end_time": ride.end_time.isoformat() if ride.end_time else None,
        "duration_secs": (ride.end_time - ride.start_time).total_seconds() if ride.start_time and ride.end_time else 0,
        "points": points,
    }


@st.cache_data(show_spinner="Scanning GoPro videos...")
def load_clips(video_dir: str) -> list[dict]:
    clips = extract_all(video_dir)
    return [
        {
            "path": str(c.path), "name": c.path.name,
            "creation_time": c.creation_time.isoformat(),
            "duration_secs": c.duration_secs,
        }
        for c in clips
    ]


@st.cache_data(show_spinner="Loading frames index...")
def load_frames_index(frames_dir: str) -> dict:
    """Load the manifest and build a ride_secs -> file_path map."""
    fdir = Path(frames_dir)
    manifest_path = fdir / "manifest.json"
    if not manifest_path.exists():
        return {"interval": 5, "frames": {}, "clips": []}

    manifest = json.loads(manifest_path.read_text())
    frame_map = {}
    for f in manifest.get("frames", []):
        frame_map[f["ride_secs"]] = str(fdir / f["file"])
    return {
        "interval": manifest.get("interval", 5),
        "frames": frame_map,
        "clips": manifest.get("clips", []),
    }


# ---------------------------------------------------------------------------
# Labels persistence
# ---------------------------------------------------------------------------

def _labels_path(labels_file: str | None, video_dir: str) -> Path:
    if labels_file:
        return Path(labels_file)
    return Path(video_dir) / _LABELS_FILE


def load_labels(labels_file: str | None, video_dir: str) -> list[dict]:
    p = _labels_path(labels_file, video_dir)
    if p.exists():
        return [normalize_label_scale(lab) for lab in json.loads(p.read_text())]
    return []


def save_labels(labels: list[dict], labels_file: str | None, video_dir: str):
    p = _labels_path(labels_file, video_dir)
    p.write_text(json.dumps(labels, indent=2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clips_on_timeline(ride_data: dict, clips_data: list[dict], offset: float) -> list[dict]:
    if not ride_data["start_time"]:
        return []
    ride_start = dt.datetime.fromisoformat(ride_data["start_time"])
    ride_duration = ride_data["duration_secs"]
    mapped = []
    for clip in clips_data:
        clip_start_wall = dt.datetime.fromisoformat(clip["creation_time"])
        clip_start_wall = normalize_tz(clip_start_wall, ride_start)
        elapsed_start = (clip_start_wall - ride_start).total_seconds() + offset
        elapsed_end = elapsed_start + clip["duration_secs"]
        if elapsed_end < 0 or elapsed_start > ride_duration:
            continue
        mapped.append({
            **clip,
            "ride_start_secs": max(0, elapsed_start),
            "ride_end_secs": min(ride_duration, elapsed_end),
            "clip_offset_into_ride": elapsed_start,
        })
    return mapped


def _find_nearest_frame(frames_index: dict, ride_secs: int) -> str | None:
    """Find the closest pre-extracted frame file path."""
    frame_map = frames_index["frames"]
    if not frame_map:
        return None
    keys = sorted(frame_map.keys())
    best = min(keys, key=lambda k: abs(k - ride_secs))
    interval = frames_index["interval"]
    if abs(best - ride_secs) <= interval + 1:
        return frame_map[best]
    return None


def _find_current_point(points: list[dict], current_secs: float) -> dict | None:
    if not points:
        return None
    best_idx = 0
    best_diff = abs(points[0]["elapsed_secs"] - current_secs)
    for i, p in enumerate(points):
        diff = abs(p["elapsed_secs"] - current_secs)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return points[best_idx]


def _sync_scrubber_commit(frame_times: list[int]) -> bool:
    """Consume a fresh scrubber commit from session state, if one exists."""
    payload = st.session_state.get(_SCRUBBER_STATE_KEY)
    if not isinstance(payload, dict):
        return False

    commit_id = payload.get("commitId")
    ride_secs = payload.get("rideSecs")
    if commit_id is None or ride_secs is None:
        return False

    if st.session_state.get(_SCRUBBER_COMMIT_KEY) == commit_id:
        return False

    nearest = min(frame_times, key=lambda t: abs(t - float(ride_secs)))
    st.session_state.current_ride_secs = nearest
    st.session_state[_SCRUBBER_COMMIT_KEY] = commit_id
    return True


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    fit_path = args.fit
    video_dir = args.video_dir
    frames_dir = args.frames_dir or str(Path(video_dir) / "frames")
    offset = args.offset
    labels_file = args.labels_file

    ride_data = load_ride(fit_path)
    clips_data = load_clips(video_dir)
    clips_on_tl = _clips_on_timeline(ride_data, clips_data, offset)
    frames_index = load_frames_index(frames_dir)
    frame_server_port = _start_frame_server(frames_dir)

    ride_duration = ride_data["duration_secs"]
    points = ride_data["points"]

    if not frames_index["frames"]:
        st.error(
            f"No frames found in `{frames_dir}/`. Run this first:\n\n"
            f"```\ngopro-garmin extract-frames {fit_path} {video_dir}\n```"
        )
        return

    # Reduce Streamlit's default padding
    st.markdown("""<style>
        .block-container { padding-top: 1rem; padding-bottom: 0; }
        h1 { margin-bottom: 0 !important; font-size: 1.5rem !important; }
        .stCaption { margin-bottom: 0 !important; }
    </style>""", unsafe_allow_html=True)

    st.title("Ride Labeler")
    st.caption(
        f"{len(clips_on_tl)} clips · "
        f"{ride_duration / 60:.0f} min ride · "
        f"{len(frames_index['frames'])} frames"
    )

    if "labels" not in st.session_state:
        st.session_state.labels = load_labels(labels_file, video_dir)

    # Build sorted list of frame timestamps (only where we have video)
    frame_times = sorted(frames_index["frames"].keys())
    if not frame_times:
        st.error("No frames available")
        return

    # ==================================================================
    # ROW 1: Scrubber (declare_component) + Map
    # Uses Streamlit's component protocol for reliable JS↔Python sync.
    # Live frame preview during drag, value sent on release.
    # ==================================================================
    # Current position from session state
    if "current_ride_secs" not in st.session_state:
        st.session_state.current_ride_secs = frame_times[0]
    _sync_scrubber_commit(frame_times)
    current_secs = st.session_state.current_ride_secs

    # Build data for scrubber component
    frame_urls = [
        f"http://127.0.0.1:{frame_server_port}/{Path(frames_index['frames'][t]).name}"
        for t in frame_times
    ]
    time_labels = [format_time(t) for t in frame_times]
    clip_names_for_frames = []
    for t in frame_times:
        name = ""
        for clip in clips_on_tl:
            if clip["ride_start_secs"] <= t <= clip["ride_end_secs"]:
                name = clip["name"]
                break
        clip_names_for_frames.append(name)

    initial_idx = 0
    for i, t in enumerate(frame_times):
        if t >= current_secs:
            initial_idx = i
            break

    col_scrubber, col_map = st.columns([2, 1])

    with col_scrubber:
        from gopro_garmin_pipeline.components.ride_scrubber import ride_scrubber

        ride_scrubber(
            frame_urls=frame_urls,
            time_labels=time_labels,
            ride_secs=list(frame_times),
            clip_names=clip_names_for_frames,
            initial_idx=initial_idx,
            key=_SCRUBBER_STATE_KEY,
        )

        # If the widget state becomes visible only after instantiation,
        # immediately rerun once so the component re-renders at the same
        # position as the map/timeline on this interaction.
        if _sync_scrubber_commit(frame_times):
            st.rerun()

        current_secs = st.session_state.current_ride_secs

        # Navigation buttons
        current_idx = frame_times.index(current_secs) if current_secs in frame_times else 0
        btn_cols = st.columns([1, 1, 6])
        with btn_cols[0]:
            if st.button("\u25c0 Back", use_container_width=True, disabled=(current_idx == 0)):
                st.session_state.current_ride_secs = frame_times[current_idx - 1]
                st.rerun()
        with btn_cols[1]:
            if st.button("Forward \u25b6", use_container_width=True, disabled=(current_idx >= len(frame_times) - 1)):
                st.session_state.current_ride_secs = frame_times[current_idx + 1]
                st.rerun()

    # Derive state from current position
    active_clip = None
    video_seek_secs = 0.0
    for clip in clips_on_tl:
        if clip["ride_start_secs"] <= current_secs <= clip["ride_end_secs"]:
            active_clip = clip
            video_seek_secs = current_secs - clip["clip_offset_into_ride"]
            break
    current_point = _find_current_point(points, current_secs)

    with col_map:
        _render_gps_map(ride_data, current_point, current_secs)

    with st.expander("Scoring guide", expanded=False):
        st.markdown("""
**Visual (1-10):** How does this frame *look*?
- **10** Iconic — bridge silhouette at golden hour, stunning composition you'd lead a reel with
- **8-9** Strong — dramatic light, water reflections, distinctive landmark, dappled shade
- **6-7** Above filler — pleasant scene with one redeeming element (sky, foliage, framing)
- **4-5** Usable filler — decent scene, nothing memorable
- **2-3** Unremarkable — flat light, generic road, parked cars
- **1** Dead — stopped, pavement only, foot down, lens artifact

**Action (1-10):** How much is *happening*?
- **10** Edit-leading — close call, near-miss, sprint with visible bounce, wheel-to-wheel attack
- **8-9** Strong — tight paceline, fast descent with lean, big power surge, visible effort
- **6-7** Above filler — moderate group riding with movement, climbing tempo, some traffic
- **4-5** Usable filler — steady solo cruising, light pedaling
- **2-3** Unremarkable — coasting, slow rolling
- **1** Dead — stopped at light, soft-pedaling, foot down
""")

    editing_idx = st.session_state.get("editing_idx")
    if editing_idx is not None and 0 <= editing_idx < len(st.session_state.labels):
        edit_t = st.session_state.labels[editing_idx].get("ride_time_str", "?")
        ec1, ec2 = st.columns([6, 1])
        ec1.caption(f"✏️  Editing label at **{edit_t}** — Save to update, Cancel to discard changes")
        if ec2.button("Cancel", key="cancel_edit", use_container_width=True):
            for k in ("label_type", "label_visual", "label_action",
                     "label_notes", "label_must_include"):
                st.session_state.pop(k, None)
            del st.session_state.editing_idx
            st.rerun()

    col_type, col_visual, col_action, col_must, col_notes, col_save = st.columns(
        [2, 1, 1, 0.9, 2.6, 1]
    )

    with col_type:
        label_type = st.selectbox("Type", [
            "scenery",
            "action",
            "urban",
            "climb",
            "descent",
            "group_riding",
            "incident",
            "landmark",
            "transition",
        ], key="label_type", label_visibility="collapsed")

    with col_visual:
        visual = st.slider("Visual", 1, 10, 5, key="label_visual", help="Light, composition, landmarks. 10=lead the edit, 5=filler, 1=dead.")

    with col_action:
        action = st.slider("Action", 1, 10, 5, key="label_action", help="Effort, proximity, speed, events. 10=lead the edit, 5=filler, 1=dead.")

    with col_must:
        must_include = st.checkbox(
            "Must include",
            key="label_must_include",
            help="Force this moment into the highlight cut, regardless of score.",
        )

    with col_notes:
        notes = st.text_input("Why?", placeholder="Describe why this moment is interesting", key="label_notes", label_visibility="collapsed")

    with col_save:
        save_label = "Update" if editing_idx is not None else "Save"
        if st.button(save_label, type="primary", use_container_width=True):
            label = {
                "ride_time_secs": round(current_secs, 1),
                "ride_time_str": format_time(current_secs),
                "type": label_type,
                "visual": visual,
                "action": action,
                "scale_version": LABEL_SCALE_VERSION,
                "must_include": bool(must_include),
                "notes": notes,
                "clip_name": active_clip["name"] if active_clip else None,
                "video_secs": round(video_seek_secs, 1) if active_clip else None,
            }
            if editing_idx is not None and 0 <= editing_idx < len(st.session_state.labels):
                st.session_state.labels[editing_idx] = label
                save_labels(st.session_state.labels, labels_file, video_dir)
                st.toast(f"Updated label at {label['ride_time_str']}")
                del st.session_state.editing_idx
            else:
                st.session_state.labels.append(label)
                save_labels(st.session_state.labels, labels_file, video_dir)
                st.toast(f"Saved label at {label['ride_time_str']}")
            st.rerun()

    # ==================================================================
    # ROW 3: Timeline charts (aligned with scrubber column)
    # ==================================================================
    tl_col, _tl_spacer = st.columns([2, 1])
    with tl_col:
        _render_timeline(ride_data, clips_on_tl, st.session_state.labels, ride_duration, current_secs)

    # ==================================================================
    # SECTION 4: Saved labels
    # ==================================================================
    if st.session_state.labels:
        st.divider()
        st.subheader(f"Labels ({len(st.session_state.labels)})")
        # Header row
        hdr = st.columns([1, 1.5, 1.6, 4, 0.7, 0.7, 0.6])
        hdr[0].caption("Time")
        hdr[1].caption("Type")
        hdr[2].caption("Score · flags")
        hdr[3].caption("Notes")
        for i, label in enumerate(st.session_state.labels):
            cols = st.columns([1, 1.5, 1.6, 4, 0.7, 0.7, 0.6])
            cols[0].write(f"**{label['ride_time_str']}**")
            cols[1].write(label.get("type", ""))
            v = label.get("visual", 0)
            a = label.get("action", 0)
            flags = " ⭐" if label.get("must_include") else ""
            cols[2].write(f"v={v} · a={a}{flags}")
            cols[3].write(label.get("notes", ""))
            if cols[4].button("Edit", key=f"edit_{i}"):
                st.session_state.label_type = label.get("type", "scenery")
                st.session_state.label_visual = int(label.get("visual", 5))
                st.session_state.label_action = int(label.get("action", 5))
                st.session_state.label_notes = label.get("notes", "") or ""
                st.session_state.label_must_include = bool(label.get("must_include", False))
                st.session_state.editing_idx = i
                target = int(label["ride_time_secs"])
                nearest = min(frame_times, key=lambda t: abs(t - target))
                st.session_state.current_ride_secs = nearest
                st.rerun()
            if cols[5].button("Go", key=f"goto_{i}"):
                target = int(label["ride_time_secs"])
                nearest = min(frame_times, key=lambda t: abs(t - target))
                st.session_state.current_ride_secs = nearest
                st.rerun()
            if cols[6].button("X", key=f"del_{i}"):
                st.session_state.labels.pop(i)
                save_labels(st.session_state.labels, labels_file, video_dir)
                # Repair editing_idx if affected
                editing = st.session_state.get("editing_idx")
                if editing is not None:
                    if editing == i:
                        del st.session_state.editing_idx
                    elif editing > i:
                        st.session_state.editing_idx = editing - 1
                st.rerun()


# ---------------------------------------------------------------------------
# GPS Map component
# ---------------------------------------------------------------------------

def _render_gps_map(ride_data: dict, current_point: dict | None, current_secs: float):
    points = ride_data["points"]
    coords = [{"lat": p["lat"], "lon": p["lon"], "t": p["elapsed_secs"]}
              for p in points if p["lat"] is not None and p["lon"] is not None]
    if not coords:
        st.info("No GPS data")
        return

    cur_lat = current_point["lat"] if current_point and current_point["lat"] else coords[0]["lat"]
    cur_lon = current_point["lon"] if current_point and current_point["lon"] else coords[0]["lon"]

    lats = [c["lat"] for c in coords]
    lons = [c["lon"] for c in coords]
    pad_lat = (max(lats) - min(lats)) * 0.08 or 0.001
    pad_lon = (max(lons) - min(lons)) * 0.08 or 0.001
    min_lat, max_lat = min(lats) - pad_lat, max(lats) + pad_lat
    min_lon, max_lon = min(lons) - pad_lon, max(lons) + pad_lon

    step = max(1, len(coords) // 300)
    route_sampled = coords[::step]
    coords_json = json.dumps(route_sampled)

    map_html = f"""
<style>
  html, body {{ margin: 0; padding: 0; background: transparent; }}
</style>
<div style="background:#1a1a2e;border-radius:8px;padding:8px;">
  <canvas id="gps-map" style="width:100%;height:240px;display:block;border-radius:6px;"></canvas>
</div>
<script>
(function() {{
  var coords = {coords_json};
  var curLat = {cur_lat}, curLon = {cur_lon}, curT = {current_secs};
  var minLat = {min_lat}, maxLat = {max_lat}, minLon = {min_lon}, maxLon = {max_lon};
  var canvas = document.getElementById("gps-map");
  var dpr = window.devicePixelRatio || 1;
  var w = canvas.parentElement.offsetWidth - 16, h = 240;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.width = w + "px"; canvas.style.height = h + "px";
  var ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0, 0, w, h);

  function proj(lat, lon) {{
    return [((lon - minLon) / (maxLon - minLon)) * (w - 20) + 10,
            (1 - (lat - minLat) / (maxLat - minLat)) * (h - 20) + 10];
  }}

  // Full route (dim)
  ctx.beginPath(); ctx.strokeStyle = "rgba(255,255,255,0.15)"; ctx.lineWidth = 2;
  for (var i = 0; i < coords.length; i++) {{
    var p = proj(coords[i].lat, coords[i].lon);
    i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]);
  }}
  ctx.stroke();

  // Traveled (bright)
  ctx.beginPath(); ctx.strokeStyle = "rgba(0,184,148,0.7)"; ctx.lineWidth = 2.5;
  for (var i = 0; i < coords.length; i++) {{
    if (coords[i].t > curT) break;
    var p = proj(coords[i].lat, coords[i].lon);
    i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]);
  }}
  ctx.stroke();

  // Current dot
  var cp = proj(curLat, curLon);
  ctx.beginPath(); ctx.arc(cp[0], cp[1], 6, 0, Math.PI * 2);
  ctx.fillStyle = "#ff6b35"; ctx.fill();
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke();
  ctx.beginPath(); ctx.arc(cp[0], cp[1], 10, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(255,107,53,0.4)"; ctx.lineWidth = 2; ctx.stroke();
}})();
</script>
"""
    st.components.v1.html(map_html, height=260)


# ---------------------------------------------------------------------------
# Timeline HTML component
# ---------------------------------------------------------------------------

def _render_timeline(ride_data: dict, clips_on_tl: list[dict], labels: list[dict], ride_duration: float, current_secs: float = 0):
    points = ride_data["points"]
    step = max(1, len(points) // 1000)
    sampled = points[::step]

    timeline_data = json.dumps({
        "duration": ride_duration,
        "points": [
            {"t": p["elapsed_secs"], "power": p["power"], "hr": p["hr"],
             "speed": p["speed_mph"], "cadence": p["cadence"], "altitude_ft": p["altitude_ft"]}
            for p in sampled
        ],
        "clips": [
            {"name": c["name"], "start": c["ride_start_secs"], "end": c["ride_end_secs"]}
            for c in clips_on_tl
        ],
        "labels": [
            {"t": lb["ride_time_secs"], "type": lb["type"]}
            for lb in labels
        ],
    })

    html = f"""
<style>
  html, body {{ margin: 0; padding: 0; background: transparent; }}
</style>
<div id="tl-root" style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#1a1a2e;border-radius:8px;padding:12px;user-select:none;-webkit-user-select:none;">
  <div style="display:flex;gap:12px;margin-bottom:6px;font-size:11px;color:#aaa;">
    <span><span style="color:#ff6b35;">■</span> Power</span>
    <span><span style="color:#e84393;">■</span> HR</span>
    <span><span style="color:#00b894;">■</span> Speed</span>
    <span><span style="color:#fdcb6e;">■</span> Cadence</span>
    <span><span style="color:#74b9ff;">■</span> Altitude</span>
  </div>
  <canvas id="chart-power" style="width:100%;height:80px;display:block;cursor:crosshair;"></canvas>
  <canvas id="chart-speed" style="width:100%;height:45px;display:block;cursor:crosshair;margin-top:1px;"></canvas>
  <canvas id="chart-alt" style="width:100%;height:30px;display:block;cursor:crosshair;margin-top:1px;"></canvas>
  <canvas id="chart-clips" style="width:100%;height:28px;display:block;cursor:pointer;margin-top:1px;"></canvas>
  <canvas id="chart-time" style="width:100%;height:18px;display:block;margin-top:1px;"></canvas>
  <div id="readout" style="text-align:center;font-size:12px;color:#ccc;margin-top:4px;"></div>
</div>
<script>
(function() {{
  var D = {timeline_data};
  var duration = D.duration, pts = D.points, clips = D.clips, labels = D.labels;
  var needlePct = {current_secs} / duration, dragging = false;
  var PC="#ff6b35",HC="#e84393",SC="#00b894",CC="#fdcb6e",AC="#74b9ff",AF="rgba(116,185,255,0.15)",NC="#ffffff",LC="#f5c518";
  var ZC=["rgba(128,128,128,0.08)","rgba(0,136,254,0.08)","rgba(0,196,0,0.08)","rgba(255,204,0,0.08)","rgba(255,102,0,0.08)","rgba(255,0,0,0.08)"];

  function setup(id,h){{var c=document.getElementById(id);var d=window.devicePixelRatio||1;var w=c.parentElement.offsetWidth-24;c.width=w*d;c.height=h*d;c.style.width=w+"px";c.style.height=h+"px";var x=c.getContext("2d");x.setTransform(d,0,0,d,0,0);return{{c:c,x:x,w:w,h:h}};}}
  function mx(a,k){{var m=0;for(var i=0;i<a.length;i++){{var v=a[i][k];if(v!=null&&v>m)m=v;}}return m||1;}}
  function line(x,p,k,mv,w,h,c,lw){{x.beginPath();x.strokeStyle=c;x.lineWidth=lw||1.2;var s=false;for(var i=0;i<p.length;i++){{var v=p[i][k];if(v==null)continue;var px=(p[i].t/duration)*w,py=h-(v/mv)*(h-4)-2;if(!s){{x.moveTo(px,py);s=true;}}else x.lineTo(px,py);}}x.stroke();}}
  function filled(x,p,k,mv,w,h,sc,fc){{x.beginPath();var s=false;for(var i=0;i<p.length;i++){{var v=p[i][k];if(v==null)continue;var px=(p[i].t/duration)*w,py=h-(v/mv)*(h-4)-2;if(!s){{x.moveTo(px,h);x.lineTo(px,py);s=true;}}else x.lineTo(px,py);}}if(s){{x.lineTo((p[p.length-1].t/duration)*w,h);x.closePath();x.fillStyle=fc;x.fill();}}line(x,p,k,mv,w,h,sc,1);}}
  function needle(x,w,h){{var px=needlePct*w;x.beginPath();x.strokeStyle=NC;x.lineWidth=1.5;x.setLineDash([4,3]);x.moveTo(px,0);x.lineTo(px,h);x.stroke();x.setLineDash([]);}}
  function drawLbls(x,w,h){{for(var i=0;i<labels.length;i++){{var lb=labels[i],px=(lb.t/duration)*w;x.beginPath();x.strokeStyle=LC;x.lineWidth=2;x.moveTo(px,0);x.lineTo(px,h);x.stroke();x.beginPath();x.fillStyle=LC;x.moveTo(px,4);x.lineTo(px+5,10);x.lineTo(px,16);x.lineTo(px-5,10);x.closePath();x.fill();}}}}

  function draw(){{
    var p=setup("chart-power",80);var mp=Math.max(mx(pts,"power"),100),mh=Math.max(mx(pts,"hr"),100);
    var zt=[0.55,0.75,0.87,0.95,1.05];for(var z=0;z<zt.length;z++){{var yB=p.h-(z===0?0:zt[z-1])*(p.h-4)-2,yT=p.h-zt[z]*(p.h-4)-2;p.x.fillStyle=ZC[z];p.x.fillRect(0,yT,p.w,yB-yT);}}
    line(p.x,pts,"power",mp,p.w,p.h,PC,1.5);line(p.x,pts,"hr",mh,p.w,p.h,HC,1.2);
    p.x.font="10px -apple-system,sans-serif";p.x.fillStyle=PC;p.x.textAlign="left";p.x.fillText(mp+"W",2,12);
    p.x.fillStyle=HC;p.x.textAlign="right";p.x.fillText(mh+"bpm",p.w-2,12);drawLbls(p.x,p.w,p.h);needle(p.x,p.w,p.h);

    var s=setup("chart-speed",45);var ms=Math.max(mx(pts,"speed"),5),mc=Math.max(mx(pts,"cadence"),50);
    line(s.x,pts,"speed",ms,s.w,s.h,SC,1.3);line(s.x,pts,"cadence",mc,s.w,s.h,CC,1);
    s.x.font="10px -apple-system,sans-serif";s.x.fillStyle=SC;s.x.textAlign="left";s.x.fillText(ms+"mph",2,12);
    s.x.fillStyle=CC;s.x.textAlign="right";s.x.fillText(mc+"rpm",s.w-2,12);needle(s.x,s.w,s.h);

    var a=setup("chart-alt",30);var ma=Math.max(mx(pts,"altitude_ft"),10);
    filled(a.x,pts,"altitude_ft",ma,a.w,a.h,AC,AF);
    a.x.font="10px -apple-system,sans-serif";a.x.fillStyle=AC;a.x.textAlign="left";a.x.fillText(Math.round(ma)+"ft",2,12);
    needle(a.x,a.w,a.h);

    var cl=setup("chart-clips",28);cl.x.fillStyle="#111";cl.x.fillRect(0,0,cl.w,cl.h);
    for(var i=0;i<clips.length;i++){{var c=clips[i],x0=(c.start/duration)*cl.w,x1=(c.end/duration)*cl.w,bw=Math.max(2,x1-x0);
    cl.x.fillStyle=i%2===0?"rgba(0,184,148,0.35)":"rgba(0,184,148,0.25)";cl.x.fillRect(x0,2,bw,cl.h-4);
    cl.x.strokeStyle="rgba(0,184,148,0.7)";cl.x.lineWidth=1;cl.x.strokeRect(x0,2,bw,cl.h-4);
    if(bw>30){{cl.x.font="8px -apple-system,sans-serif";cl.x.fillStyle="#ccc";cl.x.textAlign="center";cl.x.fillText(c.name.replace("GX01","").replace(".MP4",""),x0+bw/2,cl.h/2+3);}}}}
    var nx=needlePct*cl.w;cl.x.beginPath();cl.x.strokeStyle=NC;cl.x.lineWidth=2;cl.x.moveTo(nx,0);cl.x.lineTo(nx,cl.h);cl.x.stroke();
    cl.x.beginPath();cl.x.fillStyle=NC;cl.x.moveTo(nx-5,0);cl.x.lineTo(nx+5,0);cl.x.lineTo(nx,7);cl.x.closePath();cl.x.fill();

    var t=setup("chart-time",18);t.x.font="9px -apple-system,sans-serif";t.x.fillStyle="#888";t.x.textAlign="center";
    var iv=300;if(duration>7200)iv=600;if(duration<1800)iv=120;
    for(var sec=0;sec<=duration;sec+=iv){{var px=(sec/duration)*t.w;var hrs=Math.floor(sec/3600),mins=Math.floor((sec%3600)/60);
    var lb=hrs>0?(hrs+":"+(mins<10?"0":"")+mins+":00"):(mins+":00");t.x.fillText(lb,px,13);
    t.x.beginPath();t.x.strokeStyle="#444";t.x.moveTo(px,0);t.x.lineTo(px,3);t.x.stroke();}}

    var ro=document.getElementById("readout");var ns=needlePct*duration;
    var nh=Math.floor(ns/3600),nm=Math.floor((ns%3600)/60),nsc=Math.floor(ns%60);
    var ts=nh>0?nh+":"+(nm<10?"0":"")+nm+":"+(nsc<10?"0":"")+nsc:nm+":"+(nsc<10?"0":"")+nsc;
    var ci=0,bd=1e9;for(var i=0;i<pts.length;i++){{var d=Math.abs(pts[i].t-ns);if(d<bd){{bd=d;ci=i;}}}}
    var cp=pts[ci]||{{}};
    ro.innerHTML="<b>"+ts+"</b> &nbsp;|&nbsp; <span style='color:"+PC+"'>"+(cp.power||0)+"W</span> &nbsp; <span style='color:"+HC+"'>"+(cp.hr||0)+"bpm</span> &nbsp; <span style='color:"+SC+"'>"+(cp.speed||0)+"mph</span> &nbsp; <span style='color:"+CC+"'>"+(cp.cadence||0)+"rpm</span>";
  }}

  draw();window.addEventListener("resize",draw);

  var ids=["chart-power","chart-speed","chart-alt","chart-clips","chart-time"];
  function gp(e,c){{var r=c.getBoundingClientRect();var x=(e.touches?e.touches[0].clientX:e.clientX)-r.left;return Math.max(0,Math.min(1,x/r.width));}}
  function commit(){{var s=Math.round(needlePct*duration);var u=new URL(window.top.location.href);u.searchParams.set("pos",s.toString());u.searchParams.set("pos_changed","1");window.top.history.pushState(null,"",u.toString());window.top.dispatchEvent(new PopStateEvent("popstate"));}}
  ids.forEach(function(id){{var el=document.getElementById(id);el.addEventListener("mousedown",function(e){{dragging=true;needlePct=gp(e,el);draw();document.body.style.cursor="ew-resize";}});el.addEventListener("touchstart",function(e){{dragging=true;needlePct=gp(e,el);draw();}},{{passive:true}});}});
  document.addEventListener("mousemove",function(e){{if(!dragging)return;e.preventDefault();needlePct=gp(e,document.getElementById("chart-power"));draw();}});
  document.addEventListener("touchmove",function(e){{if(!dragging)return;needlePct=gp(e,document.getElementById("chart-power"));draw();}},{{passive:true}});
  document.addEventListener("mouseup",function(){{if(!dragging)return;dragging=false;document.body.style.cursor="";commit();}});
  document.addEventListener("touchend",function(){{if(!dragging)return;dragging=false;commit();}});
}})();
</script>
"""
    st.components.v1.html(html, height=280)


if __name__ == "__main__":
    main()
