"""Flask application for single-clip overlay preview.

Backs the `gopro-garmin review` command: play one GoPro clip with
real-time telemetry overlays drawn from the synced FIT data. Candidate
review lives in the Streamlit reviewer (candidate_review.py), not here.

    create_app(video_path='.../GX010048.MP4', fit_path='....fit')
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from ..fit_parser import parse_fit
from ..gopro_meta import extract_metadata
from ..sync import auto_sync
from .fit_data import prepare_ride_json


# ─── Helpers ────────────────────────────────────────────────

def _build_ride_payload(clip_path: Path, fit_path: Path, offset: float) -> dict:
    """Telemetry-synced payload for a single clip."""
    clip = extract_metadata(clip_path.resolve())
    ride = parse_fit(fit_path)
    synced = auto_sync(clip, ride, offset)
    return prepare_ride_json(synced)


# ─── App factory ────────────────────────────────────────────

def create_app(
    video_path: str | None = None,
    fit_path: str | None = None,
    offset: float = 0.0,
) -> Flask:
    """Create the Flask app for one clip + one FIT file."""
    if not fit_path:
        raise ValueError("fit_path is required")
    if not video_path:
        raise ValueError("video_path is required")

    app = Flask(__name__)
    app.config["FIT_PATH"] = Path(fit_path).resolve()
    app.config["OFFSET"] = offset

    clip_path = Path(video_path).resolve()
    app.config["RIDE_DIR"] = clip_path.parent
    app.config["CLIP_PATHS"] = [clip_path]
    app.config["RIDE_CACHE"] = {
        clip_path.name: _build_ride_payload(
            clip_path, app.config["FIT_PATH"], offset,
        )
    }

    # ── Routes ─────────────────────────────────────────────

    @app.route("/")
    def index():
        clips = app.config["CLIP_PATHS"]
        return render_template("index.html", filename=clips[0].name)

    @app.route("/api/ride-data")
    def ride_data():
        """Per-clip telemetry payload. Optional ?clip=<filename>."""
        clip_name = request.args.get(
            "clip", app.config["CLIP_PATHS"][0].name,
        )
        try:
            payload = _ride_payload_for(app, clip_name)
        except FileNotFoundError:
            abort(404)
        resp = jsonify(payload)
        resp.headers["Cache-Control"] = "public, max-age=3600"
        return resp

    @app.route("/video/<path:filename>")
    def video(filename):
        # Serve any clip in the ride folder by name. Path() guards against
        # traversal — only a flat filename inside RIDE_DIR is allowed.
        if "/" in filename or ".." in filename:
            abort(403)
        target = app.config["RIDE_DIR"] / filename
        if not target.exists() or not target.is_file():
            abort(404)
        return send_file(target, mimetype="video/mp4", conditional=True)

    return app


# ─── App helpers ────────────────────────────────────────────

def _ride_payload_for(app: Flask, clip_name: str) -> dict:
    """Return cached or freshly-computed telemetry payload for a clip."""
    cache: dict = app.config["RIDE_CACHE"]
    if clip_name in cache:
        return cache[clip_name]
    target = app.config["RIDE_DIR"] / clip_name
    if not target.exists():
        raise FileNotFoundError(clip_name)
    payload = _build_ride_payload(
        target, app.config["FIT_PATH"], app.config["OFFSET"],
    )
    cache[clip_name] = payload
    return payload
