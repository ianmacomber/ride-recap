"""Flask application for ride-level unified review.

Backwards-compatible: if launched with a single .MP4 path, behaves like
the original single-clip telemetry viewer. If launched with a directory,
exposes a multi-clip review surface backed by moments.json.

Single-clip mode (legacy):
    create_app(video_path='.../GX010048.MP4', fit_path='....fit')

Ride-review mode (preferred for unified review):
    create_app(ride_dir='.../2026-04-19', fit_path='....fit')

The API is shaped around MomentProposals — see proposals.py. The
frontend reads /api/moments, mutates them via /api/moments/<id>, and
finally posts to /api/save-selections to produce a compose-ready
selected_candidates.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from ..fit_parser import parse_fit
from ..gopro_meta import extract_all, extract_metadata
from ..proposals import (
    MomentProposal,
    STATUS_APPROVED,
    STATUS_PENDING,
    STATUS_REJECTED,
    load_proposals,
    save_proposals,
    stable_id_for,
)
from ..sync import auto_sync, normalize_tz
from .fit_data import prepare_ride_json


_MOMENTS_FILENAME = "moments.json"
_SELECTIONS_FILENAME = "selected_candidates.json"


# ─── Helpers ────────────────────────────────────────────────

def _list_clips(ride_dir: Path) -> list[Path]:
    """All GoPro .MP4 chapters in a ride folder, sorted."""
    mp4s = list(ride_dir.glob("*.MP4")) + list(ride_dir.glob("*.mp4"))
    # Stable order: GoPro chapter naming sorts naturally
    return sorted(mp4s)


def _build_ride_payload(clip_path: Path, fit_path: Path, offset: float) -> dict:
    """Telemetry-synced payload for a single clip — same shape as legacy."""
    clip = extract_metadata(clip_path.resolve())
    ride = parse_fit(fit_path)
    synced = auto_sync(clip, ride, offset)
    return prepare_ride_json(synced)


# ─── App factory ────────────────────────────────────────────

def create_app(
    video_path: str | None = None,
    fit_path: str | None = None,
    offset: float = 0.0,
    ride_dir: str | None = None,
) -> Flask:
    """Create the Flask app.

    Provide EITHER `video_path` (single-clip legacy mode) OR `ride_dir`
    (multi-clip ride review). `fit_path` is required in both modes.
    """
    if not fit_path:
        raise ValueError("fit_path is required")
    if not video_path and not ride_dir:
        raise ValueError("Provide either video_path or ride_dir")

    app = Flask(__name__)
    app.config["FIT_PATH"] = Path(fit_path).resolve()
    app.config["OFFSET"] = offset

    # Parse FIT once — used for ride duration + per-clip ride-time offsets.
    ride_obj = parse_fit(app.config["FIT_PATH"])
    app.config["RIDE_START"] = ride_obj.start_time
    app.config["RIDE_END"] = ride_obj.end_time
    app.config["RIDE_DURATION_SECS"] = (
        (ride_obj.end_time - ride_obj.start_time).total_seconds()
        if ride_obj.start_time and ride_obj.end_time else 0.0
    )

    if ride_dir:
        ride_dir_path = Path(ride_dir).resolve()
        # Use extract_all so multi-chapter recordings get the correct
        # cumulative creation_times — without this, all chapters of one
        # GoPro recording report the same wall-clock start and the ride
        # timeline collapses.
        adjusted = extract_all(ride_dir_path)
        if not adjusted:
            raise ValueError(f"No GoPro .MP4 files found in {ride_dir_path}")
        app.config["MODE"] = "ride"
        app.config["RIDE_DIR"] = ride_dir_path
        app.config["CLIP_PATHS"] = [c.path for c in adjusted]
        app.config["CLIPS_BY_NAME"] = {c.path.name: c for c in adjusted}
        first = adjusted[0]
        app.config["RIDE_CACHE"] = {
            first.path.name: _build_ride_payload(
                first.path, app.config["FIT_PATH"], offset,
            )
        }
    else:
        clip_path = Path(video_path).resolve()
        adjusted = extract_metadata(clip_path)
        app.config["MODE"] = "single"
        app.config["RIDE_DIR"] = clip_path.parent
        app.config["CLIP_PATHS"] = [clip_path]
        app.config["CLIPS_BY_NAME"] = {clip_path.name: adjusted}
        app.config["RIDE_CACHE"] = {
            clip_path.name: _build_ride_payload(
                clip_path, app.config["FIT_PATH"], offset,
            )
        }

    # ── Routes ─────────────────────────────────────────────

    @app.route("/")
    def index():
        clips = app.config["CLIP_PATHS"]
        return render_template(
            "index.html",
            filename=clips[0].name,
            mode=app.config["MODE"],
        )

    @app.route("/api/clips")
    def list_clips():
        """List all clips in the ride folder, sorted, with durations
        and each clip's ride-relative start time so the frontend can
        place clips on a unified ride timeline."""
        ride_start = app.config["RIDE_START"]
        clips_by_name = app.config["CLIPS_BY_NAME"]
        clips_meta = []
        for p in app.config["CLIP_PATHS"]:
            clip = clips_by_name.get(p.name)
            ride_offset = _clip_ride_offset(clip, ride_start, app.config["OFFSET"])
            clips_meta.append({
                "filename": p.name,
                "duration_secs": clip.duration_secs if clip else 0.0,
                "ride_start_secs": ride_offset,
            })
        return jsonify(clips_meta)

    @app.route("/api/ride-info")
    def ride_info():
        """Total ride duration so the frontend can scale the overview strip."""
        return jsonify({
            "ride_duration_secs": app.config["RIDE_DURATION_SECS"],
        })

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

    @app.route("/api/moments")
    def get_moments():
        """Return the list of MomentProposals for this ride."""
        proposals = _load_or_empty(app)
        return jsonify([p.to_dict() for p in proposals])

    @app.route("/api/moments/<stable_id>", methods=["POST"])
    def update_moment(stable_id: str):
        """Patch fields on a single moment. Body: {field: value, ...}.

        Accepts: status, rating, user_anchor_override, user_in_out,
        notes, final_trim_secs, portrait_crop_bias.
        """
        body = request.get_json(silent=True) or {}
        proposals = _load_or_empty(app)
        target = next((p for p in proposals if p.stable_id == stable_id), None)
        if target is None:
            abort(404)
        _apply_patch(target, body)
        save_proposals(proposals, _moments_path(app))
        return jsonify(target.to_dict())

    @app.route("/api/moments", methods=["POST"])
    def create_moment():
        """Create a new manual moment from a scrub position.

        Body: {clip_name, anchor_video_secs, video_start, video_end,
               ride_time_secs, notes}
        """
        body = request.get_json(silent=True) or {}
        clip_name = body.get("clip_name")
        anchor = body.get("anchor_video_secs")
        if not clip_name or anchor is None:
            return jsonify({"error": "clip_name and anchor_video_secs required"}), 400

        v_start = float(body.get("video_start", anchor - 5))
        v_end = float(body.get("video_end", anchor + 5))
        ride_time_secs = float(body.get("ride_time_secs", 0.0))

        proposal = MomentProposal(
            stable_id=stable_id_for(clip_name, float(anchor)),
            clip_name=clip_name,
            video_start=max(0, v_start),
            video_end=v_end,
            anchor_video_secs=float(anchor),
            ride_time_secs=ride_time_secs,
            sources=["manual"],
            score=0.0,
            notes=body.get("notes", "manual add"),
            type="manual",
            status=STATUS_APPROVED,  # user explicitly added → approved by default
        )

        proposals = _load_or_empty(app)
        # If a moment with this stable_id already exists, just approve it
        existing = next((p for p in proposals if p.stable_id == proposal.stable_id), None)
        if existing is not None:
            existing.status = STATUS_APPROVED
            if "manual" not in existing.sources:
                existing.sources = sorted({*existing.sources, "manual"})
            save_proposals(proposals, _moments_path(app))
            return jsonify(existing.to_dict())

        proposals.append(proposal)
        save_proposals(proposals, _moments_path(app))
        return jsonify(proposal.to_dict()), 201

    @app.route("/api/save-selections", methods=["POST"])
    def save_selections():
        """Write selected_candidates.json from approved moments — same
        shape compose-selected has always consumed."""
        proposals = _load_or_empty(app)
        approved = [p for p in proposals if p.status == STATUS_APPROVED]
        approved.sort(key=lambda p: p.ride_time_secs)
        out = []
        for p in approved:
            v_start, v_end = p.effective_review_span
            out.append({
                "stable_id": p.stable_id,
                "clip_name": p.clip_name,
                "video_start": v_start,
                "video_end": v_end,
                "anchor_video_secs": p.effective_anchor,
                "ride_time_secs": p.ride_time_secs,
                "score": p.score,
                "rubric": p.rubric,
                "notes": p.notes,
                "rating": p.rating,
                "portrait_crop_bias": p.portrait_crop_bias,
            })
        sel_path = app.config["RIDE_DIR"] / _SELECTIONS_FILENAME
        sel_path.write_text(json.dumps(out, indent=2))
        return jsonify({"saved": len(out), "path": str(sel_path)})

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


def _clip_ride_offset(clip, ride_start, offset_secs: float) -> float:
    """Ride-relative seconds at which this clip's video time 0 lands.

    `clip` is a chapter-adjusted GoProClip (so multi-chapter recordings
    get sequential offsets, not all stacked at chapter 1's wall time).
    Returns 0.0 if the FIT has no start_time.
    """
    if ride_start is None or clip is None:
        return 0.0
    clip_start = normalize_tz(clip.creation_time, ride_start)
    # offset_secs is FIT - GoPro, so we add it to map clip time to FIT time
    return (clip_start - ride_start).total_seconds() + offset_secs


def _moments_path(app: Flask) -> Path:
    return app.config["RIDE_DIR"] / _MOMENTS_FILENAME


def _load_or_empty(app: Flask) -> list[MomentProposal]:
    """Load moments.json, or return empty list if missing."""
    return load_proposals(_moments_path(app))


def _apply_patch(p: MomentProposal, body: dict) -> None:
    """Apply a partial-update body onto a MomentProposal (allowlist)."""
    if "status" in body:
        s = body["status"]
        if s in (STATUS_APPROVED, STATUS_REJECTED, STATUS_PENDING):
            p.status = s
    if "rating" in body:
        try:
            p.rating = max(0, min(10, int(body["rating"])))
        except (TypeError, ValueError):
            pass
    if "user_anchor_override" in body:
        v = body["user_anchor_override"]
        p.user_anchor_override = float(v) if v is not None else None
    if "user_in_out" in body:
        v = body["user_in_out"]
        if v is None:
            p.user_in_out = None
        elif isinstance(v, (list, tuple)) and len(v) == 2:
            p.user_in_out = (float(v[0]), float(v[1]))
    if "notes" in body and isinstance(body["notes"], str):
        p.notes = body["notes"]
    if "final_trim_secs" in body:
        try:
            p.final_trim_secs = max(0.5, float(body["final_trim_secs"]))
        except (TypeError, ValueError):
            pass
    if "portrait_crop_bias" in body:
        try:
            p.portrait_crop_bias = max(-1.0, min(1.0, float(body["portrait_crop_bias"])))
        except (TypeError, ValueError):
            pass
