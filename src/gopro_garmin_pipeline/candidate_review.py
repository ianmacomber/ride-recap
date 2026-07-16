"""Candidate reviewer — Streamlit app to browse, pick, and rate highlight candidates.

Shows all candidates (telemetry highlights + Strava segments + optional labels)
with 5-second preview clips and scores. User picks which to include and rates
them 1-5. Selections are saved and fed directly into the composer.

Labels are optional — the pipeline works with just video + FIT file.

Launch:
    gopro-garmin review-candidates <video-dir> <fit> [--labels FILE] [--strava-activity ID]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import streamlit as st

try:
    from .utils import format_time, gopro_lrv_proxy
except ImportError:
    from gopro_garmin_pipeline.utils import format_time, gopro_lrv_proxy

st.set_page_config(page_title="Candidate Review", layout="wide")

_SELECTIONS_FILE = "candidate_selections.json"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--fit", required=True)
    parser.add_argument("--labels", default=None, help="Optional labels JSON file")
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--strava-activity", type=int, default=None)
    return parser.parse_args()


def _serialize_segment(c) -> dict:
    """Single canonical reviewer-shape for a Segment.

    Same shape whether candidates were freshly generated here or loaded
    from a precomputed candidates.json file. anchor_video_secs, rubric,
    portrait_crop_bias, sources, and stable_id all survive.
    """
    return {
        "stable_id": c.stable_id,
        "clip_name": c.clip_name,
        "video_start": c.video_start,
        "video_end": c.video_end,
        "anchor_video_secs": c.anchor_video_secs,
        "ride_time_secs": c.ride_time_secs,
        "score": c.score,
        "source": c.source,
        "sources": c.sources,
        "rubric": c.rubric,
        "portrait_crop_bias": c.portrait_crop_bias,
        "notes": c.label.get("notes", ""),
        "type": c.label.get("type", ""),
        "star_count": c.label.get("star_count", 0),
    }


def _hydrate_serialized(raw: list[dict]) -> list[dict]:
    """Backfill stable_id (and missing optional fields) on candidates.json
    written before stable_id was introduced. Idempotent.
    """
    from gopro_garmin_pipeline.composer import stable_id_for
    for c in raw:
        if "stable_id" not in c or not c.get("stable_id"):
            anchor = c.get(
                "anchor_video_secs",
                (c.get("video_start", 0.0) + c.get("video_end", 0.0)) / 2,
            )
            c["stable_id"] = stable_id_for(c.get("clip_name", ""), anchor)
        c.setdefault("rubric", {})
        c.setdefault("portrait_crop_bias", 0.0)
        c.setdefault("sources", [c.get("source", "")] if c.get("source") else [])
        if "anchor_video_secs" not in c:
            c["anchor_video_secs"] = (c.get("video_start", 0.0) + c.get("video_end", 0.0)) / 2
    return raw


@st.cache_data(show_spinner="Generating candidates...")
def _load_candidates(fit_path, video_dir, offset, strava_activity_id,
                     labels_path=None):
    # Check for precomputed candidates (from `gopro-garmin process`)
    precomputed = Path(video_dir) / "candidates.json"
    if precomputed.exists():
        raw = json.loads(precomputed.read_text())
        candidates = _hydrate_serialized(raw)
        candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        return candidates

    from gopro_garmin_pipeline.composer import (
        ComposerConfig, generate_all_candidates,
    )

    config = ComposerConfig(
        offset=offset,
        strava_activity_id=strava_activity_id,
    )

    lp = Path(labels_path) if labels_path and Path(labels_path).exists() else None
    candidates, ride, synced = generate_all_candidates(
        video_dir, fit_path, config, labels_path=lp,
    )

    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)

    return [_serialize_segment(c) for c in candidates]


def _proxy_for(src: Path) -> Path:
    """Return GoPro low-res proxy (.LRV) if it exists, else the source MP4."""
    proxy = gopro_lrv_proxy(src)
    return proxy if proxy is not None else src


def _safe_filename(stable_id: str) -> str:
    """Filesystem-safe version of a stable_id (only [A-Za-z0-9_-])."""
    import re
    return re.sub(r"[^A-Za-z0-9_-]", "_", stable_id)


@st.cache_data(show_spinner="Extracting preview clips...")
def _extract_previews(candidates, video_dir, min_duration=5.0, max_duration=20.0):
    """Extract a preview clip spanning each candidate's full action window.

    Preview filenames key off stable_id (clip stem + anchor second), so
    rerunning with a regenerated candidate set keeps the same file path
    for the same moment — no stale-preview-on-wrong-clip surprises.
    Reads from the GoPro .LRV proxy when available and runs ffmpeg
    jobs in parallel.
    """
    preview_dir = Path(video_dir) / ".previews"
    preview_dir.mkdir(exist_ok=True)

    tasks = []
    for c in candidates:
        sid = c["stable_id"]
        out = preview_dir / f"preview_{_safe_filename(sid)}.mp4"
        if out.exists():
            continue
        src = _proxy_for(Path(video_dir) / c["clip_name"])
        v_start = c["video_start"]
        v_end = c["video_end"]
        span = v_end - v_start
        duration = max(min_duration, min(max_duration, span + 2.0))
        mid = (v_start + v_end) / 2
        start = max(0, mid - duration / 2)
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start:.2f}",
            "-i", str(src), "-t", f"{duration:.2f}",
            "-vf", "scale=480:-1",
            "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast",
            "-an", "-pix_fmt", "yuv420p",
            str(out),
        ]
        tasks.append(cmd)

    if tasks:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda c: subprocess.run(c, capture_output=True), tasks))

    previews = {}
    for c in candidates:
        sid = c["stable_id"]
        out = preview_dir / f"preview_{_safe_filename(sid)}.mp4"
        if out.exists():
            previews[sid] = str(out)
    return previews


def _load_selections(video_dir):
    p = Path(video_dir) / _SELECTIONS_FILE
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_selections(selections, video_dir):
    p = Path(video_dir) / _SELECTIONS_FILE
    p.write_text(json.dumps(selections, indent=2))


def _migrate_idx_keyed_selections(
    selections: dict, candidates: list[dict],
) -> dict:
    """Convert any legacy idx-keyed selection entries to stable_id keys.

    Older runs persisted selections as {"0": {...}, "1": {...}}. Any key
    that is purely numeric is remapped using the candidate at that
    position (when available). New entries are always stable_id-keyed
    so this only ever runs once per file.
    """
    if not selections:
        return selections
    has_legacy = any(k.isdigit() for k in selections.keys())
    if not has_legacy:
        return selections

    by_idx = {i: c for i, c in enumerate(candidates)}
    migrated: dict = {}
    for key, val in selections.items():
        if key.isdigit():
            idx = int(key)
            cand = by_idx.get(idx)
            if cand is None:
                continue  # candidate at that idx is gone
            migrated[cand["stable_id"]] = val
        else:
            migrated[key] = val
    return migrated


def main():
    args = _parse_args()
    video_dir = args.video_dir

    candidates = _load_candidates(
        args.fit, video_dir, args.offset, args.strava_activity,
        labels_path=args.labels,
    )
    previews = _extract_previews(candidates, video_dir)

    st.markdown("""<style>
        .block-container { padding-top: 1rem; }
        h1 { font-size: 1.5rem !important; margin-bottom: 0 !important; }
    </style>""", unsafe_allow_html=True)

    st.title("Candidate Review")
    st.caption(f"{len(candidates)} candidates · ranked by score")

    # Load saved selections into session state, migrating legacy idx keys
    # to stable_id keys on first load after upgrade.
    if "selections" not in st.session_state:
        loaded = _load_selections(video_dir)
        st.session_state.selections = _migrate_idx_keyed_selections(loaded, candidates)

    # Summary bar
    sel = st.session_state.selections
    picked = [c for c in candidates if sel.get(c["stable_id"], {}).get("included")]
    total_dur = sum(c["video_end"] - c["video_start"] for c in picked)
    rated = [c for c in candidates if sel.get(c["stable_id"], {}).get("rating")]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Selected", f"{len(picked)}")
    col2.metric("Total duration", f"{total_dur:.0f}s")
    col3.metric("Rated", f"{len(rated)}/{len(candidates)}")
    if col4.button("Save & Compose", type="primary"):
        _save_selections(st.session_state.selections, video_dir)
        # Write a compose-ready selections file
        selected_labels = []
        for c in sorted(picked, key=lambda x: x["ride_time_secs"]):
            selected_labels.append({
                "stable_id": c["stable_id"],
                "clip_name": c["clip_name"],
                "video_start": c["video_start"],
                "video_end": c["video_end"],
                "anchor_video_secs": c.get(
                    "anchor_video_secs",
                    (c["video_start"] + c["video_end"]) / 2,
                ),
                "ride_time_secs": c["ride_time_secs"],
                "score": c["score"],
                "rubric": c.get("rubric", {}),
                "notes": c["notes"],
                "rating": sel.get(c["stable_id"], {}).get("rating", 0),
                "portrait_crop_bias": c.get("portrait_crop_bias", 0.0),
            })
        sel_path = Path(video_dir) / "selected_candidates.json"
        sel_path.write_text(json.dumps(selected_labels, indent=2))
        st.success(f"Saved {len(selected_labels)} selections to {sel_path}")

    st.divider()

    # Filter controls
    show_filter = st.radio(
        "Show", ["All", "Selected only", "Unrated only"],
        horizontal=True, label_visibility="collapsed",
    )

    # Candidate grid — 3 columns
    if show_filter == "Selected only":
        visible = [c for c in candidates if sel.get(c["stable_id"], {}).get("included")]
    elif show_filter == "Unrated only":
        visible = [c for c in candidates if not sel.get(c["stable_id"], {}).get("rating")]
    else:
        visible = candidates

    cols_per_row = 3
    for row_start in range(0, len(visible), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, c in enumerate(visible[row_start:row_start + cols_per_row]):
            sid = c["stable_id"]
            with cols[col_idx]:
                # Preview clip
                preview_path = previews.get(sid)
                if preview_path:
                    st.video(preview_path)

                # Info
                time_str = format_time(c["ride_time_secs"])
                source = c.get("source", c["type"])
                stars = f" · {c['star_count']}★" if c.get("star_count") else ""
                st.markdown(
                    f"**{time_str}** · score {c['score']:.1f}/10{stars}  \n"
                    f"*{source}*: {c['notes']}"
                )

                # Rubric breakdown — only Gemini-rated clips have one
                rubric = c.get("rubric") or {}
                if rubric:
                    parts = []
                    for key, label in (
                        ("light", "L"),
                        ("composition", "C"),
                        ("motion", "M"),
                        ("scenery", "S"),
                        ("subject", "👁"),
                    ):
                        v = rubric.get(key)
                        if v is not None:
                            parts.append(f"{label}{v}")
                    if parts:
                        st.caption(" · ".join(parts))

                # Controls
                c1, c2 = st.columns([1, 2])
                with c1:
                    included = st.checkbox(
                        "Include",
                        value=sel.get(sid, {}).get("included", False),
                        key=f"inc_{sid}",
                    )
                with c2:
                    # Migrate any legacy 1-5 ratings to 1-10 (multiply by 2)
                    saved_rating = sel.get(sid, {}).get("rating", 0)
                    if saved_rating and saved_rating <= 5:
                        saved_rating = saved_rating * 2
                    rating = st.select_slider(
                        "Rate",
                        options=list(range(0, 11)),
                        value=saved_rating,
                        key=f"rate_{sid}",
                        label_visibility="collapsed",
                    )

                # Update session state
                st.session_state.selections[sid] = {
                    "included": included,
                    "rating": rating,
                    "ride_time_secs": c["ride_time_secs"],
                    "notes": c["notes"],
                }

    # Auto-save on every interaction
    _save_selections(st.session_state.selections, video_dir)


if __name__ == "__main__":
    main()
