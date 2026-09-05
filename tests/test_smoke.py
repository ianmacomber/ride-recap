"""Smoke tests — no video, no FIT file, no network, no API keys.

These cover the pieces a contributor is most likely to break without noticing:
the config derivation, the design-token contract, and the ride-geometry
classifier. Everything here runs in well under a second.
"""

from __future__ import annotations

import datetime as dt
import math
import sys

import pytest

from gopro_garmin_pipeline.burn_overlay import ENCODE_PREVIEW, _encode_args
from gopro_garmin_pipeline.config import Settings
from gopro_garmin_pipeline.design import tokens as T
from gopro_garmin_pipeline import route_metadata as rm
from gopro_garmin_pipeline import prompt_eval as pe
from gopro_garmin_pipeline.utils import rating_visual_action


# ─── FIT and GoPro sync ─────────────────────────────────────

def test_fit_timestamps_parse_as_utc_aware():
    """Regression: FIT timestamps are naive in the file; parse_fit attaches UTC."""
    from gopro_garmin_pipeline.fit_parser import RideData, RidePoint
    from gopro_garmin_pipeline.gopro_meta import GoProClip
    from gopro_garmin_pipeline.sync import auto_sync
    from pathlib import Path

    # FIT timestamps arrive timezone-aware UTC from parse_fit.
    fit_ts = dt.datetime(2026, 8, 16, 11, 38, 21, tzinfo=dt.timezone.utc)
    ride = RideData(points=[RidePoint(timestamp=fit_ts, speed=5.0)], start_time=fit_ts)

    # GoPro: UTC timestamp from ffprobe
    gopo_creation = dt.datetime(2026, 8, 16, 7, 41, 0, tzinfo=dt.timezone.utc)
    clip = GoProClip(
        path=Path("test.mp4"),
        creation_time=gopo_creation,
        duration_secs=60.0,
        width=1920, height=1080, fps=30.0
    )

    synced = auto_sync(clip, ride, offset_secs=0.0)
    # Without a clock correction, the adjusted time stays aware UTC.
    adjusted = synced._adjust(0.0)
    assert adjusted == gopo_creation


# ─── FFmpeg encoders ─────────────────────────────────────────

def test_preview_encode_args_use_compatible_encoder():
    args = _encode_args(ENCODE_PREVIEW)
    if sys.platform.startswith("darwin"):
        assert "h264_videotoolbox" in args
        assert "libx264" not in args
    else:
        assert "libx264" in args
        assert "h264_videotoolbox" not in args

# ─── Config ───────────────────────────────────────────────────

def test_spike_thresholds_derive_from_athlete_zones():
    s = Settings(_env_file=None, ftp=200, max_heart_rate=180)
    assert s.power_spike_threshold == pytest.approx(290)  # 1.45x FTP
    assert s.hr_spike_threshold == pytest.approx(157)      # 0.87x max HR


def test_explicit_thresholds_win_over_derivation():
    s = Settings(_env_file=None, ftp=200, power_spike_threshold=999)
    assert s.power_spike_threshold == pytest.approx(999)


def test_osm_disabled_without_contact_email():
    """No contact address → no OSM traffic. Nominatim's policy requires one."""
    s = Settings(_env_file=None)
    assert s.osm_contact_email == ""


# ─── Athlete zones actually reach detection ───────────────────
# Regression: the thresholds once derived from FTP in Settings while
# HighlightConfig kept its own hardcoded 350/170, so setting FTP silently
# did nothing to highlight detection. A derived value that reaches no
# consumer is worse than a constant — it reads as configurable and isn't.

def test_highlight_config_derives_from_athlete_zones(monkeypatch):
    from gopro_garmin_pipeline import config as cfg
    from gopro_garmin_pipeline.highlights import HighlightConfig

    monkeypatch.setenv("FTP", "150")
    monkeypatch.setenv("MAX_HEART_RATE", "160")
    cfg.get_settings.cache_clear()
    try:
        hc = HighlightConfig()
        assert hc.power_spike_threshold == pytest.approx(218)  # 1.45 * 150
        assert hc.hr_spike_threshold == pytest.approx(139)      # 0.87 * 160
    finally:
        cfg.get_settings.cache_clear()


def test_explicit_highlight_thresholds_win():
    from gopro_garmin_pipeline.highlights import HighlightConfig
    assert HighlightConfig(power_spike_threshold=600).power_spike_threshold == 600


# ─── Design tokens ────────────────────────────────────────────

def test_serif_font_resolves_to_a_real_file():
    """The italic serif is not bundled; tokens.py must resolve it or fall
    back to a bundled face. A dangling path crashes every outro render."""
    import os
    assert os.path.exists(T.FONT_SERIF), f"serif fallback is dangling: {T.FONT_SERIF}"


def test_bundled_fonts_exist():
    import os
    for path in (T.FONT_NUMERIC, T.FONT_NUMERIC_REG, T.FONT_BODY):
        assert os.path.exists(path), f"bundled font missing: {path}"


def test_package_data_stays_inside_the_package():
    """Fonts and prompts must resolve inside the installed package, not the
    source checkout. Regression: font paths once derived from the repo root,
    which only exists under `pip install -e .` — a plain install crashed
    every overlay burn."""
    import gopro_garmin_pipeline
    from pathlib import Path
    pkg = Path(gopro_garmin_pipeline.__file__).parent
    for path in (T.FONT_NUMERIC, T.FONT_NUMERIC_REG, T.FONT_BODY):
        assert Path(path).is_relative_to(pkg), f"font escapes the package: {path}"

    from gopro_garmin_pipeline.prompt_registry import prompt_body
    assert prompt_body("gemini_scan", "v10").strip()

    # The Flask preview's templates and static assets ship the same way;
    # `review` crashes on a plain install if these fall out of
    # package-data.
    assert (pkg / "web" / "templates" / "index.html").is_file()
    assert (pkg / "web" / "static" / "app.js").is_file()


def test_tokens_carry_no_personal_defaults():
    """The public defaults must not ship one rider's home turf."""
    assert T.LOCKUP_ORIGIN == "START"
    assert T.LOCKUP_ROAD == ""


# ─── Ride geometry ────────────────────────────────────────────

class _Pt:
    def __init__(self, lat, lon, distance=None):
        self.lat, self.lon, self.distance = lat, lon, distance


class _Ride:
    def __init__(self, points):
        self.points = points


def _circle(center_lat, center_lon, radius_mi, n=48):
    """n points on a circle — a loop that returns to its start."""
    deg = radius_mi / 69.0
    return [
        _Pt(center_lat + deg * math.cos(2 * math.pi * i / n),
            center_lon + deg * math.sin(2 * math.pi * i / n) / math.cos(math.radians(center_lat)))
        for i in range(n + 1)
    ]


def test_short_loop_classifies_as_local_loop():
    ride = _Ride(_circle(40.78, -73.96, radius_mi=1.0))
    assert rm.classify_ride(ride)["ride_type"] == "local_loop"


def test_out_and_back_returns_near_start():
    """Ride 20 mi out and come back: far from start at the apex, but ends home."""
    out = [_Pt(40.7 + i * 0.01, -74.0) for i in range(30)]      # ~20 mi north
    back = list(reversed(out))
    cls = rm.classify_ride(_Ride(out + back))
    assert cls["ride_type"] == "out_and_back"
    assert cls["end_to_start_mi"] < 1.0
    assert cls["max_dist_from_start_mi"] > 5.0


def test_point_to_point_ends_far_from_start():
    ride = _Ride([_Pt(40.7 + i * 0.01, -74.0) for i in range(30)])
    cls = rm.classify_ride(ride)
    assert cls["ride_type"] == "point_to_point"
    assert cls["end_to_start_mi"] > 5.0


def test_classify_ride_survives_a_track_with_no_gps():
    """Indoor/trainer FIT files have no lat/lon. Must not raise."""
    cls = rm.classify_ride(_Ride([_Pt(None, None), _Pt(None, None)]))
    assert cls["ride_type"] == "unknown"
    assert cls["distance_mi"] == 0.0


def test_haversine_matches_a_known_distance():
    # Empire State Building → Times Square, ~0.8 mi
    d = rm.haversine_mi(40.7484, -73.9857, 40.7580, -73.9855)
    assert 0.6 < d < 0.9


# ─── Gemini hit → label scale ─────────────────────────────────

def test_v10_rubric_folds_onto_the_label_scale():
    """v10 stores a five-dim rubric, labels store visual/action.

    These have to land in the same 1-10 space or `compare` reports
    nonsense. `light` is excluded, matching composer._rubric_score.
    """
    hit = {"rubric": {"light": 10, "composition": 6, "motion": 4,
                      "scenery": 8, "subject": 2}}
    visual, action = rating_visual_action(hit)
    assert visual == 7   # mean(composition 6, scenery 8)
    assert action == 3   # mean(motion 4, subject 2)


def test_pre_v10_hits_still_read_their_own_visual_action():
    visual, action = rating_visual_action({"visual": 7, "action": 4})
    assert (visual, action) == (7, 4)


def test_a_hit_with_no_scores_at_all_is_zero_not_a_crash():
    assert rating_visual_action({}) == (0, 0)
    assert rating_visual_action({"rubric": {}}) == (0, 0)


def test_anchor_video_secs_is_read_from_v10_hits():
    """v10 renamed video_secs → anchor_video_secs. Reading the old key
    silently placed every hit at ride time zero, so nothing aligned."""
    assert pe._hit_video_secs({"anchor_video_secs": 3.75}) == 3.75
    assert pe._hit_video_secs({"video_secs": 9.0}) == 9.0   # legacy
    assert pe._hit_video_secs({}) == 0.0


def test_explicit_visual_action_wins_over_a_rubric():
    """A human label that also carries a rubric must read its own scores."""
    got = rating_visual_action(
        {"visual": 9, "action": 2, "rubric": {"composition": 1, "scenery": 1}})
    assert got == (9, 2)


def test_serialized_candidate_reads_through_its_nested_label():
    """candidates.json nests the rubric under "label"."""
    seg = {"label": {"rubric": {"composition": 8, "scenery": 8,
                                "motion": 2, "subject": 2}}}
    assert rating_visual_action(seg) == (8, 2)


def test_telemetry_only_candidate_is_honestly_zero():
    """No rubric, no label — nothing looked at this clip. Not a crash."""
    assert rating_visual_action({"score": 4.0, "sources": ["telemetry"]}) == (0, 0)


# ─── Similarity-aware clip selection ────────────────────────

def _seg(ride_secs, speed_mph, power_w, grade_pct, score=6.0, rubric=None):
    from gopro_garmin_pipeline.composer import Segment
    seg = Segment(clip_name="GX010001.MP4", video_start=ride_secs,
                  video_end=ride_secs + 3, ride_time_secs=ride_secs, score=score)
    seg.telemetry_features = {"speed_mph": speed_mph, "power_w": power_w,
                              "grade_pct": grade_pct}
    seg.rubric = rubric if rubric is not None else {
        "composition": 7, "motion": 7, "scenery": 7, "subject": 7, "light": 7}
    return seg


_TAU = 750.0  # stand-in for _diversity_tau on a 4h ride at 20 slots


def test_climb_and_its_descent_are_not_redundant():
    """The case time-only crowding got backwards: grinding up a climb and
    bombing the descent 40s later are adjacent in time and opposite in
    every other axis. Both belong in the reel."""
    from gopro_garmin_pipeline.composer import _redundancy

    climb = _seg(1000, speed_mph=6, power_w=300, grade_pct=8)
    descent = _seg(1040, speed_mph=30, power_w=0, grade_pct=-8)
    assert _redundancy(descent, [climb], _TAU) < 0.5


def test_near_identical_clips_still_suppress_each_other():
    """Two 6 mph grinds a second apart are the same footage twice."""
    from gopro_garmin_pipeline.composer import _redundancy

    climb = _seg(1000, speed_mph=6, power_w=300, grade_pct=8)
    twin = _seg(1001, speed_mph=6, power_w=305, grade_pct=8)
    assert _redundancy(twin, [climb], _TAU) > 0.9


def test_no_telemetry_keeps_the_pre_similarity_spacing():
    """A FIT with no altitude has no gradient, so no pair can be compared
    on features. Those pairs must fall through to the LEGACY ride-span
    decay, not to the 120s one — on the short tau, two clips five minutes
    apart score ~0.08 instead of ~0.67 and a whole reel loses its spacing.
    """
    from gopro_garmin_pipeline.composer import (
        Segment, _proximity_crowding, _redundancy)

    def bare(ride_secs):
        return Segment(clip_name="GX010001.MP4", video_start=ride_secs,
                       video_end=ride_secs + 3, ride_time_secs=ride_secs,
                       score=6.0)

    picked = [bare(1000), bare(1500)]
    seg = bare(1300)
    legacy = _proximity_crowding(1300, [1000, 1500], _TAU)
    assert _redundancy(seg, picked, _TAU) == pytest.approx(legacy, abs=1e-9)
    # And it is the wide tau doing the work, not the short one.
    assert _redundancy(seg, picked, _TAU) > 4 * math.exp(-200 / 120)


def test_missing_rubric_does_not_reach_the_telemetry_fallback():
    """A candidate the vision model never scored still has telemetry, so
    it stays on the similarity path — _vis_sim goes neutral, kin_sim does
    not go None."""
    from gopro_garmin_pipeline.composer import _kin_sim, _redundancy

    climb = _seg(1000, speed_mph=6, power_w=300, grade_pct=8, rubric={})
    descent = _seg(1040, speed_mph=30, power_w=0, grade_pct=-8, rubric={})
    assert _kin_sim(climb, descent) is not None
    assert _redundancy(descent, [climb], _TAU) < 0.5


def test_coverage_weight_trades_quality_against_spanning_the_ride():
    """γ=0 chases peak clips; a high γ pulls picks into empty timeline."""
    from gopro_garmin_pipeline.composer import _effective_score

    picked = [_seg(0, speed_mph=18, power_w=180, grade_pct=0)]
    strong_nearby = _seg(300, speed_mph=18, power_w=180, grade_pct=0, score=7.0)
    weak_distant = _seg(5000, speed_mph=18, power_w=180, grade_pct=0, score=6.0)

    assert (_effective_score(strong_nearby, picked, 0.0, _TAU)
            > _effective_score(weak_distant, picked, 0.0, _TAU))
    assert (_effective_score(weak_distant, picked, 3.0, _TAU)
            > _effective_score(strong_nearby, picked, 3.0, _TAU))


# ── Adjacent-duplicate repair ────────────────────────────────────
# Pinned to real values measured on a 20-clip reel: two George Washington
# Bridge shots 167s apart scored 0.914 rubric similarity and read as the
# same clip twice, while the Palisades (0.743) and River Road (0.789)
# pairs are genuinely different shots of one place.

def _vseg(notes, rubric, t, score=7.0, source="gemini"):
    from gopro_garmin_pipeline.composer import Segment
    return Segment(clip_name="GX01.MP4", video_start=0.0, video_end=3.0,
                   ride_time_secs=t, score=score, source=source,
                   label={"notes": notes}, rubric=rubric)


_GWB_A = {"light": 7, "composition": 8, "motion": 5, "scenery": 9, "subject": 7}
_GWB_B = {"light": 6, "composition": 8, "motion": 5, "scenery": 9, "subject": 7}
_PAL_A = {"light": 5, "composition": 7, "motion": 8, "scenery": 8, "subject": 5}
_PAL_B = {"light": 7, "composition": 6, "motion": 5, "scenery": 8, "subject": 7}


def test_same_landmark_same_look_is_adjacent_duplicate():
    from gopro_garmin_pipeline import composer as c
    a = _vseg("Iconic approach beneath the steel tower of the George "
              "Washington Bridge", _GWB_A, 2246.0)
    b = _vseg("Iconic approach of the George Washington Bridge tower under "
              "a clear sky.", _GWB_B, 2413.0)
    assert c._vis_sim(a, b) > 0.85
    assert c._is_adjacent_duplicate(a, b)


def test_same_landmark_different_look_survives():
    """A climb and a descent on one road are two shots, not a repeat."""
    from gopro_garmin_pipeline import composer as c
    a = _vseg("Descending the iconic Henry Hudson Drive past dramatic "
              "Palisades cliffs.", _PAL_A, 2785.0)
    b = _vseg("Closing the gap to draft a friend along the Palisades "
              "cliffs.", _PAL_B, 2900.0)
    assert c._shares_landmark(a, b)
    assert c._vis_sim(a, b) < 0.85
    assert not c._is_adjacent_duplicate(a, b)


def test_landmark_extraction_edge_cases():
    from gopro_garmin_pipeline import composer as c

    def lm(n):
        return c._landmarks(_vseg(n, _GWB_A, 0.0))

    # A landmark survives anywhere but the opening of a sentence, including
    # right after a stripped lead word.
    assert "Palisades" in lm("Fast run past the Palisades overlook.")
    assert "Palisades" in lm("Approaching Palisades overlook.")
    # Acronyms and route refs are landmarks despite being short, and are
    # trusted even sentence-initially — no ordinary word looks like them.
    assert "GWB" in lm("GWB tower under a clear sky.")
    assert "9W" in lm("Riding 9W north past the overlook.")
    # A bare generic noun names nothing on its own.
    assert not lm("Fast descent, River far below.")


def test_route_refs_are_not_truncated_to_their_prefix():
    """Regex alternation is first-match: the route branch must precede the
    bare-acronym one, or every US route collapses into the landmark "US"."""
    from gopro_garmin_pipeline import composer as c
    lm = lambda n: c._landmarks(_vseg(n, _GWB_A, 0.0))  # noqa: E731
    assert lm("Riding US-1 north.") == frozenset({"US-1"})
    assert lm("Fast run down NY-17 today.") == frozenset({"NY-17"})
    assert not c._shares_landmark(_vseg("Riding US-1 north.", _GWB_A, 0.0),
                                  _vseg("Fast on US-9 today.", _GWB_A, 100.0))


def test_sentence_openers_in_a_multi_sentence_note_are_not_landmarks():
    """Every sentence has an opener, not just the first, and a lone
    capitalised verb is indistinguishable from a lone capitalised place."""
    from gopro_garmin_pipeline import composer as c
    lm = lambda n: c._landmarks(_vseg(n, _GWB_A, 0.0))  # noqa: E731
    assert lm("Ends at the Hudson River. Bridge deck ahead.") == frozenset(
        {"Hudson River"})
    assert not c._shares_landmark(
        _vseg("Ends at the pier. Quiet road.", _GWB_A, 0.0),
        _vseg("Ends near the wall. Open sky.", _GWB_A, 100.0))


def test_generic_noun_does_not_conflate_distinct_places():
    from gopro_garmin_pipeline import composer as c
    hudson = _vseg("Crossing the Hudson River.", _GWB_A, 0.0)
    river_rd = _vseg("Shaded climb on River Road.", _GWB_A, 100.0)
    assert not c._shares_landmark(hudson, river_rd)


def test_repair_never_drops_a_must_include():
    from gopro_garmin_pipeline import composer as c
    a = _vseg("Iconic approach beneath the steel tower of the George "
              "Washington Bridge", _GWB_A, 2246.0)
    b = _vseg("Iconic approach of the George Washington Bridge tower under "
              "a clear sky.", _GWB_B, 2413.0)
    b.label["must_include"] = True
    out = c._repair_adjacent_duplicates([a, b], [], 3.0, 1.5, 600.0)
    assert b in out, "a must-include clip must never be removed"
    assert len(out) == 1, "the unprotected clip of the pair goes instead"


def test_repair_drops_rather_than_swapping_in_filler():
    from gopro_garmin_pipeline import composer as c
    a = _vseg("Iconic approach beneath the steel tower of the George "
              "Washington Bridge", _GWB_A, 2246.0)
    b = _vseg("Iconic approach of the George Washington Bridge tower under "
              "a clear sky.", _GWB_B, 2413.0)
    junk = _vseg("Generic empty road with no subject.", _PAL_A, 9000.0,
                 score=1.0)
    out = c._repair_adjacent_duplicates([a, b], [junk], 3.0, 1.5, 600.0)
    assert junk not in out, "below-floor filler must not enter the reel"
    assert len(out) == 1


def test_repair_swaps_in_a_good_unique_clip():
    """A replacement above the reel's quality floor is preferred to a drop."""
    from gopro_garmin_pipeline import composer as c
    a = _vseg("Iconic approach beneath the steel tower of the George "
              "Washington Bridge", _GWB_A, 2246.0)
    b = _vseg("Iconic approach of the George Washington Bridge tower under "
              "a clear sky.", _GWB_B, 2413.0)
    alt = _vseg("Descending the iconic Henry Hudson Drive past dramatic "
                "Palisades cliffs.", _PAL_A, 3000.0, score=7.5)
    out = c._repair_adjacent_duplicates([a, b], [alt], 3.0, 1.5, 600.0)
    assert len(out) == 2
    assert alt in out and b not in out


def test_one_unfixable_pair_does_not_block_other_repairs():
    """A candidate is judged on the adjacencies it would create, not on the
    whole reel — otherwise a locked duplicate pair turns every other swap
    into a drop."""
    from gopro_garmin_pipeline import composer as c
    locked_a = _vseg("Iconic approach of the George Washington Bridge tower.",
                     _GWB_A, 100.0)
    locked_b = _vseg("Steady roll across the George Washington Bridge deck.",
                     _GWB_B, 200.0)
    locked_a.label["must_include"] = True
    locked_b.label["must_include"] = True
    dup_a = _vseg("Climbing onto the Verrazzano Bridge span.", _GWB_A, 400.0)
    dup_b = _vseg("Cresting the Verrazzano Bridge span again.", _GWB_B, 500.0)
    alt = _vseg("Descending past dramatic Palisades cliffs.", _PAL_A, 800.0,
                score=8.0)

    out = c._repair_adjacent_duplicates(
        [locked_a, locked_b, dup_a, dup_b], [alt], 3.0, 1.5, 600.0)

    assert alt in out, "the repairable pair should have been swapped, not dropped"
    assert len(out) == 4
    assert locked_a in out and locked_b in out, "must-includes are untouched"
