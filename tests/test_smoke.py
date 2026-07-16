"""Smoke tests — no video, no FIT file, no network, no API keys.

These cover the pieces a contributor is most likely to break without noticing:
the config derivation, the design-token contract, and the ride-geometry
classifier. Everything here runs in well under a second.
"""

from __future__ import annotations

import math

import pytest

from gopro_garmin_pipeline.config import Settings
from gopro_garmin_pipeline.design import tokens as T
from gopro_garmin_pipeline import route_metadata as rm


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
