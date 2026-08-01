"""Pure-math tests for the colour grade filter builder (no ffmpeg needed)."""

import pytest

from gopro_garmin_pipeline.grade import LOOKS, NEUTRAL, ShotBalance, build_filter, scaled_look


def test_no_grade_is_empty_string():
    assert build_filter(None, look="none") == ""


def test_neutral_balance_is_identity():
    assert build_filter(NEUTRAL, look="none") == ""


def test_correction_pins_format_before_colorlevels():
    # ffmpeg 8.1 renders solid black if colorlevels sets a black point on
    # 10-bit full-range input without a pinned format — the pin must survive.
    bal = ShotBalance(b=0.02, w=0.95, gr=1.0, gb=1.0, ev=0.0, med=0.4)
    vf = build_filter(bal, look="none")
    assert "colorlevels" in vf
    assert vf.index("format=gbrp16le") < vf.index("colorlevels")


def test_wb_gains_are_luma_normalised():
    # WB may shift colour but never brightness: the luma dot-product of the
    # applied gains must be ~1 regardless of the measured gains.
    bal = ShotBalance(b=0.0, w=1.0, gr=1.05, gb=1.05, ev=0.0, med=0.4)
    vf = build_filter(bal, look="none")
    parts = dict(kv.split("=") for kv in
                 vf.split("colorchannelmixer=")[1].split(",")[0].split(":"))
    luma = 0.2126 * float(parts["rr"]) + 0.7152 * float(parts["gg"]) \
        + 0.0722 * float(parts["bb"])
    assert abs(luma - 1.0) < 1e-3


def test_exposure_correction_is_gamma_not_linear():
    # The per-shot ev lands as midtone gamma (pins black/white); linear
    # exposure is reserved for looks.
    bal = ShotBalance(b=0.0, w=1.0, gr=1.0, gb=1.0, ev=0.4, med=0.25)
    vf = build_filter(bal, look="none")
    assert "eq=gamma=" in vf
    assert "exposure" not in vf


def test_look_scales_with_strength():
    weak = build_filter(None, look="house", strength=0.2)
    strong = build_filter(None, look="house", strength=0.8)
    assert weak != strong
    assert scaled_look("house", 0.0) == {k: 0.0 for k in LOOKS["house"]}


def test_unknown_look_raises():
    with pytest.raises(ValueError):
        build_filter(None, look="teal-and-orange")
