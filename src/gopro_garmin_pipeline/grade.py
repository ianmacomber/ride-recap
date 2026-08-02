"""Two-layer colour grading: per-shot correction, then a shared look.

Pass 1 (correct) measures each shot on its own and neutralises it: black/white
anchors from the luma histogram, neutral-pixel white balance (luma-normalised),
and a highlight-safe gamma nudge toward a shared middle. Pass 2 (look) is one
shared creative recipe applied identically to every shot, scaled by a strength
factor.

This assumes the camera shot with white balance LOCKED (e.g. 5500K on a GoPro),
so it did not balance anything per frame. That is the right way to shoot for a
graded cut — every clip carries one known transform instead of a live,
unrecorded, drifting one — but it means the correction has to happen here.
Light changes over a multi-hour ride, so the correction is per-shot, never
one grade over the whole ride.

    from .grade import measure_shot, build_filter
    bal = measure_shot("GX010406.MP4", at_secs=812.0)
    vf  = build_filter(bal, look="house", strength=0.35)
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

# WB gains beyond this are almost never a real cast — they are a scene that
# is genuinely dominated by one colour (a wall of summer canopy, a glass tower
# against sky). Clamping keeps the correction from eating the subject. The gains
# are measured on near-neutral pixels only (asphalt, concrete, cloud) and applied
# luma-normalised, so the clamp can sit tight.
_WB_CLAMP = (0.93, 1.07)
_NEUTRAL_CHROMA = 0.12   # max (channel spread) for a pixel to count as neutral
_NEUTRAL_MIN = 500       # fall back to the full midtone mask below this many
_MID_TARGET = 0.40   # median luma the exposure nudge aims for
_EV_DAMP = 0.6       # correct only this fraction of the way to the target
_EV_CLAMP = 0.6      # stops
_LUMA = (0.2126, 0.7152, 0.0722)

# Recipes are written at full strength and land scaled by `strength`. They lean on
# vibrance rather than saturation (asphalt and skin stay put while canopy separates)
# and pull highlights down rather than pushing exposure up, because a blown sky is
# the one thing in this footage that cannot be recovered.
LOOKS: dict[str, dict[str, float]] = {
    "none": {},
    "house": {"contrast": 16, "vib": 20, "highlights": -12, "shadows": 5},
    "warm-afternoon": {"temp": 22, "tint": 4, "exposure": 0.10, "contrast": 12,
                       "vib": 16, "highlights": -14},
    "cool-morning": {"temp": -16, "tint": -4, "contrast": 14, "vib": 18, "shadows": -5},
    "soft-film": {"fade": 24, "contrast": 12, "sat": -8, "temp": 7, "shadows": 9},
    "overcast-lift": {"exposure": 0.22, "contrast": 18, "vib": 26, "temp": 10,
                      "shadows": 13, "highlights": -18},
}


@dataclass(frozen=True)
class ShotBalance:
    """Per-shot correction, median-combined from a few frames around the anchor."""
    b: float    # black anchor (luma)
    w: float    # white anchor (luma)
    gr: float   # neutral-pixel WB gain applied to red
    gb: float   # neutral-pixel WB gain applied to blue
    ev: float   # exposure correction, stops
    med: float = 0.35   # median luma the ev was measured against (drives the gamma)


NEUTRAL = ShotBalance(b=0.0, w=1.0, gr=1.0, gb=1.0, ev=0.0)


def _measure_frame(video_path: Path, at_secs: float):
    """Stats for one frame: (b, w, gr, gb, med), or None if it cannot be decoded."""
    import io

    import numpy as np
    from PIL import Image

    proc = subprocess.run(
        ["ffmpeg", "-v", "error",
         "-ss", f"{max(0.0, at_secs):.3f}", "-i", str(video_path),
         "-frames:v", "1", "-vf", "scale=640:-1",
         "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
        capture_output=True, check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None

    im = np.asarray(Image.open(io.BytesIO(proc.stdout)).convert("RGB"),
                    dtype=np.float32) / 255.0
    luma = np.array(_LUMA, dtype=np.float32)

    lum = im @ luma
    b = float(np.percentile(lum, 0.5))
    w = float(max(np.percentile(lum, 99.7), b + 0.05))

    lv = np.clip((im - b) / (w - b), 0, 1)
    l2 = lv @ luma
    mid = (l2 > 0.15) & (l2 < 0.85)
    if not mid.any():           # near-black or blown frame: skip WB, keep levels
        return b, w, 1.0, 1.0, 0.35

    # WB is measured on near-neutral pixels only (asphalt, concrete, cloud).
    # Raw grey-world over the midtones reads a wall of canopy as a green cast
    # and slams both gains into the clamp, which is a brightness lift and a
    # magenta shift, not a correction.
    spread = lv.max(axis=2) - lv.min(axis=2)
    neutral = mid & (spread < _NEUTRAL_CHROMA)
    sample = lv[neutral] if int(neutral.sum()) >= _NEUTRAL_MIN else lv[mid]
    means = sample.mean(axis=0)
    gr = float(np.clip(means[1] / max(means[0], 1e-4), *_WB_CLAMP))
    gb = float(np.clip(means[1] / max(means[2], 1e-4), *_WB_CLAMP))

    med = float(np.median(np.clip(lv * [gr, 1.0, gb], 0, 1) @ luma))
    return b, w, gr, gb, med


def measure_shot(video_path: str | Path, at_secs: float = 0.0) -> ShotBalance:
    """Measure `video_path` around `at_secs` and return its correction.

    Samples three frames (anchor ±1s) and median-combines the stats, so one
    moment of shade or a passing truck cannot decide the grade for the whole cut.

    Reads from the master file rather than a low-res proxy: the proxy is 8-bit
    H.264 and would hide the shadow noise and banding the correction has to
    respect. Falls back to NEUTRAL if no frame can be decoded, so a bad seek
    degrades to "ungraded" rather than killing the burn.
    """
    import numpy as np

    times = [t for t in (at_secs - 1.0, at_secs, at_secs + 1.0) if t >= 0.0]
    stats = [s for s in (_measure_frame(Path(video_path), t) for t in times)
             if s is not None]
    if not stats:
        print(f"  Warning: grade could not sample {Path(video_path).name} "
              f"@{at_secs:.1f}s — leaving this shot uncorrected")
        return NEUTRAL

    b, w, gr, gb, med = (float(np.median(v)) for v in zip(*stats))
    ev = float(np.clip(_EV_DAMP * np.log2(_MID_TARGET / max(med, 0.05)),
                       -_EV_CLAMP, _EV_CLAMP))
    return ShotBalance(round(b, 4), round(w, 4), round(gr, 4), round(gb, 4),
                       round(ev, 3), round(med, 4))


def scaled_look(look: str, strength: float) -> dict[str, float]:
    """The look recipe with every term scaled. Each term is written so 0 is identity,
    so one multiplier interpolates the whole pass from untouched to full."""
    recipe = LOOKS.get(look.lower().replace(" ", "-"))
    if recipe is None:
        raise ValueError(f"unknown look {look!r} — choose from {', '.join(LOOKS)}")
    return {k: v * strength for k, v in recipe.items()}


def build_filter(
    balance: ShotBalance | None = None,
    look: str = "none",
    strength: float = 0.35,
) -> str:
    """Build the ffmpeg filter chain for one shot. Empty string means "no grade".

    Args:
        balance: this shot's correction from measure_shot(), or None to skip
            the per-shot pass entirely (look only).
        look: key into LOOKS.
        strength: 0..1 multiplier over the whole look pass.
    """
    parts: list[str] = []
    L = scaled_look(look, strength)

    # Levels and white balance are two separate filters on purpose.
    #
    # The obvious-looking trick is to fold the WB gains into colorlevels by
    # scaling each channel's input max (imax_c = b + (w-b)/gain_c). That is
    # algebraically right but breaks in practice: colorlevels requires every
    # imax in [0,1], and a gain below 1 on a shot whose white anchor is near
    # 1.0 pushes imax above 1.0, so ffmpeg aborts with "Result too large".
    # Shots with clipped highlights (w == 1.000) hit this routinely.
    #
    # colorchannelmixer applies per-channel gain directly and takes values
    # above 1, so it carries the WB and colorlevels does only the stretch —
    # where all three channels share the same in-range anchors.
    if balance is not None:
        b = min(max(balance.b, 0.0), 0.94)
        w = min(max(balance.w, b + 0.05), 1.0)
        # Stretch only partway to the anchor, and not at all when the frame
        # already reaches near-white: the anchor is one percentile of a few
        # frames, and a full stretch on a hazy shot burns the sky for a
        # contrast gain nobody asked for.
        if w < 0.985:
            w = w + (1.0 - w) * 0.4
        else:
            w = 1.0
        if b > 5e-4 or w < 1 - 5e-4:
            # Pin a concrete RGB format first: ffmpeg 8.1 lets colorlevels
            # negotiate a format on 10-bit full-range masters that renders
            # solid black whenever a black-point (imin) is set.
            parts.append("format=gbrp16le")
            parts.append(
                f"colorlevels=rimin={b:.3f}:gimin={b:.3f}:bimin={b:.3f}"
                f":rimax={w:.3f}:gimax={w:.3f}:bimax={w:.3f}"
            )
        if abs(balance.gr - 1.0) > 5e-4 or abs(balance.gb - 1.0) > 5e-4:
            # Luma-normalised: WB may only trade colour, never buy brightness.
            # Without this, a green-heavy scene that pushes both gains up
            # applies a hidden exposure lift on top of the ev pass and clips
            # the sky.
            gr, gb = balance.gr, balance.gb
            norm = _LUMA[0] * gr + _LUMA[1] + _LUMA[2] * gb
            parts.append(
                f"colorchannelmixer=rr={gr / norm:.4f}:gg={1 / norm:.4f}"
                f":bb={gb / norm:.4f}"
            )

        # The shot's own exposure correction lands as a midtone gamma, which
        # pins black and white in place: a linear `exposure` gain on a
        # +0.4-stop shot clips every sky pixel above ~0.75, and a blown sky is
        # the one thing this footage cannot recover. The look's exposure term
        # stays linear below — looks are authored knowing that.
        if abs(balance.ev) > 0.02 and 0.05 < balance.med < 0.9:
            import math
            target = min(max(balance.med * 2 ** balance.ev, 0.05), 0.9)
            gamma = math.log(balance.med) / math.log(target)
            gamma = min(max(gamma, 0.75), 1.3)
            if abs(gamma - 1.0) > 0.01:
                parts.append(f"eq=gamma={gamma:.3f}")

    ev = L.get("exposure", 0.0)
    if abs(ev) > 0.001:
        parts.append(f"exposure=exposure={ev:.2f}")
    if L.get("temp"):
        parts.append(f"colortemperature=temperature={round(6500 - L['temp'] * 20)}")

    eq = []
    if L.get("contrast"):
        eq.append(f"contrast={1 + L['contrast'] / 100 * 0.85:.2f}")
    if L.get("sat"):
        eq.append(f"saturation={1 + L['sat'] / 100:.2f}")
    if eq:
        parts.append("eq=" + ":".join(eq))

    if L.get("vib"):
        parts.append(f"vibrance=intensity={L['vib'] / 100:.2f}")

    if L.get("shadows") or L.get("highlights") or L.get("fade"):
        a = L.get("fade", 0.0) / 100 * 0.07
        s = min(1.0, max(0.0, 0.25 + L.get("shadows", 0.0) / 100 * 0.09))
        h = min(1.0, max(0.0, 0.80 + L.get("highlights", 0.0) / 100 * 0.08))
        parts.append(f"curves=all='0/{a:.3f} 0.25/{s:.3f} 0.80/{h:.3f} 1/1'")

    return ",".join(parts)
