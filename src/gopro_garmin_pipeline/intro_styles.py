"""Opening reveal styles for the first clip of a highlight reel.

The opener has one job: stop the scroll in the first second. The original
treatment ramped a gaussian blur to sharp over the whole first clip, which
reads as "the camera was out of focus," not as an intentional cut.

Each style builds an ffmpeg filterchain that transforms the footage *before*
the HUD composites, over a short `reveal` window at the head of the clip.
Everything is timeline-gated (`enable=`) or expression-driven so the filters
pass through untouched — and cost nothing — once the reveal is done.

Conventions:
  - `t` is clip-relative seconds (ffmpeg resets T to 0 after the -ss seek).
  - Sizes are quoted against 2160p and scaled by `_s`, so 1080p and 4K read
    the same.
  - Chains are returned as a list of filterchain strings, ready to join with
    ";" into a filter_complex.

Two ffmpeg constraints shape most of what follows:
  - A timeline-disabled filter emits its FIRST input unchanged. Chains are
    ordered so that "disabled" means "already resolved", never "still
    obscured" — that is what lets the whole graph idle after the reveal.
  - Per-pixel `blend` expressions at 4K cost seconds *per frame*. Anything
    spatial goes through a mask generated at block resolution and scaled up
    nearest-neighbour instead, which is ~100x cheaper for the same image.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .design.tokens import CREAM

# Reveal window (seconds) — how long the footage takes to resolve. Kept
# deliberately short: the title lockup can hold longer over sharp footage,
# but obscured footage past ~2s is where viewers swipe.
DEFAULT_REVEAL_SECS = 1.7

DEFAULT_STYLE = "signal"

# Title-card window: how long the lockup holds and the HUD takes to fade in.
# Independent of the reveal, and the first clip is lengthened to fit it.
DEFAULT_INTRO_SECS = 3.4


@dataclass(frozen=True)
class Geom:
    """Output geometry the filterchains are built against."""

    width: int
    height: int
    fps: float

    def s(self, px: float) -> int:
        """Scale a 2160p-referenced pixel size to this frame height."""
        return max(1, int(round(px * self.height / 2160)))

    def block(self, px: float) -> int:
        """Same, clamped to pixelize's 1024px ceiling."""
        return min(1024, self.s(px))


@dataclass(frozen=True)
class _Style:
    description: str
    build: Callable[[str, str, float, Geom], list[str]]
    # Reveal length when the caller doesn't specify one. None means "span the
    # whole intro window" — only "blur" wants that, and only for continuity
    # with the openers already published.
    reveal: float | None = DEFAULT_REVEAL_SECS


def describe(style: str) -> str:
    return _STYLES[style].description


def default_reveal(style: str, intro_secs: float) -> float:
    """Reveal length when the caller doesn't specify one."""
    reveal = _STYLES[style].reveal
    return intro_secs if reveal is None else min(intro_secs, reveal)


def build_intro_filter(
    style: str,
    in_label: str,
    out_label: str,
    *,
    secs: float,
    width: int,
    height: int,
    fps: float = 59.94,
) -> list[str]:
    """Return filterchain strings taking *in_label* to *out_label*.

    *secs* is the reveal duration. Labels are bracketed ffmpeg stream labels
    (e.g. "[base]", "[introbase]").
    """
    if style not in _STYLES:
        raise ValueError(
            f"unknown intro style {style!r}; expected one of {STYLES}")
    return _STYLES[style].build(
        in_label, out_label, secs, Geom(width, height, fps))


# ── Shared machinery ─────────────────────────────────────────────────

def _mask(label: str, expr: str, *, cols: int, rows: int,
          secs: float, geom: Geom) -> str:
    """A 0/255 mask stream, generated small and scaled up hard-edged.

    *expr* is a geq luma expression over the mask's own X/Y (i.e. block
    coordinates, not pixels) and T. Generating at block resolution is what
    makes per-block logic affordable — the same expression evaluated per
    full-res pixel is orders of magnitude slower.
    """
    return (
        f"color=c=black:s={cols}x{rows}:r={geom.fps:.5f}:d={secs + 0.5:.3f},"
        f"format=gray,geq=lum='{expr}',"
        f"scale={geom.width}:{geom.height}:flags=neighbor{label}"
    )


def _reveal_over(coarse: str, fine: str, mask: str, out: str) -> list[str]:
    """Composite *fine* over *coarse* wherever *mask* is 255.

    Uses alphamerge + overlay rather than the more obvious maskedmerge:
    against the 10-bit full-range GoPro source, maskedmerge's format
    negotiation mangles the merge and returns something close to the fine
    input no matter what the mask says.
    """
    tag = out.strip("[]")
    return [
        f"{fine}format=yuva420p[{tag}_f]",
        f"[{tag}_f]{mask}alphamerge[{tag}_a]",
        f"{coarse}[{tag}_a]overlay=0:0:format=auto{out}",
    ]


def _gate(secs: float) -> str:
    return f":enable='lt(t,{secs:.3f})'"


# ── blur (baseline) ──────────────────────────────────────────────────

def _blur(src: str, dst: str, secs: float, g: Geom) -> list[str]:
    sigma = max(20.0, g.height / 54.0)  # ~40 at 2160p
    gate = _gate(secs)
    # Sharp is the FIRST blend input and the ramp is inverted so that once the
    # blend is disabled it passes sharp straight through. The naive ordering
    # runs the per-pixel expression over the whole clip — ~65s on a 6s 4K clip
    # versus ~20s here.
    return [
        f"{src}split=2[isharp][iblur]",
        f"[iblur]gblur=sigma={sigma:.0f}{gate}[iblurred]",
        f"[isharp][iblurred]blend=all_expr="
        f"'B*(1-clip(T/{secs:.3f},0,1))+A*clip(T/{secs:.3f},0,1)'{gate}{dst}",
    ]


# ── mosaic (Matrix boot) ─────────────────────────────────────────────

# Block size at 2160p, and the fraction of the reveal each step holds. The
# steps accelerate — long hold on the coarsest block so the eye registers
# "this is deliberate," then a fast cascade into detail.
#
# The coarsest step is capped at ~30 rows of blocks: below that the frame is
# genuinely unreadable and the viewer gets no reason to stay. Coarse enough to
# be obviously deliberate, legible enough to show a road and three riders.
_MOSAIC_LADDER = (
    (72, 0.24),
    (40, 0.24),
    (22, 0.20),
    (12, 0.15),
    (6, 0.10),
    (3, 0.07),
)


def _mosaic(src: str, dst: str, secs: float, g: Geom) -> list[str]:
    parts, t = [], 0.0
    for i, (block, frac) in enumerate(_MOSAIC_LADDER):
        t_end = t + frac * secs
        gate = f"lt(t,{t_end:.3f})" if i == 0 else f"between(t,{t:.3f},{t_end:.3f})"
        parts.append(f"pixelize=w={g.block(block)}:h={g.block(block)}:enable='{gate}'")
        t = t_end
    # Cream-ward lift while blocky, resolving to neutral — sells "booting up"
    # rather than "low bitrate."
    parts.append(
        f"eq=saturation='1-0.45*(1-clip(t/{secs:.3f},0,1))'"
        f":contrast='1+0.18*(1-clip(t/{secs:.3f},0,1))':eval=frame{_gate(secs)}"
    )
    return [src + ",".join(parts) + dst]


# ── scan (digital wipe) ──────────────────────────────────────────────

def _scan(src: str, dst: str, secs: float, g: Geom) -> list[str]:
    block = g.s(40)
    bar = g.s(10)
    glow = g.s(90)
    gate = _gate(secs)
    # Mask rows are quantised to 4px, which is invisible on the moving edge
    # and keeps the mask ~500 rows instead of 2160.
    rows = max(2, g.height // 4)
    pos = f"min(1,t/{secs:.3f})*{g.height}"
    chains = [f"{src}split=2[scsharp][scpix]"]
    chains.append(
        f"[scpix]pixelize=w={block}:h={block}{gate},"
        f"eq=saturation=0.30:contrast=1.12:brightness=-0.06{gate}[scpx]"
    )
    # Everything above the sweeping line has resolved; below it is still
    # blocky. Hard edge — that is what makes it read as a wipe rather than a
    # dissolve.
    chains.append(_mask("[scm]", f"if(lt(Y,clip(T/{secs:.3f},0,1)*{rows}),255,0)",
                        cols=2, rows=rows, secs=secs, geom=g))
    chains += _reveal_over("[scpx]", "[scsharp]", "[scm]", "[scwiped]")
    # The scan line rides the same edge. drawbox looks like the obvious tool
    # but has no time variable (its `t` is thickness), so an animated y is
    # silently frozen at its t=0 value; overlay does evaluate per frame.
    cream = "0x%02X%02X%02X" % CREAM
    chains.append(
        f"color=c=white:s={g.width}x{glow}:r={g.fps:.5f}:d={secs + 0.5:.3f},"
        f"format=yuva420p,colorchannelmixer=aa=0.10[scglow]"
    )
    chains.append(
        f"color=c={cream}:s={g.width}x{bar}:r={g.fps:.5f}:d={secs + 0.5:.3f},"
        f"format=yuva420p,colorchannelmixer=aa=0.85[scbar]"
    )
    chains.append(
        f"[scwiped][scglow]overlay=0:'{pos}-{glow}':eval=frame{gate}[scg]")
    chains.append(
        f"[scg][scbar]overlay=0:'{pos}-{bar}':eval=frame{gate}{dst}")
    return chains


# ── signal (RGB split lock) ──────────────────────────────────────────

# Block size at 2160p per resolve level, and the window (as a fraction of the
# reveal) over which blocks at that level hand off to the next one down. The
# windows deliberately OVERLAP: at any instant three different block sizes are
# on screen at once, which is what keeps it from reading as a global step.
_SIGNAL_LEVELS = (
    (76, 0.00, 0.55),
    (36, 0.20, 0.80),
    (15, 0.42, 1.00),
)

# Seeded so a given ride renders the same opener every time. Bumping this
# reshuffles which blocks resolve first and the chromatic jitter pattern.
_SIGNAL_SEED = 1337


def _hash_expr(seed: int) -> str:
    """Deterministic per-block noise in [0,1) from block coords X, Y.

    The sin/fract hash — stable across frames, unlike expr's random(), which
    re-rolls every evaluation and strobes.
    """
    return f"abs(mod(sin(X*12.9898+Y*78.233+{seed})*43758.5453,1))"


def _signal_shift_schedule(secs: float, g: Geom) -> list[tuple[float, float, int, int]]:
    """Uneven (start, end, horizontal shift, vertical green shift) windows.

    An even ladder of stops reads mechanical — like an aperture ticking. This
    keeps the decay envelope but jitters both the step durations and the
    magnitudes around it, and lets the split occasionally kick back out.
    """
    rng = random.Random(_SIGNAL_SEED)
    peak = g.s(52)
    spans, total = [], 0.0
    while total < 1.0:
        span = rng.uniform(0.035, 0.135)
        spans.append(span)
        total += span
    spans = [s / total for s in spans]  # normalise to the reveal window

    out, t = [], 0.0
    for span in spans:
        t_end = min(1.0, t + span)
        # Envelope: exponential decay across the reveal, jittered hard. The
        # occasional >1 multiplier is the kick-back-out.
        mag = peak * (1.0 - (t ** 0.65)) * rng.uniform(0.25, 1.55)
        shift = int(round(mag))
        if rng.random() < 0.28:
            shift = -shift  # flip which channel leads
        gv = int(round(abs(mag) * rng.uniform(0.0, 0.45)))
        out.append((t * secs, t_end * secs, shift, gv))
        t = t_end
    return out


def _signal(src: str, dst: str, secs: float, g: Geom) -> list[str]:
    rng = random.Random(_SIGNAL_SEED + 1)
    gate = _gate(secs)

    # ── Spatially staggered mosaic resolve ───────────────────────────
    # One pixelize branch per level plus the sharp copy, composited pairwise
    # through a per-block mask: a block flips to the next-finer size as soon
    # as the ramp passes its own hash value, so they scatter in independently
    # rather than the whole frame stepping at once.
    n = len(_SIGNAL_LEVELS)
    chains = [src + f"split={n + 1}"
              + "".join(f"[sg{i}]" for i in range(n)) + "[sgsharp]"]
    for i, (block, _, _) in enumerate(_SIGNAL_LEVELS):
        b = g.block(block)
        chains.append(f"[sg{i}]pixelize=w={b}:h={b}{gate}[sgp{i}]")

    for i, (block, f_start, f_end) in enumerate(_SIGNAL_LEVELS):
        t0, t1 = f_start * secs, f_end * secs
        b = g.block(block)
        # p ramps 0→1 across this level's window; a block flips as soon as p
        # passes its own hash value.
        p = f"clip((T-{t0:.3f})/{max(t1 - t0, 1e-3):.3f},0,1)"
        chains.append(_mask(
            f"[sgm{i}]",
            f"if(lt({_hash_expr(_SIGNAL_SEED + 7 * i)},{p}),255,0)",
            cols=max(1, g.width // b), rows=max(1, g.height // b),
            secs=secs, geom=g,
        ))
        chains += _reveal_over(
            f"[sgp{i}]" if i == 0 else f"[sgmix{i - 1}]",
            f"[sgp{i + 1}]" if i + 1 < n else "[sgsharp]",
            f"[sgm{i}]",
            f"[sgmix{i}]" if i + 1 < n else "[sgmosaic]",
        )

    # ── Jittered chromatic split + static bursts ─────────────────────
    parts = [
        f"rgbashift=rh={-shift}:bh={shift}:gv={gv}"
        f":enable='between(t,{t0:.3f},{t1:.3f})'"
        for t0, t1, shift, gv in _signal_shift_schedule(secs, g)
        if shift or gv
    ]
    # Static in a few short bursts rather than one continuous block — the
    # gaps are what make it read as a signal fighting to lock.
    for _ in range(3):
        t0 = rng.uniform(0.0, secs * 0.62)
        parts.append(
            f"noise=alls={rng.randint(14, 30)}:allf=t"
            f":enable='between(t,{t0:.3f},{t0 + rng.uniform(0.03, 0.08):.3f})'"
        )
    chains.append("[sgmosaic]" + ",".join(parts) + dst)
    return chains


# ── punch (no-pixel control) ─────────────────────────────────────────

def _punch(src: str, dst: str, secs: float, g: Geom) -> list[str]:
    # z(t) eases 1.26 → 1.0 on a cubic, so most of the move happens early and
    # it *settles* rather than drifting. crop can't animate w/h (its size
    # expressions are evaluated once), so this goes through zoompan, which is
    # the only filter that re-solves geometry per frame.
    z = f"(1+0.26*pow(1-min(it/{secs:.3f},1),3))"
    return [
        f"{src}zoompan=z='{z}':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":s={g.width}x{g.height}:fps={g.fps:.5f},setsar=1{dst}",
    ]


_STYLES: dict[str, _Style] = {
    "blur": _Style(
        "Baseline: gaussian blur crossfading to sharp (the original opener).",
        _blur, reveal=None),
    "mosaic": _Style(
        "Matrix boot — pixel blocks halve in discrete steps until sharp.",
        _mosaic),
    "scan": _Style(
        "Digital wipe — a bright scan bar sweeps down, leaving sharp footage behind.",
        _scan),
    "signal": _Style(
        "Signal lock — per-block mosaic resolve + RGB split converging into register.",
        _signal),
    "punch": _Style(
        "Punch-in snap — starts pushed in, eases out to frame. No pixels.",
        _punch),
}

STYLES = tuple(_STYLES)
