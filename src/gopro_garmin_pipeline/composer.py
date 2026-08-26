"""Compose highlight videos from ride telemetry, Strava segments, and optional labels.

Multi-source candidate generation pipeline:
  1. FIT telemetry highlights — power spikes, speed spikes, HR spikes,
     climbs, sprints detected from Garmin data alone (highlights.py).
  2. Strava segments — popular segments on the route, scored via
     log(star_count) + telemetry at midpoint.
  3. Vision model — sparse frame scan of entire ride for visual interest.
  4. Manual labels — optional enrichment from ride_labels.json.

All sources are fused, deduplicated (nearby candidates merged), capped
at ~30, and ranked. Selected candidates get overlays burned and are
concatenated into landscape (16:9) and portrait (9:16) highlight videos.
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from . import intro_styles
from .fit_parser import RideData
from .models import VISION_SOURCES
from .sync import SyncedClip
from .utils import MS_TO_MPH, normalize_label_scale, rating_visual_action

if TYPE_CHECKING:
    from .burn_overlay import OverlayRenderer  # noqa: F401 — annotations only

# Maximum candidates after fusion (before user review). Raised 60→90 so the
# narrative/greedy selectors have a deeper, better-spread pool to choose from
# and aren't forced to reach for crowded filler on dense rides.
_CANDIDATE_CAP = 90

# Narrative selection prompt version (see prompts/narrative_select/)
_NARRATIVE_PROMPT_VERSION = "v3"

# Diversity-aware greedy tunables. Two picks near each other in ride time
# get a proximity penalty — keeping the second only if its score is strong
# enough to outpay the crowding cost. There is a hard floor at
# ``segment_duration`` (two cuts closer than that would share video frames
# and burn as overlapping windows), but the rest of the tradeoff is smooth.
_PROXIMITY_LAMBDA = 6.0       # weight on the crowding penalty
_PROXIMITY_CROWDING_CAP = 2.0  # cap the sum so one big cluster can't lock out
                               # the whole neighborhood. Past two near-by picks
                               # the penalty stops growing.
_LABEL_BOOST = 6.0            # editorial-voice boost for strong labels
_LABEL_MIN_RATING = 6         # min visual/action rating that earns the boost

# ── Similarity-aware redundancy ──────────────────────────────
# Two clips are redundant only when they are close in ride time AND alike
# in what they show. Time alone is a bad proxy: a rider grinds up a climb
# (6 mph / 300 W / +8%) and bombs the descent 40s later (30 mph / 0 W /
# -8%) — adjacent in time, opposite in every other axis, and narratively
# the most interesting pair on the ride. Conversely two 19 mph cruises
# down the same road ARE redundant however far apart they sit.
#
#   redundancy = exp(-Δt/τ) × (W_KIN·kin_sim + W_VIS·vis_sim)
#
# τ is a fixed short constant (not ride-span derived): past a couple of
# minutes, footage is simply a different part of the ride.
_REDUNDANCY_TAU = 120.0        # seconds — temporal decay of redundancy
_REDUNDANCY_W_KIN = 0.65       # weight on telemetry (speed/power/grade)
_REDUNDANCY_W_VIS = 0.35       # weight on the vision-model rubric axes
# Scale denominators: the difference that counts as "meaningfully unalike".
_KIN_SCALE_SPEED_MPH = 12.0
_KIN_SCALE_POWER_W = 150.0
_KIN_SCALE_GRADE_PCT = 6.0
_VIS_SHARPNESS = 2.0           # higher = rubric differences separate faster
_VIS_SIM_UNKNOWN = 0.5         # neutral when a candidate has no rubric

# Coverage term: rewards picks that open up the biggest untouched stretch
# of the timeline, so a chronological reel still spans the whole ride.
# Traded off directly against quality — raise it for coverage, lower it
# for peak clips. Tuned on a 4h11m ride carrying 82 minutes of footage.
_COVERAGE_GAP_CAP = 600.0      # seconds — gap at which the bonus saturates

# Hard floor between two cuts. Above segment_duration (which only prevents
# literal frame sharing) because near-identical anchors a few seconds apart
# are never both worth having; similarity handles the rest smoothly.
_MIN_CUT_GAP_SECS = 8.0

# Greedy quality phase: the eff score (base + label boost − crowding) drives
# WHICH clips are picked first. Once the best remaining clip is net-negative
# (crowded/dead), the quality phase stops and coverage-fill takes over to
# guarantee the full clip count — so eff is a quality *ordering* signal, not
# a hard cap on how many clips the reel gets.
_GREEDY_MIN_MARGINAL = 0.0     # end the quality phase once best marginal eff < this

# Coverage fill: after quality picks, guarantee the exact slot count by
# filling the biggest remaining holes in the timeline. Ranks fillers by
# gap-distance blended with score, so big holes are filled by the liveliest
# available clip (dead clips are already score-damped by _liveness_penalty).
_FILL_QUALITY_WEIGHT = 30.0    # legacy blend weight — superseded by
                               # _FILL_COVERAGE_MULTIPLIER, kept for reference
# Coverage-fill runs the same ranking as the quality phase with a heavier
# coverage weight — its job is to span the ride, but score and redundancy
# still count, so it can no longer drop a generic clip into a gap that a
# genuinely good one could fill.
_FILL_COVERAGE_MULTIPLIER = 2.0

# Telemetry × vision liveness. Gemini scores composition, blind to effort,
# so a scenic frame shot while soft-pedalling to a near-stop (5-7 mph at
# 0 W) can outscore real riding. A clip is "dead" only when speed AND power
# are BOTH low — a coasting descent (fast, no power) or a grinding climb
# (slow, high power) is not. Soft, product-scaled score penalty; the
# min_motion_mph gate already removes true stops.
_LIVENESS_SPEED_LOW = 8.0     # mph — at/below, speed fully "dead-slow"
_LIVENESS_SPEED_OK = 14.0     # mph — at/above, no speed-based deadness
_LIVENESS_POWER_LOW = 30.0    # W — at/below, fully "coasting"
_LIVENESS_POWER_OK = 130.0    # W — at/above, clearly working
_LIVENESS_MAX_PENALTY = 4.0   # score points subtracted for a fully-dead clip
_LIVENESS_WINDOW_HALF_SECS = 3.0


def _proximity_crowding(t: float, picked_ts: list[float], tau: float) -> float:
    """Sum of exponential proximity weights against already-picked clips.

    crowding = min(Σ exp(-|t - t_i| / tau), _PROXIMITY_CROWDING_CAP).
    Adjacent picks contribute ~1.0, far-apart picks contribute ~0.
    Capped so a dense cluster of must_includes can't lock the greedy
    out of an entire region of the ride.

    Time-only fallback, kept for candidates with no telemetry features
    (a ride with no power meter, or a FIT gap over the anchor). The
    similarity-aware path in ``_redundancy`` supersedes it whenever both
    candidates carry features.
    """
    if not picked_ts:
        return 0.0
    raw = sum(math.exp(-abs(t - tp) / tau) for tp in picked_ts)
    return min(raw, _PROXIMITY_CROWDING_CAP)


def _kin_sim(a: "Segment", b: "Segment") -> float | None:
    """Telemetry similarity in [0, 1]. None when either lacks features.

    Distance over (speed, power, grade), each normalised by the spread
    that reads as "a different kind of riding". Climb-vs-descent lands
    near 0; two cruises at the same speed land near 1.
    """
    fa, fb = a.telemetry_features, b.telemetry_features
    if not fa or not fb:
        return None
    try:
        d = math.sqrt(
            ((fa["speed_mph"] - fb["speed_mph"]) / _KIN_SCALE_SPEED_MPH) ** 2
            + ((fa["power_w"] - fb["power_w"]) / _KIN_SCALE_POWER_W) ** 2
            + ((fa["grade_pct"] - fb["grade_pct"]) / _KIN_SCALE_GRADE_PCT) ** 2
        )
    except KeyError:
        return None
    return math.exp(-d)


def _vis_sim(a: "Segment", b: "Segment") -> float:
    """Rubric similarity in [0, 1] over the vision model's visual axes.

    Neutral (_VIS_SIM_UNKNOWN) when either side was never rubric-scored,
    so a missing rubric neither excuses nor condemns a candidate.
    """
    ra, rb = a.rubric or {}, b.rubric or {}
    if not ra or not rb:
        return _VIS_SIM_UNKNOWN
    axes = set(ra) & set(rb)
    if not axes:
        return _VIS_SIM_UNKNOWN
    d = math.sqrt(
        sum((float(ra[k]) - float(rb[k])) ** 2 for k in axes) / len(axes)
    ) / 10.0
    return math.exp(-_VIS_SHARPNESS * d)


def _redundancy(seg: "Segment", picked: list["Segment"]) -> float:
    """How much ``seg`` duplicates what is already selected, in [0, cap].

    Per already-picked clip: temporal closeness × feature similarity.
    Summed and capped like the legacy crowding term, so one dense cluster
    of must-includes cannot lock the greedy out of a whole region.
    """
    if not picked:
        return 0.0
    total = 0.0
    for p in picked:
        dt = abs(seg.ride_time_secs - p.ride_time_secs)
        temporal = math.exp(-dt / _REDUNDANCY_TAU)
        if temporal < 1e-3:
            continue  # far enough away that nothing else matters
        kin = _kin_sim(seg, p)
        if kin is None:
            # No telemetry to compare — fall back to pure time overlap so
            # near-duplicates are still suppressed on power-meter-less rides.
            similarity = 1.0
        else:
            similarity = (
                _REDUNDANCY_W_KIN * kin + _REDUNDANCY_W_VIS * _vis_sim(seg, p)
            )
        total += temporal * similarity
    return min(total, _PROXIMITY_CROWDING_CAP)


def _coverage_bonus(seg: "Segment", picked: list["Segment"]) -> float:
    """[0, 1] — how much untouched timeline this pick opens up.

    Distance to the nearest already-picked clip, saturating at
    _COVERAGE_GAP_CAP. Multiplied by config.coverage_weight at the call
    site, which is what trades peak quality against spanning the ride.
    """
    if not picked:
        return 1.0
    nearest = min(abs(seg.ride_time_secs - p.ride_time_secs) for p in picked)
    return min(nearest, _COVERAGE_GAP_CAP) / _COVERAGE_GAP_CAP


def _effective_score(
    seg: "Segment", picked: list["Segment"], coverage_weight: float,
) -> float:
    """The single ranking function used by every selection stage.

        eff = score + label_boost − λ·redundancy + γ·coverage

    Quality and coverage-fill differ only in ``coverage_weight`` — there
    is no stage where score stops mattering.
    """
    return (
        seg.score
        + _label_boost(seg)
        - _PROXIMITY_LAMBDA * _redundancy(seg, picked)
        + coverage_weight * _coverage_bonus(seg, picked)
    )


def _diversity_tau(ride_times: list[float], n_clips: int) -> float:
    """Characteristic decay distance for the proximity penalty.

    Roughly the average gap between picks at full budget. A pick exactly
    at this distance contributes 1/e ≈ 0.37 of the crowding it would at
    zero distance.
    """
    if not ride_times or n_clips <= 0:
        return 60.0
    span = max(ride_times) - min(ride_times)
    return max(60.0, span / max(n_clips, 1))


def _label_boost(seg: "Segment") -> float:
    """Editorial boost for strongly-rated labels — applied wherever
    candidates compete on score (narrative filter AND greedy loop)."""
    if "label" not in (seg.sources or []):
        return 0.0
    v = seg.label.get("visual", 0)
    a = seg.label.get("action", 0)
    return _LABEL_BOOST if max(v, a) >= _LABEL_MIN_RATING else 0.0


@dataclass
class Segment:
    """A video segment candidate, scored for inclusion in the highlight reel.

    [video_start, video_end] is the *review span* — the window the user
    sees in the reviewer preview. anchor_video_secs is the *cut point* —
    where compose-selected centers its tight final clip. These can differ:
    a Gemini-rated clip might span 12s for review but cut a 3s clip
    around its midpoint anchor.

    stable_id derives from (clip_stem, anchor_video_secs at 1-second
    precision) so the same moment keeps the same identity across
    pipeline reruns. Reviewer state (selections, ratings, preview cache)
    keys off this — small re-anchor jitter from rescoring won't lose
    user input.
    """

    clip_name: str
    video_start: float
    video_end: float
    ride_time_secs: float
    anchor_video_secs: float = 0.0  # where compose-selected cuts around
    score: float = 0.0
    source: str = ""  # "telemetry", "strava", "gemini", "label"
    sources: list[str] = field(default_factory=list)  # all contributing sources
    label: dict = field(repr=False, default_factory=dict)
    portrait_crop_bias: float = 0.0  # -1.0=left, 0.0=center, +1.0=right
    rubric: dict = field(default_factory=dict)  # multi-dim scores from Gemini
    # {speed_mph, power_w, grade_pct} sampled around the anchor. Drives the
    # similarity-aware redundancy term. Empty when telemetry is unavailable.
    telemetry_features: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.video_end - self.video_start

    @property
    def stable_id(self) -> str:
        """Identity that survives pipeline reruns and re-anchoring jitter."""
        return stable_id_for(self.clip_name, self.anchor_video_secs)


def stable_id_for(clip_name: str, anchor_video_secs: float) -> str:
    """Build a Segment-style stable_id from raw fields (e.g. on a dict)."""
    stem = Path(clip_name).stem
    return f"{stem}@{int(round(anchor_video_secs))}"


@dataclass
class ComposerConfig:
    landscape_duration: float = 60.0
    portrait_duration: float = 30.0
    segment_duration: float = 3.0
    offset: float = 0.0
    ride_timezone: str | None = None
    ride_timezone_explicit: bool = False
    # Optional per-clip offsets keyed by GoPro filename. When present,
    # the burning code looks up each clip's value here and falls back to
    # ``offset`` for unlisted clips. Fixes multi-recording rides where
    # different chapters have different RTC drift.
    per_clip_offsets: dict[str, float] | None = None
    landscape_only: bool = False
    strava_activity_id: int | None = None
    candidate_cap: int = _CANDIDATE_CAP
    # Minimum median speed (mph) over a candidate's window for it to count
    # as "moving". Stationary clips — red lights, track-stands, coasting to
    # a halt — are dead footage. Set to 0 to disable the motion-gate.
    min_motion_mph: float = 5.0
    skip_gemini: bool = False
    skip_narrative: bool = False  # disable model narrative selection (use greedy)
    # γ in the selection score — how hard picks are pushed to span the whole
    # ride. 0 chases peak quality and will happily leave multi-hour holes;
    # ~2 reproduces the old coverage behaviour. 1.5 is the tuned default.
    coverage_weight: float = 1.5
    include_outro: bool = True  # crossfade last segment into recap card
    outro_crossfade_secs: float = 3.0
    outro_lead_in_secs: float = 2.0  # extra normal-playback seconds before crossfade

    # First clip opens with a blur→clear title card (date + time-of-day) in
    # the recap-card graphic vocabulary. The opener plays over the first
    # ``intro_secs`` seconds, and the first clip is lengthened to fit it.
    include_intro: bool = True
    intro_secs: float = intro_styles.DEFAULT_INTRO_SECS
    intro_style: str = intro_styles.DEFAULT_STYLE
    intro_reveal_secs: float = 0.0

    # Per-ride lockup / recap card overrides. ``None`` falls back to GPS
    # derivation, then to the ``lockups`` design tokens.
    origin: str | None = None
    destination: str | None = None
    subtitle: str | None = None
    road: str | None = None
    crew: str | None = None
    lockup: str | None = None  # in-segment bottom band; defaults to "{origin} · {road}"
    # Anchor the recap end pin at the farthest point reached (the ride apex /
    # turnaround) instead of the GPS terminus. Use when the meaningful
    # destination isn't where the ride ended — e.g. ride up to Bear Mountain,
    # then train/drive home from a different town.
    far_pin: bool = False

    # Colour grade (see grade.py). The defaults leave footage untouched, so
    # grading is strictly opt-in.
    grade_look: str = "none"
    grade_strength: float = 0.35
    grade_wb: str = "off"  # "shot" | "off"

    def __post_init__(self) -> None:
        # Fail at config time rather than per-segment at burn time, which is
        # after candidate generation and the Gemini spend.
        if self.include_intro and self.intro_style not in intro_styles.STYLES:
            raise ValueError(
                f"unknown intro_style {self.intro_style!r}; "
                f"expected one of {intro_styles.STYLES}")

    @property
    def pad_before(self) -> float:
        """Seconds before the candidate midpoint — 50% of segment duration."""
        return self.segment_duration * 0.50

    @property
    def pad_after(self) -> float:
        """Seconds after the candidate midpoint — 50% of segment duration."""
        return self.segment_duration * 0.50


# ═══════════════════════════════════════════════════════════════
# Telemetry scoring — shared by all candidate sources
# ═══════════════════════════════════════════════════════════════

def _telemetry_score_at(ride: RideData, ride_secs: float) -> float:
    """Score a ride timestamp using Garmin telemetry (power, speed, HR).

    Returns a 0-18 scale score:
      Power: 0-10 (500W = 10)
      Speed: 0-5  (30mph = 5)
      HR:    0-3  (190bpm = 3)
    """
    if not ride.start_time or not ride.points:
        return 0.0

    import datetime as dt
    target = ride.start_time + dt.timedelta(seconds=ride_secs)
    point = ride.point_at(target)
    if point is None:
        return 0.0

    score = 0.0
    if point.power:
        score += min(point.power / 50.0, 10.0)
    if point.speed:
        score += min(point.speed * MS_TO_MPH / 6.0, 5.0)
    if point.heart_rate:
        score += max(0, (point.heart_rate - 130) / 20.0)
    return score


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _window_median(ride: RideData, ride_secs: float, attr: str) -> float | None:
    """Median of a telemetry attr over a ±window around ``ride_secs``.

    A window median (not a single sample) so a one-second power dropout or
    a GPS speed glitch can't swing the result. Returns None if the window
    has no samples for that attr.
    """
    if not ride.points:
        return None
    t0 = ride.points[0].timestamp
    vals = [
        getattr(p, attr)
        for p in ride.points
        if getattr(p, attr) is not None
        and abs((p.timestamp - t0).total_seconds() - ride_secs)
        <= _LIVENESS_WINDOW_HALF_SECS
    ]
    return statistics.median(vals) if vals else None


def _window_grade_pct(ride: RideData, ride_secs: float) -> float | None:
    """Average gradient (%) over a ±window around ``ride_secs``.

    Rise over run from the first to the last sample in the window, using
    recorded distance rather than GPS math. Returns None when the window
    is too short to be meaningful — under ~5 m of travel the quotient is
    dominated by barometric noise.
    """
    if not ride.points:
        return None
    t0 = ride.points[0].timestamp
    win = [
        p for p in ride.points
        if p.altitude is not None and p.distance is not None
        and abs((p.timestamp - t0).total_seconds() - ride_secs)
        <= _LIVENESS_WINDOW_HALF_SECS
    ]
    if len(win) < 2:
        return None
    d_dist = win[-1].distance - win[0].distance
    if d_dist < 5.0:
        return None
    return 100.0 * (win[-1].altitude - win[0].altitude) / d_dist


def _populate_telemetry_features(
    candidates: list["Segment"], ride: RideData,
) -> None:
    """Attach {speed_mph, power_w, grade_pct} to each candidate in place.

    Feeds the similarity term in ``_redundancy``. A candidate keeps an
    empty dict when speed or grade is unavailable — ``_kin_sim`` then
    returns None and redundancy degrades to the time-only behaviour.
    Power defaults to 0 on rides with no power meter, which is correct:
    two clips both lacking power should not look *dissimilar* over it.
    """
    filled = 0
    for c in candidates:
        speed_ms = _window_median(ride, c.ride_time_secs, "speed")
        grade = _window_grade_pct(ride, c.ride_time_secs)
        if speed_ms is None or grade is None:
            continue
        power = _window_median(ride, c.ride_time_secs, "power") or 0.0
        c.telemetry_features = {
            "speed_mph": speed_ms * MS_TO_MPH,
            "power_w": float(power),
            "grade_pct": grade,
        }
        filled += 1
    if candidates:
        print(f"  Telemetry features: {filled}/{len(candidates)} candidates")


def _liveness_penalty(ride: RideData, ride_secs: float) -> float:
    """Penalty in [0, _LIVENESS_MAX_PENALTY] for low-speed AND low-power footage.

    Combines telemetry with vision-scored candidates so a beautiful but
    dead-slow coast (e.g. 5-7 mph at 0 W) is demoted. Requires BOTH speed
    and power to be low — a descent (fast, no power) or a climb (slow, high
    power) scores 0. Neutral (0) when telemetry is missing or the ride has
    no power meter (can't judge effort, so don't guess).
    """
    if not ride.has_power:
        return 0.0
    speed_ms = _window_median(ride, ride_secs, "speed")
    power_w = _window_median(ride, ride_secs, "power")
    if speed_ms is None or power_w is None:
        return 0.0
    speed_mph = speed_ms * MS_TO_MPH
    # 0 when moving/working, 1 when fully dead-slow / coasting.
    slow = _clamp01((_LIVENESS_SPEED_OK - speed_mph) /
                    (_LIVENESS_SPEED_OK - _LIVENESS_SPEED_LOW))
    coast = _clamp01((_LIVENESS_POWER_OK - power_w) /
                     (_LIVENESS_POWER_OK - _LIVENESS_POWER_LOW))
    # Product → penalise only when BOTH are low. Either one healthy → ~0.
    return _LIVENESS_MAX_PENALTY * slow * coast


def _apply_liveness_penalty(candidates: list[Segment], ride: RideData) -> None:
    """Damp 'dead' (low-speed AND low-power) candidates in place.

    Skips label-sourced and must-include candidates — the rider's own
    picks are editorial ground truth. Applied to the fused pool BEFORE
    selection, so the lower scores flow through both the model narrative
    pass (which sees seg.score) and the diversity-aware greedy loop.
    """
    damped = 0
    total = 0.0
    for c in candidates:
        srcs = getattr(c, "sources", None) or []
        if c.label.get("must_include") or "label" in srcs or c.source == "label":
            continue
        pen = _liveness_penalty(ride, c.ride_time_secs)
        if pen > 0.01:
            c.score -= pen
            damped += 1
            total += pen
    if damped:
        print(f"  Liveness: damped {damped} low-speed/low-power candidate(s) "
              f"(avg -{total / damped:.1f} pts) — combining telemetry with vision")


# ═══════════════════════════════════════════════════════════════
# Stage 1: Multi-source candidate generation
# ═══════════════════════════════════════════════════════════════

def _candidates_from_short_clips(
    synced_clips: list[SyncedClip], ride: RideData, config: ComposerConfig,
) -> list[Segment]:
    """Treat each short clip as a pre-selected candidate.

    For phone videos (5-15s clips), the user chose to film them — each
    clip IS the highlight. Extracts a standard segment_duration window
    centered on the clip midpoint (same as all other candidate sources).
    """
    from .sync import normalize_tz

    candidates = []
    for sc in synced_clips:
        clip = sc.clip
        mid_video = clip.duration_secs / 2
        mid_wall = clip.video_time_to_wall_time(mid_video)
        if ride.start_time:
            mid_wall = normalize_tz(mid_wall, ride.start_time)
            ride_secs = (mid_wall - ride.start_time).total_seconds()
        else:
            ride_secs = 0

        tel_score = _telemetry_score_at(ride, ride_secs)

        # Standard 3s window, clamped to clip bounds
        v_start = max(0, mid_video - config.pad_before)
        v_end = min(clip.duration_secs, mid_video + config.pad_after)

        candidates.append(Segment(
            clip_name=clip.path.name,
            video_start=v_start,
            video_end=v_end,
            ride_time_secs=ride_secs,
            anchor_video_secs=mid_video,
            score=tel_score + 5.0,  # base bonus for being user-selected
            source="clip",
            label={
                "ride_time_secs": ride_secs,
                "type": "phone_clip",
                "notes": f"Phone clip ({clip.duration_secs:.0f}s)",
                "clip_name": clip.path.name,
                "video_secs": mid_video,
            },
        ))

    return candidates


def _candidates_from_highlights(
    ride: RideData, synced_clips: list[SyncedClip], config: ComposerConfig,
) -> list[Segment]:
    """Generate candidates from FIT telemetry highlights (power/speed/HR spikes, climbs, sprints).

    This is the primary candidate source — works with just a FIT file, no labels needed.
    """
    from .highlights import detect_highlights, HighlightConfig

    hl_config = HighlightConfig(
        padding_before_secs=0,  # we apply our own padding
        padding_after_secs=0,
    )
    highlights = detect_highlights(ride, hl_config)

    candidates = []
    for hl in highlights:
        # Anchor on the actual peak of the highlight, not the run midpoint.
        # For a long sustained run (e.g. 5-min cruise containing one 30mph
        # descent), the midpoint can be minutes away from the moment that
        # made the run interesting. peak_time is set by the detector.
        anchor_time = hl.peak_time or (hl.start_time + (hl.end_time - hl.start_time) / 2)
        ride_secs = (anchor_time - ride.start_time).total_seconds()
        if ride_secs < 0:
            continue

        clip_name, video_secs = _ride_time_to_video(
            ride_secs, ride, synced_clips, config.offset,
        )
        if clip_name is None:
            continue

        # Score: highlight's own score (0-1) scaled to 0-10, plus telemetry
        hl_score = hl.score * 10.0
        tel_score = _telemetry_score_at(ride, ride_secs)
        score = hl_score + tel_score

        candidates.append(Segment(
            clip_name=clip_name,
            video_start=max(0, video_secs - config.pad_before),
            video_end=video_secs + config.pad_after,
            ride_time_secs=ride_secs,
            anchor_video_secs=video_secs,  # FIT peak — exact moment
            score=score,
            source="telemetry",
            label={
                "ride_time_secs": ride_secs,
                "type": hl.reason.value,
                "notes": f"{hl.reason.value}: {hl.peak_value:.0f} ({hl.description or ''})",
                "clip_name": clip_name,
                "video_secs": video_secs,
            },
        ))

    return candidates


def _candidates_from_strava(
    strava_efforts: list, ride: RideData, synced_clips: list[SyncedClip], config: ComposerConfig,
) -> list[Segment]:
    """Generate candidates from Strava segment efforts.

    Score: log(star_count) popularity bonus + telemetry at midpoint.
    """
    candidates = []
    for effort in strava_efforts:
        ride_secs = effort.start_time_secs + effort.elapsed_time_secs / 2
        if ride_secs < 0:
            continue

        clip_name, video_secs = _ride_time_to_video(
            ride_secs, ride, synced_clips, config.offset,
        )
        if clip_name is None:
            continue

        star_score = math.log(max(1, effort.star_count))
        tel_score = _telemetry_score_at(ride, ride_secs)
        score = star_score + tel_score

        candidates.append(Segment(
            clip_name=clip_name,
            video_start=max(0, video_secs - config.pad_before),
            video_end=video_secs + config.pad_after,
            ride_time_secs=ride_secs,
            anchor_video_secs=video_secs,  # segment midpoint
            score=score,
            source="strava",
            label={
                "ride_time_secs": ride_secs,
                "type": "strava_segment",
                "notes": f"{effort.name} ({effort.star_count} stars)",
                "clip_name": clip_name,
                "video_secs": video_secs,
                "strava_segment_id": effort.segment_id,
                "star_count": effort.star_count,
            },
        ))

    return candidates


def _is_must_include(label: dict) -> bool:
    """Detect labels marked for mandatory inclusion.

    Honours both the explicit boolean field (set by the labeler's
    "Must include" checkbox) and the legacy "must include" / "must
    have" phrase in notes for older labels.
    """
    if label.get("must_include"):
        return True
    notes = label.get("notes", "") or ""
    lower = notes.lower()
    return "must include" in lower or "must have" in lower


def _candidates_from_labels(
    labels: list[dict], ride: RideData, config: ComposerConfig,
) -> list[Segment]:
    """Generate candidates from manual labels (optional enrichment).

    Labels flagged "must include" (via the labeler checkbox or a legacy
    notes phrase) get a massive score boost and extended duration (15s
    instead of the default segment_duration).
    """
    candidates = []
    for raw_lab in labels:
        lab = normalize_label_scale(raw_lab)
        vs = lab.get("video_secs", 0.0)
        if vs is None:
            continue
        clip_name = lab.get("clip_name")
        if not clip_name:
            continue

        ride_secs = lab.get("ride_time_secs", 0)
        tel_score = _telemetry_score_at(ride, ride_secs)
        must_include = _is_must_include(lab)

        # Type bonuses — types match Gemini's clip_type vocabulary
        label_type = lab.get("type", "")
        bonus = 0.0
        if must_include:
            bonus = 100.0  # guarantees top ranking
        elif label_type in ("incident",):
            bonus = 3.0
        elif label_type in ("landmark", "descent"):
            bonus = 1.5
        elif label_type in ("group_riding", "action"):
            bonus = 1.0

        seg = Segment(
            clip_name=clip_name,
            video_start=max(0, vs - config.pad_before),
            video_end=vs + config.pad_after,
            ride_time_secs=ride_secs,
            anchor_video_secs=vs,  # user-labeled exact moment
            score=tel_score + bonus,
            source="label",
            label=lab,
            portrait_crop_bias=float(lab.get("portrait_crop_bias", 0.0)),
        )
        if must_include:
            seg.label["must_include"] = True
        candidates.append(seg)

    return candidates


# Rubric dimensions used for scoring. `light` is intentionally excluded:
# on a test ride it returned a single value (4) across all 17 rated
# clips, contributing nothing as a discriminator. It is still stored
# in the rubric and shown in the reviewer for human inspection.
_SCORING_RUBRIC_DIMS = ("composition", "motion", "scenery", "subject")


def _rubric_score(rubric: dict) -> float | None:
    """Score a Gemini rubric on a 1-10 scale using composition / motion
    / scenery / subject (light excluded — it does not discriminate).

    Returns None if the rubric is empty or missing all scoring dims.
    """
    if not rubric:
        return None
    vals = [rubric.get(k) for k in _SCORING_RUBRIC_DIMS]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


# Non-Gemini sources cap below the Gemini ceiling so a top-of-source
# telemetry candidate can't out-rank a strong-rubric Gemini clip on
# visual quality alone. Gemini clips score 1-10 from rubric directly.
_NON_GEMINI_SCORE_FLOOR = 1.0
_NON_GEMINI_SCORE_CEILING = 7.0
# Telemetry-only candidates with no nearby Gemini hit are visually
# blind. They get scored, but their final score is capped here so they
# only fill if there's nothing better.
_BLIND_TELEMETRY_CEILING = 3.0


def _normalize_scores(candidates: list[Segment]) -> None:
    """Score candidates on a unified 1-10 scale in-place.

    Rule: Gemini-rated clips are scored directly from rubric quality
    (composition/motion/scenery/subject). Other sources go through
    per-source min-max but are capped at 1-7, so a top telemetry
    candidate cannot outrank a strong-rubric Gemini clip on visual
    quality alone.
    """
    for seg in candidates:
        rs = _rubric_score(seg.rubric)
        if rs is not None:
            seg.score = rs

    by_source: dict[str, list[Segment]] = {}
    for seg in candidates:
        if not seg.rubric:
            by_source.setdefault(seg.source, []).append(seg)

    span_size = _NON_GEMINI_SCORE_CEILING - _NON_GEMINI_SCORE_FLOOR
    for source, segs in by_source.items():
        scores = [s.score for s in segs]
        lo, hi = min(scores), max(scores)
        span = hi - lo
        if span < 1e-6:
            mid = (_NON_GEMINI_SCORE_FLOOR + _NON_GEMINI_SCORE_CEILING) / 2
            for s in segs:
                s.score = mid
        else:
            for s in segs:
                s.score = _NON_GEMINI_SCORE_FLOOR + ((s.score - lo) / span) * span_size


# Vision providers share the same priority band as historical "gemini".
_SOURCE_PRIORITY = {
    "label": 0,
    "telemetry": 1,
    "strava": 2,
    "clip": 3,
    **{src: 4 for src in VISION_SOURCES},
}

# Review-cap tunables: fraction of slots filled top-down by score, then the
# remainder sampled round-robin across these score bands so weak Gemini
# judgments stay visible for calibration.
_REVIEW_CAP_TOP_QUOTA = 0.60
_REVIEW_BAND_LOW = 3.5    # score < LOW
_REVIEW_BAND_HIGH = 6.5   # LOW <= score < HIGH <= score


def _cap_candidates_with_distribution(
    candidates: list[Segment], cap: int,
) -> list[Segment]:
    """Cap candidates without erasing the visual-score distribution.

    Review is a calibration tool, not just a top-N list. Keep the strongest
    clips, then reserve the remaining slots for a deterministic sample from
    high/mid/low score bands so Gemini's weaker visual judgments stay visible.
    """
    if cap <= 0 or len(candidates) <= cap:
        return sorted(candidates, key=lambda s: s.score, reverse=True)

    ordered = sorted(candidates, key=lambda s: s.score, reverse=True)

    must = [s for s in ordered if s.label.get("must_include")]
    selected: list[Segment] = []
    selected_ids: set[int] = set()

    for seg in must:
        if id(seg) not in selected_ids:
            selected.append(seg)
            selected_ids.add(id(seg))
        if len(selected) >= cap:
            return sorted(selected, key=lambda s: s.score, reverse=True)

    top_quota = max(1, int(cap * _REVIEW_CAP_TOP_QUOTA))
    for seg in ordered:
        if len(selected) >= top_quota:
            break
        if id(seg) in selected_ids:
            continue
        selected.append(seg)
        selected_ids.add(id(seg))

    # Bands are disjoint by score and exclude already-selected segments, so
    # each remaining segment lives in exactly one band — no cross-band dedup
    # needed inside the round-robin.
    bands = [
        [s for s in ordered if s.score < _REVIEW_BAND_LOW and id(s) not in selected_ids],
        [s for s in ordered if _REVIEW_BAND_LOW <= s.score < _REVIEW_BAND_HIGH and id(s) not in selected_ids],
        [s for s in ordered if s.score >= _REVIEW_BAND_HIGH and id(s) not in selected_ids],
    ]
    band_idx = 0
    while len(selected) < cap and any(bands):
        band = bands[band_idx % len(bands)]
        if band:
            seg = band.pop(0)
            selected.append(seg)
            selected_ids.add(id(seg))
        band_idx += 1

    if len(selected) < cap:
        for seg in ordered:
            if id(seg) in selected_ids:
                continue
            selected.append(seg)
            selected_ids.add(id(seg))
            if len(selected) >= cap:
                break

    return sorted(selected, key=lambda s: s.score, reverse=True)


def _union_review_window(winner: "Segment", a: "Segment", b: "Segment") -> None:
    """Union two segments' review windows onto ``winner`` when they share a
    clip, so the preview shows the full action span. Cross-clip merges
    (chapter boundaries) leave windows alone.
    """
    if a.clip_name == b.clip_name:
        winner.video_start = min(a.video_start, b.video_start)
        winner.video_end = max(a.video_end, b.video_end)


def _fuse_candidates(candidates: list[Segment], cap: int = _CANDIDATE_CAP) -> list[Segment]:
    """Normalize scores, merge nearby cross-source candidates, cap at best N.

    When candidates within 15s of each other come from different sources
    (telemetry + Gemini, label + Gemini, etc.), they describe the same
    event and should be one candidate with cross-source confirmation. The
    union of their windows becomes the review preview span, but the cut
    anchor goes to the highest-priority source: label > telemetry > strava
    > Gemini. Same-source candidates do NOT merge, with one exception: two
    Gemini-rated frames within ``_GEMINI_DEDUP_WINDOW`` are visual
    near-duplicates of the same scene and collapse to the higher-scored one.
    Adjacent telemetry spikes are independent events and always stay separate.
    """
    if not candidates:
        return []

    _normalize_scores(candidates)

    for seg in candidates:
        if not seg.sources:
            seg.sources = [seg.source]

    candidates.sort(key=lambda s: s.ride_time_secs)

    _MERGE_WINDOW = 15.0
    _LABEL_MERGE_WINDOW = 45.0  # wider when a label is involved
    _GEMINI_DEDUP_WINDOW = 20.0  # collapse same-scene Gemini near-duplicates

    fused: list[Segment] = []
    for seg in candidates:
        merged = False
        for i, existing in enumerate(fused):
            has_label = ("label" in (seg.sources or [])
                         or "label" in (existing.sources or []))
            window = _LABEL_MERGE_WINDOW if has_label else _MERGE_WINDOW
            same_source = seg.source == existing.source
            dt = abs(seg.ride_time_secs - existing.ride_time_secs)
            if dt < window and not same_source:
                new_sources = set(existing.sources) | set(seg.sources)

                # Anchor goes to the highest-priority source — its
                # timestamp is the most precise (FIT peak, user label).
                if (_SOURCE_PRIORITY.get(seg.source, 99)
                        < _SOURCE_PRIORITY.get(existing.source, 99)):
                    winner = seg
                    loser = existing
                else:
                    winner = existing
                    loser = seg

                winner.sources = sorted(new_sources)
                winner.source = min(
                    new_sources, key=lambda s: _SOURCE_PRIORITY.get(s, 99),
                )

                _union_review_window(winner, seg, existing)

                # Inherit rubric from whichever source has one
                if not winner.rubric and loser.rubric:
                    winner.rubric = loser.rubric

                loser_notes = loser.label.get("notes", "")
                if loser_notes and winner.label.get("notes", "") != loser_notes:
                    winner.label["notes"] = (
                        f"{winner.label.get('notes', '')} + {loser_notes}"
                    )

                # Pick the merged score: when either side has a rubric,
                # the rubric is the source of truth for visual quality
                # (a strong telemetry hit shouldn't carry a weak-rubric
                # frame past a stronger-rubric clip). When neither has,
                # take the higher base score.
                merged_rubric = winner.rubric or loser.rubric
                rubric_score = _rubric_score(merged_rubric)
                if rubric_score is not None:
                    winner.score = rubric_score
                else:
                    winner.score = max(winner.score, seg.score, existing.score)

                fused[i] = winner
                merged = True
                break
            if (same_source and seg.source in VISION_SOURCES
                    and dt < _GEMINI_DEDUP_WINDOW):
                # Two vision-rated frames this close describe the same
                # scene (e.g. the same bridge crossing flagged twice with
                # slightly different wording). Collapse to the higher-scored
                # one so the cut never shows the same view back-to-back.
                # Vision scores are already rubric-derived (_normalize_scores
                # above), so .score IS the visual-quality comparison. Gated to
                # vision sources on purpose: adjacent telemetry spikes ARE
                # independent events and must stay separate.
                winner = seg if seg.score > existing.score else existing
                _union_review_window(winner, seg, existing)
                fused[i] = winner
                merged = True
                break
        if not merged:
            fused.append(seg)

    # Cross-source confirmation boost — applied once on the FINAL source
    # set. Gated by rubric quality: a clip's visual rubric must justify
    # the boost. Otherwise telemetry agreement on a visually weak frame
    # (e.g. an 858W power peak landing on an empty intersection) would
    # let the clip outrank stronger Gemini-only candidates.
    for seg in fused:
        n = len(seg.sources)
        if n <= 1:
            continue
        rs = _rubric_score(seg.rubric)
        if rs is None:
            # Pure non-Gemini multi-source merge — modest agreement boost
            seg.score += (n - 1) * 0.5
        elif rs >= 5.5:
            # Strong rubric (avg 5.5+ across the 4 scoring dims) — full
            # boost matches the original +2 in spirit but on the new scale
            seg.score += (n - 1) * 1.0
        elif rs >= 4.5:
            seg.score += (n - 1) * 0.3
        # else: rubric is too weak; agreement isn't trustworthy

    # Telemetry-only candidates with no Gemini partner are visually
    # unverified. Cap their score so they only fill in for empty windows.
    for seg in fused:
        if seg.rubric:
            continue
        if VISION_SOURCES.intersection(seg.sources):
            continue
        if seg.score > _BLIND_TELEMETRY_CEILING:
            seg.score = _BLIND_TELEMETRY_CEILING

    return _cap_candidates_with_distribution(fused, cap)


# Seconds either side of a candidate's anchor to sample speed when deciding
# whether it's moving. Wide enough to ride out a single GPS glitch, narrow
# enough not to bleed into adjacent moving footage.
_MOTION_WINDOW_HALF_SECS = 3.0


def _drop_stationary(
    candidates: list[Segment], ride: RideData, config: ComposerConfig,
) -> list[Segment]:
    """Remove candidates with no motion.

    Two ways a clip reads as stopped, both caught here:
      - near-zero median speed over the window (red light, track-stand), or
      - no telemetry in the window at all — the Garmin auto-pauses when
        stationary, leaving a GPS gap.

    Gated on ``config.min_motion_mph`` (0 disables). Must-include labels are
    exempt: if the user explicitly flagged a stopped moment, honor it.
    """
    if config.min_motion_mph <= 0 or not ride.points:
        return candidates
    t0 = ride.points[0].timestamp
    kept: list[Segment] = []
    dropped = 0
    for c in candidates:
        if c.label.get("must_include"):
            kept.append(c)
            continue
        speeds = [
            p.speed * MS_TO_MPH
            for p in ride.points
            if p.speed is not None
            and abs((p.timestamp - t0).total_seconds() - c.ride_time_secs)
            <= _MOTION_WINDOW_HALF_SECS
        ]
        if not speeds or statistics.median(speeds) < config.min_motion_mph:
            dropped += 1
            continue
        kept.append(c)
    if dropped:
        print(f"  Motion-gate: dropped {dropped} stationary candidate(s) "
              f"(< {config.min_motion_mph:.0f} mph or auto-paused)")
    return kept


def _generate_candidates(
    ride: RideData,
    config: ComposerConfig,
    synced_clips: list[SyncedClip] | None = None,
    labels: list[dict] | None = None,
    strava_efforts=None,
    gemini_candidates: list[Segment] | None = None,
) -> list[Segment]:
    """Generate, fuse, and cap candidates from all available sources.

    Sources (all optional except ride):
      1. FIT telemetry highlights (always available)
      2. Strava segment efforts (if activity ID provided)
      3. Vision-model candidates (if not skipped)
      4. Manual labels (if provided)
    """
    all_candidates: list[Segment] = []

    # 0. Short clips mode: if all clips are short (< 30s), each clip is
    #    a user-selected moment (phone footage). Use entire clips as candidates
    #    instead of searching for highlights within them.
    short_clip_mode = False
    if synced_clips:
        max_dur = max(sc.clip.duration_secs for sc in synced_clips)
        if max_dur < 30.0:
            short_clip_mode = True
            short = _candidates_from_short_clips(synced_clips, ride, config)
            print(f"  Short clips (phone): {len(short)} candidates (each clip = 1 candidate)")
            all_candidates.extend(short)

    # 1. Telemetry highlights — always available (skip in short-clip mode
    #    since highlights rarely map to the tiny coverage windows)
    if synced_clips and not short_clip_mode:
        tel = _candidates_from_highlights(ride, synced_clips, config)
        print(f"  Telemetry highlights: {len(tel)} candidates")
        all_candidates.extend(tel)

    # 2. Strava segments
    if strava_efforts and synced_clips:
        strava = _candidates_from_strava(strava_efforts, ride, synced_clips, config)
        print(f"  Strava segments: {len(strava)} candidates")
        all_candidates.extend(strava)

    # 3. Vision model
    if gemini_candidates:
        src = gemini_candidates[0].source or "vision"
        print(f"  Vision ({src}): {len(gemini_candidates)} candidates")
        all_candidates.extend(gemini_candidates)

    # 4. Manual labels
    if labels:
        lab = _candidates_from_labels(labels, ride, config)
        print(f"  Manual labels: {len(lab)} candidates")
        all_candidates.extend(lab)

    # Drop dead footage — stops at lights, track-stands, auto-pause gaps —
    # before fusion, so a stationary clip can't survive on a strong Gemini
    # rubric or cross-source telemetry agreement. Must-include labels exempt.
    all_candidates = _drop_stationary(all_candidates, ride, config)

    # Fuse and cap
    fused = _fuse_candidates(all_candidates, config.candidate_cap)
    print(f"  After fusion: {len(fused)} candidates (cap={config.candidate_cap})")

    # Pad with telemetry-scored fillers so the landscape/portrait selectors
    # always have enough candidates to fill their slot budgets. Without this,
    # Central Park-style rides (low power/speed peaks, light Gemini hits)
    # produce a candidate pool smaller than landscape_duration/segment_duration
    # and the final reel comes in well under the target length.
    landscape_slots = int(config.landscape_duration / config.segment_duration)
    portrait_slots = (
        0 if config.landscape_only
        else int(config.portrait_duration / config.segment_duration)
    )
    target_n = max(landscape_slots, portrait_slots)
    if synced_clips and len(fused) < target_n:
        fillers = _telemetry_fillers(ride, synced_clips, fused, target_n, config)
        if fillers:
            print(f"  Filler: +{len(fillers)} telemetry candidates "
                  f"(target_n={target_n})")
            fused.extend(fillers)
            fused.sort(key=lambda s: s.ride_time_secs)

    # Combine telemetry with vision: demote clips that are dead-slow AND
    # putting down no power (Gemini rated them on looks alone). Runs after
    # fusion/fillers so every candidate the selectors see carries an
    # effort-aware score. Label/must-include picks are exempt inside.
    _apply_liveness_penalty(fused, ride)
    return fused


def _telemetry_fillers(
    ride: RideData,
    synced_clips: list[SyncedClip],
    existing: list[Segment],
    target_n: int,
    config: ComposerConfig,
) -> list[Segment]:
    """Add filler candidates scored by raw telemetry until the pool hits target_n.

    Scans the ride at a coarse interval, computes a telemetry score at each
    point, then greedily picks the highest-scored points whose ride time is
    at least MIN_GAP seconds from existing candidates and other picks. This
    ensures even sparse rides (mostly-flat city loops, easy spins) still
    yield enough candidates to fill the landscape budget.
    """
    needed = target_n - len(existing)
    if needed <= 0 or not ride.start_time or not ride.points:
        return []

    _MIN_GAP_SECS = 10.0
    _STEP_SECS = 5  # sampling cadence — fine enough to catch short peaks

    ride_start = ride.start_time
    ride_end_secs = (ride.points[-1].timestamp - ride_start).total_seconds()

    samples: list[tuple[float, float, str, float]] = []  # (t, score, clip, vsecs)
    t = 0.0
    while t <= ride_end_secs:
        score = _telemetry_score_at(ride, t)
        if score > 0:
            clip_name, video_secs = _ride_time_to_video(
                t, ride, synced_clips, config.offset,
            )
            if clip_name is not None:
                samples.append((t, score, clip_name, video_secs))
        t += _STEP_SECS

    samples.sort(key=lambda x: -x[1])

    picked_times = [s.ride_time_secs for s in existing]
    fillers: list[Segment] = []
    for ride_secs, score, clip_name, video_secs in samples:
        if len(fillers) >= needed:
            break
        if any(abs(ride_secs - pt) < _MIN_GAP_SECS for pt in picked_times):
            continue
        picked_times.append(ride_secs)
        # Fillers must rank BELOW every real candidate so selection only
        # uses them to fill temporal gaps after the genuine picks. Real
        # telemetry/Gemini/label candidates land in the 1-15 score range,
        # so we squash raw telemetry score (0-18) into 0.0-0.9.
        filler_score = min(score * 0.05, 0.9)
        fillers.append(Segment(
            clip_name=clip_name,
            video_start=max(0, video_secs - config.pad_before),
            video_end=video_secs + config.pad_after,
            ride_time_secs=ride_secs,
            anchor_video_secs=video_secs,
            score=filler_score,
            source="filler",
            label={
                "ride_time_secs": ride_secs,
                "type": "filler",
                "notes": f"filler (telemetry score {score:.1f})",
                "clip_name": clip_name,
                "video_secs": video_secs,
            },
        ))
    return fillers


def _ride_time_to_video(
    ride_secs: float,
    ride: RideData,
    synced_clips: list[SyncedClip],
    offset: float = 0.0,
) -> tuple[str | None, float | None]:
    """Map a ride-time offset to a GoPro clip name + video position.

    Uses each ``SyncedClip``'s own ``offset_secs`` (set by ``sync_all``
    when per-clip offsets are configured), falling back to the legacy
    ``offset`` parameter only if a clip has no offset of its own. This
    keeps candidate→clip mapping correct on multi-recording rides where
    each chapter has its own RTC drift.
    """
    import datetime as dt
    from .sync import normalize_tz

    if not ride.start_time:
        return None, None

    target_wall = ride.start_time + dt.timedelta(seconds=ride_secs)

    for sc in synced_clips:
        clip = sc.clip
        clip_start = normalize_tz(clip.creation_time, target_wall)
        clip_end = clip_start + dt.timedelta(seconds=clip.duration_secs)

        clip_offset = sc.offset_secs or offset
        adjusted = target_wall - dt.timedelta(seconds=clip_offset)

        if clip_start <= adjusted <= clip_end:
            video_secs = (adjusted - clip_start).total_seconds()
            return clip.path.name, video_secs

    return None, None


# ═══════════════════════════════════════════════════════════════
# Stage 2: Ranking — select top-N by score, ensure diversity
# ═══════════════════════════════════════════════════════════════


def _backfill_gaps(
    selected: list[Segment],
    pool: list[Segment],
    budget_secs: float,
    config: ComposerConfig,
    max_gap_secs: float = 180.0,
    edge_pad: float = 30.0,
) -> list[Segment]:
    """Force-include the best candidate inside oversized gaps.

    After narrative selection, walk consecutive picks chronologically.
    For any gap > max_gap_secs, pull the highest-scored candidate from
    the pool that falls inside the window (with edge_pad buffer from
    each side so we don't immediately re-pick something next to an
    already-selected clip). Iterates so a fresh insertion can itself
    create new sub-gaps to consider. Bounded by the time budget so we
    don't add filler past the duration target.
    """
    if len(selected) < 2:
        return selected
    pool_by_id = {id(s): s for s in pool}
    selected = list(selected)

    # Slot count derived from budget, using the burn duration (not the
    # candidate's review-window span). Same reasoning as in select_segments.
    n_slots = max(1, int(budget_secs / config.segment_duration))

    while True:
        selected.sort(key=lambda s: s.ride_time_secs)
        used_ids = {id(s) for s in selected}
        if len(selected) >= n_slots:
            break

        best_for_gap = None  # (gap_size, candidate, position)
        for i in range(len(selected) - 1):
            a, b = selected[i], selected[i + 1]
            gap = b.ride_time_secs - a.ride_time_secs
            if gap <= max_gap_secs:
                continue
            window_lo = a.ride_time_secs + edge_pad
            window_hi = b.ride_time_secs - edge_pad
            best = None
            for c in pool_by_id.values():
                if id(c) in used_ids:
                    continue
                if not (window_lo <= c.ride_time_secs <= window_hi):
                    continue
                if best is None or c.score > best.score:
                    best = c
            if best is None:
                continue
            if best_for_gap is None or gap > best_for_gap[0]:
                best_for_gap = (gap, best, i + 1)

        if best_for_gap is None:
            break
        _, fill, _ = best_for_gap
        selected.append(fill)

    return selected


def _fill_to_count(
    selected: list[Segment], pool: list[Segment], n: int, min_gap: float,
    fill_coverage_weight: float,
) -> list[Segment]:
    """Guarantee exactly ``n`` picks, spreading fillers across the ride.

    Runs after the quality greedy when it stopped short of the target count
    (its remaining candidates were net-negative — crowded/dead). Uses the
    SAME ``_effective_score`` as the quality phase, only with a heavier
    coverage weight: the biggest hole in the reel wins, but redundancy and
    base score still count, so this stage can no longer drop a generic clip
    into a gap that a genuinely good one could fill.

    Because redundancy is similarity-aware, a clip the quality phase passed
    over for sitting near an existing pick is eligible again here whenever
    it is *unlike* that pick — the climb→descent case. Only the hard
    ``min_gap`` is absolute, and it exists so cuts never share frames.
    """
    selected = list(selected)
    used = {id(s) for s in selected}
    added = 0
    while len(selected) < n:
        best = None
        best_key = -1e18
        for seg in pool:
            if id(seg) in used:
                continue
            if any(abs(seg.ride_time_secs - s.ride_time_secs) < min_gap
                   for s in selected):
                continue
            key = _effective_score(seg, selected, fill_coverage_weight)
            if key > best_key:
                best_key, best = key, seg
        if best is None:
            break  # nothing left that respects min_gap
        selected.append(best)
        used.add(id(best))
        added += 1
    if added:
        print(f"  Coverage fill: +{added} clip(s) to reach the target count "
              f"(same ranking as the quality phase, coverage weighted "
              f"{fill_coverage_weight:g})")
    return selected


def _narrative_select(
    candidates: list[Segment],
    n_clips: int,
    budget_secs: float,
    layout: str = "landscape",
) -> list[Segment] | None:
    """Ask the configured model to select clips that tell a ride story.

    Passes the full candidate list with timestamps, scores, descriptions, and
    clip types. Returns indices of selected clips in narrative order.
    Falls back to None on any failure (caller should use greedy selection).
    Uses the same MODEL_PROVIDER as the vision scan — never a second provider.
    """
    from .config import get_settings
    from .models import get_model_adapter, provider_api_key

    settings = get_settings()
    if not provider_api_key(settings):
        return None

    candidates = sorted(candidates, key=lambda s: s.ride_time_secs)

    # Format candidates for the prompt
    lines = []
    for i, seg in enumerate(candidates):
        notes = seg.label.get("notes", "")[:60]
        clip_type = seg.label.get("clip_type", seg.label.get("type", ""))
        # Vision candidates carry a rubric, not visual/action — read through
        # the fold so they don't all report 0 to the selector.
        visual, action = rating_visual_action(seg.label)
        sources = ",".join(seg.sources) if seg.sources else seg.source
        mins = int(seg.ride_time_secs // 60)
        secs = int(seg.ride_time_secs % 60)
        lines.append(
            f"[{i}] t={mins}:{secs:02d} score={seg.score:.1f} "
            f"type={clip_type} visual={visual} action={action} "
            f"sources={sources} | {notes}"
        )

    candidate_text = "\n".join(lines)

    layout_guidance = (
        "This is for a landscape YouTube highlight (60s). "
        "Favor visual quality, scenic moments, and dramatic reveals."
        if layout == "landscape"
        else "This is for a portrait Instagram Reel (30s). "
        "Favor intense action and fast-paced moments, BUT ALSO include "
        "the most scenic peaks — a reel still needs beauty between the action."
    )

    from .prompt_registry import prompt_body
    prompt = prompt_body("narrative_select", _NARRATIVE_PROMPT_VERSION).format(
        n_clips=n_clips,
        budget_secs=budget_secs,
        layout_guidance=layout_guidance,
        candidate_text=candidate_text,
    )

    def _call(prompt_text: str) -> list:
        """One model call → flat list of candidate indices (flattening any
        nested arrays like [[0],[5]]). Empty list on a non-list response."""
        adapter = get_model_adapter(settings)
        result = adapter.complete_json(
            prompt=prompt_text,
            temperature=0.3,
            max_output_tokens=1024,
        )
        if not isinstance(result, list):
            return []
        flat = []
        for item in result:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat

    def _absorb(flat: list, selected: list, seen: set) -> None:
        """Map raw indices → Segments, in order, de-duped, capped at n_clips."""
        for idx in flat:
            try:
                idx = int(idx)
            except (ValueError, TypeError):
                continue
            if 0 <= idx < len(candidates) and idx not in seen:
                selected.append(candidates[idx])
                seen.add(idx)
            if len(selected) >= n_clips:
                break

    try:
        selected: list = []
        seen: set = set()
        _absorb(_call(prompt), selected, seen)

        # Re-ask once if the model under-delivered on the count. Models anchor
        # on a short answer and quietly ignore "exactly N" (it returned ~n/2
        # before this). A targeted follow-up naming what's already chosen
        # reliably closes the gap and costs a fraction of a cent.
        if 0 < len(selected) < n_clips:
            need = n_clips - len(selected)
            followup = (
                f"{prompt}\n\nYou already chose these indices: {sorted(seen)}.\n"
                f"That is only {len(selected)} of the required {n_clips}. "
                f"Return a JSON array of {need} MORE distinct indices (NONE "
                f"already listed) — the next-best story moments, keeping clip_type "
                f"variety and spread across the whole ride. Return ONLY the JSON array."
            )
            _absorb(_call(followup), selected, seen)

        # Accept a partial narrative — even ~half is worth keeping for its
        # story logic, because the greedy + coverage-fill below now top it up
        # to the full count cleanly (that safety net is why we can trust a
        # short narrative here where we previously demanded 80%).
        if len(selected) < max(1, int(round(n_clips * 0.5))):
            return None

        selected.sort(key=lambda s: s.ride_time_secs)  # chronological
        provider = (settings.model_provider or "gemini").lower()
        print(f"  Narrative selection ({provider}): {len(selected)} clips "
              f"(target {n_clips})")
        return selected

    except Exception as exc:
        print(f"  Narrative selection failed, falling back to greedy: {exc}")
        return None


def select_segments(
    budget_secs: float,
    config: ComposerConfig,
    ride=None,
    synced_clips=None,
    labels: list[dict] | None = None,
    strava_efforts=None,
    gemini_candidates: list[Segment] | None = None,
    precomputed_candidates: list[Segment] | None = None,
    layout: str = "landscape",
) -> list[Segment]:
    """Two-stage selection: generate candidates, then rank and select for budget.

    First asks the configured model to pick clips that tell a compelling ride
    story. Falls back to greedy gap-filling on failure.

    If precomputed_candidates is provided, skips candidate generation and
    uses those directly. Otherwise generates from all available sources.

    Returns candidates in chronological order.
    """
    if precomputed_candidates is not None:
        candidates = list(precomputed_candidates)
    else:
        candidates = _generate_candidates(
            ride, config,
            synced_clips=synced_clips,
            labels=labels,
            strava_efforts=strava_efforts,
            gemini_candidates=gemini_candidates,
        )
    if not candidates:
        return []

    # Similarity-aware selection needs telemetry on every candidate. Do this
    # here rather than in _generate_candidates so precomputed candidates
    # (re-composes from a saved moments.json, which predates the field) get
    # features too. Cheap — a windowed median per candidate.
    if ride is not None and any(not c.telemetry_features for c in candidates):
        _populate_telemetry_features(candidates, ride)

    # Seed selection with "must include" candidates — they always make the cut
    must_includes = [s for s in candidates if s.label.get("must_include")]
    # n is the total slot count for the highlight (e.g. 20 for landscape, 10 for
    # portrait). Each burned clip is exactly config.segment_duration long
    # regardless of the candidate's review-window span, so budget math must use
    # the burn duration, not s.duration.
    n = max(1, int(budget_secs / config.segment_duration))

    # Try model narrative selection first — apply proximity-aware filter and
    # backfill with greedy if the model returned too few or clustered clips.
    if not config.skip_narrative:
        narrative = _narrative_select(candidates, n, budget_secs, layout)
        if narrative is not None:
            # Diversity-aware filter. We rank narrative picks by their
            # boosted score (descending) so the strongest claims its spot
            # first — order-independent of the model's story-order response.
            # A pick survives unless the crowding penalty drives its
            # effective score below zero, i.e., it's clearly net-negative
            # against what's already in. No absolute threshold — that
            # would silently nuke whole narrative on low-key rides.
            #
            # Two cuts closer than ``segment_duration`` would share
            # frames in the burn, so that's a hard reject regardless of
            # score.
            min_gap = max(config.segment_duration, _MIN_CUT_GAP_SECS)
            filtered = list(must_includes)
            filtered_ids = {id(s) for s in filtered}

            ranked = sorted(
                narrative,
                key=lambda s: s.score + _label_boost(s),
                reverse=True,
            )
            for seg in ranked:
                if id(seg) in filtered_ids:
                    continue
                if any(abs(seg.ride_time_secs - s.ride_time_secs) < min_gap
                       for s in filtered):
                    continue
                # Coverage is not a factor here — the narrative pass already
                # chose these for story shape; this only drops redundant picks.
                eff = _effective_score(seg, filtered, 0.0)
                if eff < 0.0:
                    continue
                filtered.append(seg)
                filtered_ids.add(id(seg))

            # Force-include the best candidate inside any oversized gap.
            # Narrative selection often clusters around a single arc and
            # leaves multi-minute holes elsewhere in the ride. We patch
            # those gaps with the highest-scored available candidate
            # before returning, so the cut spans the whole ride.
            filtered = _backfill_gaps(filtered, candidates, budget_secs, config)

            # If we have enough, return as-is
            if len(filtered) >= n:
                filtered.sort(key=lambda s: s.ride_time_secs)
                return filtered[:n]

            # Otherwise, seed greedy with what Gemini gave us and let it
            # fill the remaining slots using gap-filling logic below
            must_includes = filtered

    # ── Similarity-aware greedy ───────────────────────────────
    # Each iteration picks the candidate with the highest effective
    # score:
    #
    #   eff = base_score + label_boost − λ·redundancy + γ·coverage
    #
    # redundancy is temporal closeness TIMES feature similarity, so two
    # near-duplicates collapse to one while a climb and the descent off
    # its summit both survive — adjacent in time, opposite in telemetry.
    # γ (config.coverage_weight) trades peak quality against spanning the
    # whole ride. Hard floor at ``_MIN_CUT_GAP_SECS`` so cuts never share
    # frames and never sit on top of each other.
    min_gap = max(config.segment_duration, _MIN_CUT_GAP_SECS)

    selected = list(must_includes)
    selected_set = {id(s) for s in must_includes}

    pool = [s for s in candidates if id(s) not in selected_set]

    # Quality phase: pick highest-eff clips first. When the best remaining
    # clip goes net-negative (crowded/dead), stop the quality phase — but do
    # NOT stop the reel: coverage-fill below tops it up to the full count
    # using the same ranking with a heavier coverage weight.
    remaining = max(0, n - len(selected))
    for _ in range(remaining):
        best_seg = None
        best_eff = -999.0

        for seg in pool:
            if id(seg) in selected_set:
                continue
            # Hard min-gap: avoid burning overlapping cuts.
            if any(abs(seg.ride_time_secs - s.ride_time_secs) < min_gap
                   for s in selected):
                continue

            eff = _effective_score(seg, selected, config.coverage_weight)
            if eff > best_eff:
                best_eff = eff
                best_seg = seg

        if best_seg is None:
            break
        if best_eff < _GREEDY_MIN_MARGINAL:
            break  # quality picks exhausted — hand off to coverage-fill

        selected.append(best_seg)
        selected_set.add(id(best_seg))

    # Guarantee the full slot count, leaning harder on coverage.
    selected = _fill_to_count(
        selected, pool, n, min_gap,
        config.coverage_weight * _FILL_COVERAGE_MULTIPLIER,
    )

    selected.sort(key=lambda s: s.ride_time_secs)
    return selected[:n]


# ═══════════════════════════════════════════════════════════════
# Composition pipeline
# ═══════════════════════════════════════════════════════════════

def _normalize_selection_input(raw: list[dict]) -> tuple[list[dict], str]:
    """Detect file format and project into a uniform selection-shape.

    A moments.json item is identified by the presence of a `status`
    field. Such items are filtered to {approved, auto} and projected
    to the flat selection dict shape compose_from_selections expects,
    folding effective_anchor / effective_review_span into anchor_video_
    secs / video_start / video_end.

    A selected_candidates.json item is passed through unchanged.
    """
    if not raw:
        return [], "selected_candidates"

    is_moments = any("status" in item for item in raw)
    if not is_moments:
        return raw, "selected_candidates"

    out: list[dict] = []
    for m in raw:
        status = m.get("status", "pending")
        if status not in ("approved", "auto"):
            continue
        anchor = m.get("user_anchor_override")
        if anchor is None:
            anchor = m.get("anchor_video_secs", 0.0)
        in_out = m.get("user_in_out")
        if in_out and len(in_out) == 2:
            v_start, v_end = float(in_out[0]), float(in_out[1])
        else:
            v_start = float(m.get("video_start", 0.0))
            v_end = float(m.get("video_end", 0.0))
        out.append({
            "stable_id": m.get("stable_id"),
            "clip_name": m.get("clip_name"),
            "video_start": v_start,
            "video_end": v_end,
            "anchor_video_secs": float(anchor),
            "ride_time_secs": float(m.get("ride_time_secs", 0.0)),
            "score": float(m.get("score", 0.0)),
            "rubric": m.get("rubric", {}),
            "notes": m.get("notes", ""),
            "rating": int(m.get("rating", 0) or 0),
            "portrait_crop_bias": float(m.get("portrait_crop_bias", 0.0)),
            "final_trim_secs": m.get("final_trim_secs", 3.0),
        })
    return out, "moment"


def concatenate_clips(clip_paths: list[Path], output_path: Path) -> Path:
    """Concatenate clips using ffmpeg concat demuxer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = output_path.parent / f"_concat_{output_path.stem}.txt"
    list_file.write_text("\n".join(f"file '{p.resolve()}'" for p in clip_paths))
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    list_file.unlink()
    return output_path


def _anchored_cut(v_start: float, v_end: float, anchor: float,
                  trim: float, intro_floor: float = 0.0) -> tuple[float, float]:
    """Anchor-centered cut window, clamped to the rated span [v_start, v_end].

    If the span is narrower than ``trim``, slide past the span edges rather
    than shortening the cut. ``intro_floor`` (the opening title-card length,
    0 on all but the first clip) sets a minimum duration so the full opener
    plays. Shared by the reviewed and autonomous burn paths so both produce
    equivalently-paced cuts. Returns ``(cut_start, duration)``.
    """
    half = trim / 2
    cut_start = max(v_start, anchor - half)
    cut_end = min(v_end, anchor + half)
    if cut_end - cut_start < trim:
        cut_start = max(0, anchor - half)
        cut_end = anchor + half
    duration = cut_end - cut_start
    if intro_floor > 0 and duration < intro_floor:
        duration = intro_floor
    return cut_start, duration


def _version_stamp(video_dir: Path) -> str:
    """``<ridedate>_<HHMMSS>`` — ride date from the folder name (not today's
    date), edit time from the clock, so re-edits of one ride sort together."""
    from datetime import datetime
    return f"{video_dir.name.replace('-', '')}_{datetime.now().strftime('%H%M%S')}"


def build_segment_grades(
    shots: list[tuple[str, float]],
    video_dir: Path,
    look: str,
    strength: float,
    wb: str,
) -> dict[int, str]:
    """Measure every shot's balance once and return {index: ffmpeg filter chain}.

    `shots` is [(clip_name, anchor_video_secs), ...] — the two paths into the
    burn carry segments as objects and as dicts, so this takes the flattened
    form both can produce.

    Measurement samples a few frames around each segment's anchor, so this
    costs a handful of cheap ffmpeg seeks per clip in the cut rather than
    anything per-frame.

    Returns an empty dict when grading is off, which the burn loop reads as
    "pass no filter".
    """
    from .grade import build_filter, measure_shot

    if look == "none" and wb == "off":
        return {}

    balances: dict[int, object] = {}
    if wb == "shot":
        print(f"\nMeasuring shot balance for {len(shots)} segments...")
        for i, (clip_name, anchor) in enumerate(shots):
            balances[i] = measure_shot(video_dir / clip_name, anchor)

    grades = {
        i: build_filter(balances.get(i), look=look, strength=strength)
        for i in range(len(shots))
    }
    active = sum(1 for v in grades.values() if v)
    print(f"  Grade: look={look} @ {strength:.0%}, wb={wb} "
          f"({active}/{len(shots)} segments)")
    return grades


def compose_from_selections(
    selections_path: Path,
    video_dir: Path,
    fit_path: Path,
    output_dir: Path,
    offset: float = 0.0,
    layout: str = "landscape",
    encode_preset: str = "master",
    trim_secs: float = 3.0,
    include_outro: bool = True,
    outro_crossfade_secs: float = 3.0,
    outro_lead_in_secs: float = 2.0,
    include_intro: bool = True,
    intro_secs: float = intro_styles.DEFAULT_INTRO_SECS,
    intro_style: str = intro_styles.DEFAULT_STYLE,
    intro_reveal_secs: float = 0.0,
    origin: str | None = None,
    destination: str | None = None,
    road: str | None = None,
    crew: str | None = None,
    subtitle: str | None = None,
    lockup: str | None = None,
    far_pin: bool = False,
    grade_look: str = "none",
    grade_strength: float = 0.35,
    grade_wb: str = "off",
) -> Path:
    """Compose a highlight video from a selections file.

    Accepts either format transparently:

    1. selected_candidates.json — flat list of selection dicts produced
       by the Streamlit reviewer or the unified web review's "Save".
    2. moments.json — full MomentProposal list. Items with status in
       {approved, auto} are composed; others are skipped. user_anchor_
       override / user_in_out (the unified reviewer's hand edits) win
       over the detector defaults.

    trim_secs: width of the final cut around each candidate's anchor.
        Defaults to 3s. Set to 0 to use each candidate's full review-
        span window. A per-moment final_trim_secs in moments.json
        overrides this argument.
    """
    from .burn_overlay import build_renderer, burn_overlay
    from .fit_parser import parse_fit

    selections_path = Path(selections_path)
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work = output_dir / "_work"
    work.mkdir(exist_ok=True)

    raw = json.loads(selections_path.read_text())
    segments, source_kind = _normalize_selection_input(raw)
    print(
        f"Composing from {len(segments)} {source_kind} selections "
        f"(trim={trim_secs}s, layout={layout})"
    )
    # Apply reviewer ratings: legacy 1-5 (3=neutral) and new 1-10 (5-6=neutral).
    # 1-5 scale: +/- 2/rating-step. 1-10 scale: +/- 1/rating-step.
    for seg in segments:
        rating = seg.get("rating", 0)
        if rating >= 1:
            if rating <= 5:
                seg["score"] = seg.get("score", 0) + (rating - 3) * 2.0
            else:
                seg["score"] = seg.get("score", 0) + (rating - 5.5) * 1.0
    # Sort chronologically for the final cut
    segments.sort(key=lambda s: s["ride_time_secs"])

    ride = parse_fit(fit_path)

    # Resolve per-ride lockup / recap-card strings — same precedence as
    # compose_highlight: explicit override → GPS-derived → design defaults.
    lockup_string, outro_kwargs = resolve_lockup_strings(
        ride,
        origin_override=origin,
        destination_override=destination,
        road_override=road,
        crew_override=crew,
        subtitle_override=subtitle,
        lockup_override=lockup,
        far_pin=far_pin,
    )

    renderers: dict[str, "OverlayRenderer"] = {}

    from .gopro_meta import extract_all
    adjusted_clips = extract_all(video_dir)
    clip_by_name = {c.path.name: c for c in adjusted_clips}

    grades = build_segment_grades(
        [(s["clip_name"],
          s.get("anchor_video_secs") or (s["video_start"] + s["video_end"]) / 2)
         for s in segments],
        video_dir, grade_look, grade_strength, grade_wb,
    )

    clips = []
    for i, seg in enumerate(segments):
        notes = seg.get("notes", "")[:60]
        print(f"\n[{i+1}/{len(segments)}] {notes}")
        source = video_dir / seg["clip_name"]
        out = work / f"sel_{layout}_{i:03d}.mp4"

        v_start = seg["video_start"]
        v_end = seg["video_end"]

        # Per-moment trim override (set by the unified reviewer) wins
        # over the CLI-level trim_secs default.
        per_seg_trim = seg.get("final_trim_secs")
        active_trim = float(per_seg_trim) if per_seg_trim else trim_secs

        # Last segment gets a lead-in so the final crossfade has clean
        # video to fade out *of*, not just the crossfade itself.
        is_last = i == len(segments) - 1
        if is_last and include_outro and active_trim > 0:
            active_trim = active_trim + outro_lead_in_secs

        # First clip carries the opening blur→title card; its cut is
        # lengthened (via intro_floor) so the full opener plays.
        seg_intro = intro_secs if (i == 0 and include_intro) else 0.0

        # Pick the cut window:
        #   active_trim > 0: tight cut centered on anchor_video_secs (or
        #     review-span midpoint if no anchor). Clamped to [v_start, v_end]
        #     so we don't leave the rated region.
        #   active_trim == 0: use the full review span (legacy behavior).
        if active_trim > 0:
            anchor = seg.get("anchor_video_secs")
            if anchor is None:
                anchor = (v_start + v_end) / 2
            cut_start, duration = _anchored_cut(
                v_start, v_end, anchor, active_trim, intro_floor=seg_intro,
            )
        else:
            cut_start = v_start
            duration = max(v_end - v_start, seg_intro)

        cache_key = seg["clip_name"]
        if cache_key not in renderers:
            adjusted_clip = clip_by_name.get(seg["clip_name"])
            renderer, _, _ = build_renderer(
                source, str(fit_path), offset, layout,
                ride=ride, clip=adjusted_clip,
                lockup=lockup_string,
            )
            renderers[cache_key] = renderer

        burn_overlay(
            str(source), str(fit_path), str(out),
            offset=offset,
            layout=layout,
            start_offset=cut_start,
            trim_duration=duration,
            renderer=renderers[cache_key],
            ride=ride,
            encode_preset=encode_preset,
            portrait_crop_bias=float(seg.get("portrait_crop_bias", 0.0)),
            intro_secs=seg_intro,
            grade=grades.get(i, ""),
            intro_style=intro_style,
            intro_reveal_secs=intro_reveal_secs,
        )
        clips.append(out)

    if include_outro and clips:
        from .intro_outro import (
            compute_ride_stats, crossfade_outro, probe_segment_params, render_outro,
        )
        try:
            params = probe_segment_params(clips[0])
            stats = compute_ride_stats(ride)
            card_path = work / f"sel_{layout}_outro_card.mp4"
            tail_path = work / f"sel_{layout}_tail.mp4"
            print(f"\nRendering recap card + crossfade ({outro_crossfade_secs:.1f}s)...")
            render_outro(card_path, stats, params, duration_secs=outro_crossfade_secs, **outro_kwargs)
            crossfade_outro(clips[-1], card_path, tail_path, params, outro_crossfade_secs)
            clips = [*clips[:-1], tail_path]
        except Exception as exc:
            print(f"  Warning: skipping outro — {exc}")

    final = output_dir / f"highlight_{layout}_selected_{_version_stamp(video_dir)}.mp4"
    concatenate_clips(clips, final)
    print(f"\nOutput: {final} ({final.stat().st_size / 1e6:.1f} MB)")
    return final


def generate_all_candidates(
    video_dir: Path,
    fit_path: Path,
    config: ComposerConfig,
    labels_path: Path | None = None,
) -> tuple[list[Segment], "RideData", list]:
    """Generate fused candidates from all available sources.

    Returns (candidates, ride, synced_clips) so callers can use the
    candidates for review or directly for composition.
    """
    from .fit_parser import parse_fit
    from .gopro_meta import extract_all
    from .sync import sync_all

    video_dir = Path(video_dir)
    fit_path = Path(fit_path)

    ride = parse_fit(fit_path)
    clips = extract_all(video_dir)
    synced_clips = sync_all(
        clips,
        ride,
        config.offset,
        per_clip_offsets=config.per_clip_offsets,
        ride_timezone=config.ride_timezone,
    )
    print(f"Loaded {len(ride.points)} FIT points, {len(synced_clips)} synced clips")

    # Optional: labels
    labels = None
    if labels_path and Path(labels_path).exists():
        labels = json.loads(Path(labels_path).read_text())
        print(f"  Labels: {len(labels)}")

    # Optional: Strava
    strava_efforts = None
    if config.strava_activity_id:
        try:
            from .strava import get_segment_efforts, get_activity_timezone
            strava_efforts = get_segment_efforts(config.strava_activity_id)
            print(f"  Strava: {len(strava_efforts)} segment efforts")
            if not config.ride_timezone_explicit:
                detected_tz = get_activity_timezone(config.strava_activity_id)
                if detected_tz:
                    print(f"  Detected timezone from activity location: {detected_tz}")
                    config.ride_timezone = detected_tz
                    synced_clips = sync_all(
                        clips,
                        ride,
                        config.offset,
                        per_clip_offsets=config.per_clip_offsets,
                        ride_timezone=config.ride_timezone,
                    )
        except Exception as exc:
            print(f"  Warning: couldn't fetch Strava data: {exc}")

    # Optional: configured vision-model sparse scan
    gemini_candidates = None
    if not config.skip_gemini:
        try:
            from .gemini_scan import scan_ride
            gemini_candidates = scan_ride(
                video_dir, synced_clips, ride, config, labels=labels,
            )
        except Exception as exc:
            print(f"  Warning: vision scan failed: {exc}")

    # Generate and fuse
    print("\nGenerating candidates...")
    candidates = _generate_candidates(
        ride, config,
        synced_clips=synced_clips,
        labels=labels,
        strava_efforts=strava_efforts,
        gemini_candidates=gemini_candidates,
    )

    return candidates, ride, synced_clips


def resolve_lockup_strings(
    ride: RideData,
    *,
    origin_override: str | None = None,
    destination_override: str | None = None,
    road_override: str | None = None,
    crew_override: str | None = None,
    subtitle_override: str | None = None,
    lockup_override: str | None = None,
    far_pin: bool = False,
    skip_route_metadata: bool = False,
) -> tuple[str, dict]:
    """Resolve the in-clip lockup band string + recap-card outro kwargs.

    Per-field resolution order: explicit override → GPS-derived value
    (compute_route_metadata) → design-token default. For destination, an
    empty derived value just means "local-loop ride, no destination" and
    the lockup collapses to "{origin} · {road}".

    Returns ``(lockup_string, outro_kwargs)`` where outro_kwargs is
    suitable for ``render_outro(**outro_kwargs)``.
    """
    from .design.tokens import (
        LOCKUP_ORIGIN as _DEF_ORIGIN,
        LOCKUP_ROAD as _DEF_ROAD,
        LOCKUP_CREW as _DEF_CREW,
    )

    derived_origin: str | None = None
    derived_destination: str | None = None
    derived_road: str | None = None
    derived_end_pin: tuple[float, float] | None = None
    ride_end_point: tuple[float, float] | None = None
    ride_farthest_point: tuple[float, float] | None = None
    if not skip_route_metadata:
        try:
            from .route_metadata import compute_route_metadata
            meta = compute_route_metadata(ride)
            if meta.origin and meta.origin != "UNKNOWN" and not meta.origin.startswith("ERR:"):
                derived_origin = meta.origin
            if meta.destination and not meta.destination.startswith("ERR:"):
                derived_destination = meta.destination
            if meta.road and not meta.road.startswith("ERR:"):
                derived_road = meta.road
            derived_end_pin = meta.destination_point
            ride_end_point = meta.end_point
            ride_farthest_point = meta.farthest_point
            print(
                f"  Route metadata: "
                f"{derived_origin or '?'}"
                f"{' → ' + derived_destination if derived_destination else ''}"
                f"  ·  {derived_road or '?'}  "
                f"[{meta.ride_type}, {meta.distance_mi:.1f}mi, "
                f"max {meta.max_dist_from_start_mi:.1f}mi from start, "
                f"end {meta.end_to_start_mi:.1f}mi from start]"
            )
        except Exception as exc:
            print(f"  Route metadata derivation failed ({exc.__class__.__name__}: {exc}); "
                  "falling back to design-token defaults.")

    origin_str = origin_override or derived_origin or _DEF_ORIGIN
    road_str = road_override or derived_road or _DEF_ROAD
    crew_str = crew_override or _DEF_CREW
    destination_str = destination_override or derived_destination or ""

    # Pin placement: use the auto-classifier's derived end pin
    # (farthest_point for out_and_back, end_point for point_to_point,
    # None for local_loop). A --destination override only changes the
    # label text, not the geometry — the user names the place, but the
    # GPS still says where the ride apex / terminus actually was.
    #
    # When --destination is supplied but the classifier said local_loop
    # (so derived_end_pin is None), the user named a place they rode *to*
    # at the apex of the loop — pin to farthest_point, not end_point
    # (which is back at the start for a loop). Fall through to end_point
    # only if farthest_point is also unavailable.
    end_pin = derived_end_pin
    if end_pin is None and destination_override:
        end_pin = ride_farthest_point or ride_end_point
    # --far-pin: pin the recap end marker at the ride apex (farthest from
    # start) rather than the terminus. For point-to-point rides whose
    # meaningful destination is the turnaround, not where the GPS stopped.
    if far_pin and ride_farthest_point is not None:
        end_pin = ride_farthest_point
    if destination_override and derived_end_pin is None \
            and ride_farthest_point is None and ride_end_point is None:
        print("  WARN: --destination override given but no GPS point available; "
              "recap pin will use the legacy farthest-from-start fallback.")

    if lockup_override is not None:
        lockup_string = lockup_override
    elif destination_str:
        # In-clip lockup uses the destination (where the ride was going),
        # not the start. Recap card still shows "{origin} → {destination}"
        # so the journey reads in full at the end.
        lockup_string = f"{destination_str.upper()}   ·   {road_str}"
    else:
        lockup_string = f"{origin_str}   ·   {road_str}"

    outro_kwargs: dict = {
        "origin": origin_str,
        "destination": destination_str,
        "road": road_str,
        "crew": crew_str,
    }
    if subtitle_override is not None:
        outro_kwargs["subtitle"] = subtitle_override
    if end_pin is not None:
        outro_kwargs["end_pin"] = end_pin

    return lockup_string, outro_kwargs


def compose_highlight(
    video_dir: Path,
    fit_path: Path,
    output_dir: Path,
    config: ComposerConfig = ComposerConfig(),
    labels_path: Path | None = None,
    precomputed_candidates: list[Segment] | None = None,
    encode_preset: str = "master",
) -> dict[str, Path]:
    """Compose landscape + portrait highlight videos.

    Works with just video_dir + fit_path. Labels and Strava are optional.
    If precomputed_candidates are provided (e.g. from the reviewer UI),
    skips candidate generation.

    Returns {"landscape": Path, "portrait": Path}.
    """
    from .burn_overlay import build_renderer, burn_overlay
    from .fit_parser import parse_fit
    from .gopro_meta import extract_all

    video_dir = Path(video_dir)
    fit_path = Path(fit_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work = output_dir / "_work"
    work.mkdir(exist_ok=True)

    # Parse FIT once — shared across all segments and both layouts
    ride = parse_fit(fit_path)

    # Resolve per-ride lockup / recap-card strings. Explicit overrides
    # on config win; otherwise we derive origin/destination/road from GPS
    # (compute_route_metadata) so the title reflects the actual ride, not
    # the hardcoded default. Falls back to design-token defaults on failure.
    lockup_string, outro_kwargs = resolve_lockup_strings(
        ride,
        origin_override=config.origin,
        destination_override=config.destination,
        road_override=config.road,
        crew_override=config.crew,
        subtitle_override=config.subtitle,
        lockup_override=config.lockup,
        far_pin=config.far_pin,
    )
    clips = extract_all(video_dir)  # chapter-adjusted timestamps

    # Lookup: clip filename → chapter-adjusted GoProClip
    clip_by_name = {c.path.name: c for c in clips}

    if precomputed_candidates is not None:
        all_candidates = precomputed_candidates
        print(f"Using {len(all_candidates)} precomputed candidates")
    else:
        all_candidates, _, _ = generate_all_candidates(
            video_dir, fit_path, config, labels_path=labels_path,
        )

    # Select for each format from the shared candidate pool
    landscape_segs = select_segments(
        config.landscape_duration, config,
        ride=ride,
        precomputed_candidates=all_candidates,
        layout="landscape",
    )
    print(f"\nLandscape ({len(landscape_segs)} selected, "
          f"~{config.segment_duration:.0f}s each):")
    for seg in landscape_segs:
        notes = seg.label.get("notes", "")[:50]
        src = f"[{'+'.join(seg.sources)}]" if seg.sources else f"[{seg.source}]"
        print(f"  {seg.ride_time_secs:>7.0f}s  score={seg.score:5.1f}  {src} {notes}")

    portrait_segs = None
    if not config.landscape_only:
        portrait_segs = select_segments(
            config.portrait_duration, config,
            ride=ride,
            precomputed_candidates=all_candidates,
            layout="portrait",
        )
        print(f"\nPortrait ({len(portrait_segs)} selected):")
        for seg in portrait_segs:
            notes = seg.label.get("notes", "")[:50]
            src = f"[{'+'.join(seg.sources)}]" if seg.sources else f"[{seg.source}]"
            print(f"  {seg.ride_time_secs:>7.0f}s  score={seg.score:5.1f}  {src} {notes}")

    # Collect training data for the learned ranker
    try:
        from .learned_ranker import collect_from_segments
        ride_id = Path(video_dir).name  # e.g. "2026-04-14"
        all_selected = list(landscape_segs)
        if portrait_segs:
            all_selected.extend(portrait_segs)
        collect_from_segments(all_candidates, all_selected, ride_id)
    except Exception as exc:
        print(f"  Warning: couldn't collect training data: {exc}")

    results = {}

    def _burn_segments(segs, layout, prefix):
        """Burn overlay for a list of segments, reusing renderer per source clip.

        Each segment is cut to config.segment_duration around its anchor —
        the same logic compose_from_selections uses for hand-picked
        selections — so the autonomous and reviewed paths produce
        equivalently-paced cuts. Without this, Gemini-rated 10-20s rubric
        windows would be burned full-width and the final reel would
        overshoot the duration target by 3-4x.
        """
        renderers: dict[str, object] = {}
        burned = []
        last_idx = len(segs) - 1
        outro_extra = (
            config.outro_lead_in_secs if config.include_outro else 0.0
        )
        grades = build_segment_grades(
            [(s.clip_name, s.anchor_video_secs or (s.video_start + s.video_end) / 2)
             for s in segs],
            video_dir, config.grade_look, config.grade_strength, config.grade_wb,
        )
        for i, seg in enumerate(segs):
            print(f"\n[{i+1}/{len(segs)}] {seg.label.get('notes', '')[:60]}")
            source = video_dir / seg.clip_name
            out = work / f"{prefix}_{i:03d}.mp4"

            trim = config.segment_duration + (outro_extra if i == last_idx else 0.0)
            anchor = seg.anchor_video_secs or (seg.video_start + seg.video_end) / 2
            # First clip carries the opening blur→title card; intro_floor
            # lengthens its cut so the full opener plays.
            intro_secs = config.intro_secs if (i == 0 and config.include_intro) else 0.0
            cut_start, duration = _anchored_cut(
                seg.video_start, seg.video_end, anchor, trim, intro_floor=intro_secs,
            )

            # Reuse renderer for same source clip + layout
            cache_key = seg.clip_name
            # Per-clip offset wins over the global default. Multi-recording
            # rides drift differently per chapter (RTC auto-correct between
            # sessions), so a single global offset misaligns later clips.
            clip_offset = (
                (config.per_clip_offsets or {}).get(seg.clip_name, config.offset)
            )
            if cache_key not in renderers:
                # Pass chapter-adjusted clip so multi-file recordings
                # get correct telemetry alignment
                adjusted_clip = clip_by_name.get(seg.clip_name)
                renderer_obj, _, _ = build_renderer(
                    source, str(fit_path), clip_offset, layout,
                    ride=ride, clip=adjusted_clip,
                    lockup=lockup_string,
                )
                renderers[cache_key] = renderer_obj

            burn_overlay(
                str(source), str(fit_path), str(out),
                offset=clip_offset,
                layout=layout,
                start_offset=cut_start,
                trim_duration=duration,
                renderer=renderers[cache_key],
                ride=ride,
                encode_preset=encode_preset,
                portrait_crop_bias=seg.portrait_crop_bias,
                intro_secs=intro_secs,
                grade=grades.get(i, ""),
                intro_style=config.intro_style,
                intro_reveal_secs=config.intro_reveal_secs,
            )
            burned.append(out)
        return burned

    _stamp = _version_stamp(video_dir)

    # ── Outro stats (computed once, used by both layouts) ───────
    outro_stats = None
    if config.include_outro:
        from .intro_outro import compute_ride_stats
        outro_stats = compute_ride_stats(ride)
        print(
            f"\nOutro stats: {outro_stats.distance_mi:.1f}mi · "
            f"{int(round(outro_stats.elev_gain_ft)):,}ft · "
            f"{_format_bookend_duration(outro_stats.moving_secs)} · "
            f"top {outro_stats.top_speed_mph:.0f}mph"
        )

    def _append_outro_crossfade(clips: list[Path], layout_name: str, prefix: str) -> list[Path]:
        if not clips or not config.include_outro or outro_stats is None:
            return clips
        from .intro_outro import (
            crossfade_outro, probe_segment_params, render_outro,
        )
        try:
            params = probe_segment_params(clips[0])
        except Exception as exc:
            print(f"  Warning: skipping outro — probe failed: {exc}")
            return clips
        card_path = work / f"{prefix}_outro_card.mp4"
        tail_path = work / f"{prefix}_tail.mp4"
        xd = config.outro_crossfade_secs
        print(f"  Rendering {layout_name} recap card ({xd:.1f}s)...")
        render_outro(card_path, outro_stats, params, duration_secs=xd, **outro_kwargs)
        print("  Crossfading last segment into recap...")
        crossfade_outro(clips[-1], card_path, tail_path, params, xd)
        return [*clips[:-1], tail_path]

    # ── Landscape pipeline ──────────────────────────────────────
    print("\n=== Landscape (16:9) ===")
    landscape_clips = _burn_segments(landscape_segs, "landscape", "land")
    landscape_clips = _append_outro_crossfade(landscape_clips, "landscape", "land")

    if landscape_clips:
        final = output_dir / f"highlight_landscape_{_stamp}.mp4"
        concatenate_clips(landscape_clips, final)
        print(f"\nLandscape: {final} ({final.stat().st_size / 1e6:.1f} MB)")
        results["landscape"] = final

    # ── Portrait pipeline ───────────────────────────────────────
    if not config.landscape_only and portrait_segs:
        print("\n=== Portrait (9:16) ===")
        portrait_clips = _burn_segments(portrait_segs, "portrait", "port")
        portrait_clips = _append_outro_crossfade(portrait_clips, "portrait", "port")

        if portrait_clips:
            final = output_dir / f"highlight_portrait_{_stamp}.mp4"
            concatenate_clips(portrait_clips, final)
            print(f"\nPortrait: {final} ({final.stat().st_size / 1e6:.1f} MB)")
            results["portrait"] = final

    return results


def _format_bookend_duration(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    return f"{h}:{m:02d}" if h else f"0:{m:02d}"
