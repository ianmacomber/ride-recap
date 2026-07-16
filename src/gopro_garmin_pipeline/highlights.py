"""Detect interesting ride segments using heuristic rules on FIT metrics."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum

from .config import get_settings
from .fit_parser import RideData, RidePoint


class HighlightReason(Enum):
    POWER_SPIKE = "power_spike"
    SPEED_SPIKE = "speed_spike"
    HR_SPIKE = "hr_spike"
    CLIMB = "climb"
    SPRINT = "sprint"  # high power + high cadence


@dataclass
class Highlight:
    """A detected interesting segment of the ride."""

    start_time: dt.datetime
    end_time: dt.datetime
    reason: HighlightReason
    score: float  # 0-1, higher = more interesting
    peak_value: float  # the metric value that triggered detection
    peak_time: dt.datetime | None = None  # exact moment the peak occurred
    description: str = ""

    @property
    def duration(self) -> dt.timedelta:
        return self.end_time - self.start_time


@dataclass
class HighlightConfig:
    """Thresholds for highlight detection.

    The spike thresholds default to the athlete zones in ``config.Settings``
    (i.e. ``FTP`` / ``MAX_HEART_RATE`` in ``.env``, scaled), so setting your
    FTP once tunes detection to your fitness. Pass any field explicitly to
    override it for a single run.
    """

    # Power thresholds (watts) — defaults to ~1.45x FTP
    power_spike_threshold: float = field(
        default_factory=lambda: get_settings().power_spike_threshold)
    power_spike_duration_secs: float = 10  # sustained for at least this long

    # Speed thresholds (m/s) — ~27 mph / ~43 km/h
    speed_spike_threshold: float = field(
        default_factory=lambda: get_settings().speed_spike_threshold)
    speed_spike_duration_secs: float = 10

    # Heart rate thresholds (bpm) — defaults to ~87% of max HR
    hr_spike_threshold: float = field(
        default_factory=lambda: get_settings().hr_spike_threshold)
    hr_spike_duration_secs: float = 30

    # Climbing: altitude gain over a window
    climb_gain_threshold: float = 15  # meters gained in window
    climb_window_secs: float = 120

    # Sprint detection: power + cadence combo
    sprint_power_threshold: float = 400
    sprint_cadence_threshold: int = 100
    sprint_duration_secs: float = 5

    # Padding added before/after each highlight for video context
    padding_before_secs: float = 5
    padding_after_secs: float = 10

    # Minimum gap between highlights before they get merged
    merge_gap_secs: float = 15

    # Sanity caps — drop highlights whose peak exceeds physically
    # plausible bounds. The Garmin Edge 540 occasionally emits GPS
    # glitches that produce impossible speed spikes (e.g. 24 m/s on a
    # bike, ~54 mph), and any peak past these thresholds is almost
    # certainly bad data, not a signal worth surfacing.
    speed_cap_mps: float = 22.0   # ~49 mph — cyclists can hit this on steep descents but not casually
    power_cap_w: float = 1500.0   # higher than any sustained road effort
    hr_cap_bpm: float = 220.0     # hard physiological ceiling


def _detect_sustained(
    points: list[RidePoint],
    metric_fn,
    threshold: float,
    min_duration_secs: float,
    reason: HighlightReason,
    sanity_cap: float | None = None,
) -> list[Highlight]:
    """Detect sustained periods where a metric exceeds a threshold.

    sanity_cap: if set, individual sample values above this are treated
    as bad data (e.g. GPS glitches reading 54 mph on a road bike) and
    ignored — they don't end the run, they just don't update the peak.
    """
    highlights = []
    run_start = None
    peak = 0.0
    peak_time: dt.datetime | None = None

    for point in points:
        value = metric_fn(point)
        if sanity_cap is not None and value is not None and value > sanity_cap:
            value = None  # treat as missing
        if value is not None and value >= threshold:
            if run_start is None:
                run_start = point.timestamp
                peak = value
                peak_time = point.timestamp
            elif value > peak:
                peak = value
                peak_time = point.timestamp
        else:
            if run_start is not None:
                duration = (point.timestamp - run_start).total_seconds()
                if duration >= min_duration_secs:
                    # Score based on how far above threshold the peak was
                    score = min(1.0, (peak - threshold) / threshold + 0.5)
                    highlights.append(
                        Highlight(
                            start_time=run_start,
                            end_time=point.timestamp,
                            reason=reason,
                            score=score,
                            peak_value=peak,
                            peak_time=peak_time,
                        )
                    )
            run_start = None
            peak = 0.0
            peak_time = None

    return highlights


def _detect_climbs(
    points: list[RidePoint], config: HighlightConfig
) -> list[Highlight]:
    """Detect climbing segments based on altitude gain over a sliding window."""
    highlights = []
    if not points:
        return highlights

    window_start = 0
    for i, point in enumerate(points):
        if point.altitude is None:
            continue

        # Advance window start
        while window_start < i:
            dt_secs = (point.timestamp - points[window_start].timestamp).total_seconds()
            if dt_secs <= config.climb_window_secs:
                break
            window_start += 1

        start_alt = points[window_start].altitude
        if start_alt is not None and point.altitude is not None:
            gain = point.altitude - start_alt
            if gain >= config.climb_gain_threshold:
                score = min(1.0, gain / (config.climb_gain_threshold * 3))
                highlights.append(
                    Highlight(
                        start_time=points[window_start].timestamp,
                        end_time=point.timestamp,
                        reason=HighlightReason.CLIMB,
                        score=score,
                        peak_value=gain,
                        peak_time=point.timestamp,  # top of the climb window
                    )
                )

    return highlights


def _detect_sprints(
    points: list[RidePoint], config: HighlightConfig
) -> list[Highlight]:
    """Detect sprint efforts (high power + high cadence)."""
    highlights = []
    run_start = None
    peak_power = 0.0
    peak_time: dt.datetime | None = None

    for point in points:
        power = point.power or 0
        cadence = point.cadence or 0
        if power >= config.sprint_power_threshold and cadence >= config.sprint_cadence_threshold:
            if run_start is None:
                run_start = point.timestamp
                peak_power = power
                peak_time = point.timestamp
            elif power > peak_power:
                peak_power = power
                peak_time = point.timestamp
        else:
            if run_start is not None:
                duration = (point.timestamp - run_start).total_seconds()
                if duration >= config.sprint_duration_secs:
                    score = min(1.0, peak_power / 800)
                    highlights.append(
                        Highlight(
                            start_time=run_start,
                            end_time=point.timestamp,
                            reason=HighlightReason.SPRINT,
                            score=score,
                            peak_value=peak_power,
                            peak_time=peak_time,
                        )
                    )
            run_start = None
            peak_power = 0.0
            peak_time = None

    return highlights


def _merge_highlights(
    highlights: list[Highlight], merge_gap_secs: float
) -> list[Highlight]:
    """Merge overlapping or adjacent highlights."""
    if not highlights:
        return []

    sorted_hl = sorted(highlights, key=lambda h: h.start_time)
    merged = [sorted_hl[0]]

    for hl in sorted_hl[1:]:
        prev = merged[-1]
        gap = (hl.start_time - prev.end_time).total_seconds()
        if gap <= merge_gap_secs:
            # Merge across reasons by SCORE, not peak_value. peak_value's
            # unit varies by reason (m/s for speed, meters of gain for
            # climb, watts for power). Picking max(peak_value) silently
            # mixes units — a 24-meter climb gain would override a
            # 13.5 m/s speed peak, and the merged highlight would be
            # anchored on the climb's top instead of the speed peak.
            # score is the unit-free signal we already use to pick reason;
            # use the same side's peak_time and peak_value.
            if prev.score >= hl.score:
                winner, _loser = prev, hl
            else:
                winner, _loser = hl, prev
            # Name the merged highlight. Most reason pairs read fine as
            # "merged: A + B" (e.g. power + speed), but climb + speed_spike
            # looks self-contradictory. It's really one over-the-top event:
            # the climb window ends at a crest, then the fast descent trips
            # the speed spike within merge_gap. Label it by what physically
            # happened, ordered by when each side peaked (start_time if a
            # peak_time is missing).
            if {prev.reason, hl.reason} == {
                HighlightReason.CLIMB, HighlightReason.SPEED_SPIKE
            }:
                climb_hl = prev if prev.reason == HighlightReason.CLIMB else hl
                speed_hl = hl if climb_hl is prev else prev
                climb_t = climb_hl.peak_time or climb_hl.start_time
                speed_t = speed_hl.peak_time or speed_hl.start_time
                description = (
                    "descent over crest" if climb_t <= speed_t
                    else "sprint into climb"
                )
            else:
                description = f"merged: {prev.reason.value} + {hl.reason.value}"
            merged[-1] = Highlight(
                start_time=prev.start_time,
                end_time=max(prev.end_time, hl.end_time),
                reason=winner.reason,
                score=winner.score,
                peak_value=winner.peak_value,
                peak_time=winner.peak_time,
                description=description,
            )
        else:
            merged.append(hl)

    return merged


def detect_highlights(
    ride: RideData, config: HighlightConfig | None = None
) -> list[Highlight]:
    """Detect all interesting segments in a ride.

    Args:
        ride: Parsed FIT ride data.
        config: Detection thresholds. Uses defaults if None.

    Returns:
        List of highlights sorted by start time, with padding applied.
    """
    if config is None:
        config = HighlightConfig()

    points = ride.points
    all_highlights: list[Highlight] = []

    # Power spikes (requires power meter)
    if ride.has_power:
        all_highlights.extend(
            _detect_sustained(
                points,
                lambda p: p.power,
                config.power_spike_threshold,
                config.power_spike_duration_secs,
                HighlightReason.POWER_SPIKE,
                sanity_cap=config.power_cap_w,
            )
        )

    # Speed spikes
    all_highlights.extend(
        _detect_sustained(
            points,
            lambda p: p.speed,
            config.speed_spike_threshold,
            config.speed_spike_duration_secs,
            HighlightReason.SPEED_SPIKE,
            sanity_cap=config.speed_cap_mps,
        )
    )

    # HR spikes
    all_highlights.extend(
        _detect_sustained(
            points,
            lambda p: p.heart_rate,
            config.hr_spike_threshold,
            config.hr_spike_duration_secs,
            HighlightReason.HR_SPIKE,
            sanity_cap=config.hr_cap_bpm,
        )
    )

    # Climbs
    all_highlights.extend(_detect_climbs(points, config))

    # Sprints (requires both power meter and cadence sensor)
    if ride.has_power and ride.has_cadence:
        all_highlights.extend(_detect_sprints(points, config))

    # Apply padding
    for hl in all_highlights:
        hl.start_time -= dt.timedelta(seconds=config.padding_before_secs)
        hl.end_time += dt.timedelta(seconds=config.padding_after_secs)

    # Merge nearby highlights
    merged = _merge_highlights(all_highlights, config.merge_gap_secs)

    # Clamp to ride bounds
    if ride.start_time and ride.end_time:
        for hl in merged:
            hl.start_time = max(hl.start_time, ride.start_time)
            hl.end_time = min(hl.end_time, ride.end_time)

    return sorted(merged, key=lambda h: h.start_time)
