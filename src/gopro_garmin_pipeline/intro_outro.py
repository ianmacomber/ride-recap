"""Intro stat card + outro recap card for highlight reels.

Each card is rendered as an MP4 whose video/audio params match a probed
reference segment, so the existing concat demuxer with -c copy stitches
intro + segments + outro in one pass.

Design language sourced from `design/tokens.json` — near-black bg, cream
Inter Bold title, sage Times-italic subtitle, Barlow Bold numerics, route
trace as the hero (cream + sage glow, sage start dot, mint turn dot).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

# Pull design tokens from the project's single source of truth.
from .design.tokens import (
    SAGE, MINT, CREAM, CARD_BG, TEXT_MUTED,
    FONT_NUMERIC, FONT_BODY, FONT_SERIF,
    LOCKUP_ROAD, LOCKUP_CREW, LOCKUP_ORIGIN, LOCKUP_SUBTITLE,
)

from .fit_parser import RideData
from .utils import M_TO_FT, M_TO_MILES, MS_TO_MPH

_FONT_DIR = Path(__file__).parent.parent.parent / "assets" / "fonts"
_FONT_BOLD = str(_FONT_DIR / "BarlowCondensed-Bold.ttf")
_FONT_REG = str(_FONT_DIR / "BarlowCondensed-Regular.ttf")

# Last-resort outro-card defaults, from design/tokens.json. Normally the CLI
# resolves these per-ride (flag → prompt → GPS); these only apply when nothing
# else supplied a value. Set them in tokens.json, not here.
_DEFAULT_OUTRO_SUBTITLE = LOCKUP_SUBTITLE
_DEFAULT_ORIGIN = LOCKUP_ORIGIN


@dataclass
class RideStats:
    distance_mi: float
    elev_gain_ft: float
    moving_secs: float
    top_speed_mph: float
    avg_speed_mph: float
    avg_power_w: float
    has_power: bool
    date_str: str
    route: list[tuple[float, float]]
    # Optional explicit lat/lon for the destination pin on the recap card.
    # When provided, the end pin lands on the polyline vertex nearest this
    # point. When None, the route block falls back to "farthest polyline
    # vertex from start" — appropriate for out-and-back rides where the
    # apex is the destination.
    end_pin: tuple[float, float] | None = None


def compute_ride_stats(ride: RideData) -> RideStats:
    points = ride.points

    # Distance: prefer the session total (Garmin's authoritative number)
    # over max(record.distance), which can lag the final point.
    if ride.session_distance_m is not None:
        distance_mi = ride.session_distance_m * M_TO_MILES
    else:
        dists = [p.distance for p in points if p.distance is not None]
        distance_mi = max(dists) * M_TO_MILES if dists else 0.0

    alts = [p.altitude for p in points if p.altitude is not None]
    elev_gain_ft = 0.0
    if len(alts) > 5:
        smooth = []
        for i in range(len(alts)):
            window = alts[max(0, i - 2):i + 3]
            smooth.append(sum(window) / len(window))
        for i in range(1, len(smooth)):
            d = smooth[i] - smooth[i - 1]
            if d > 0:
                elev_gain_ft += d
        elev_gain_ft *= M_TO_FT

    # Moving time: prefer FIT session total_timer_time (excludes auto-paused
    # stops) over elapsed wall time. Falls back to elapsed if missing.
    if ride.total_timer_secs is not None:
        moving_secs = ride.total_timer_secs
    elif ride.start_time and ride.end_time:
        moving_secs = (ride.end_time - ride.start_time).total_seconds()
    else:
        moving_secs = 0.0

    if ride.session_max_speed_ms is not None:
        top_speed_mph = ride.session_max_speed_ms * MS_TO_MPH
    else:
        speeds = [p.speed * MS_TO_MPH for p in points if p.speed]
        top_speed_mph = max(speeds) if speeds else 0.0

    if ride.session_avg_speed_ms is not None:
        avg_speed_mph = ride.session_avg_speed_ms * MS_TO_MPH
    elif moving_secs > 0:
        avg_speed_mph = distance_mi / (moving_secs / 3600)
    else:
        avg_speed_mph = 0.0

    powers = [p.power for p in points if p.power]
    avg_power_w = sum(powers) / len(powers) if powers else 0.0

    date_str = ride.start_time.strftime("%b %d, %Y").upper() if ride.start_time else ""
    route = [(p.lat, p.lon) for p in points if p.lat is not None and p.lon is not None]

    return RideStats(
        distance_mi=distance_mi,
        elev_gain_ft=elev_gain_ft,
        moving_secs=moving_secs,
        top_speed_mph=top_speed_mph,
        avg_speed_mph=avg_speed_mph,
        avg_power_w=avg_power_w,
        has_power=bool(powers),
        date_str=date_str,
        route=route,
    )


def _format_duration(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    if h > 0:
        return f"{h}:{m:02d}"
    return f"0:{m:02d}"


def _fit_text_width(strings, *, base_size: int, font_path: str,
                    draw: ImageDraw.ImageDraw, max_width: int,
                    min_size: int = 20) -> ImageFont.FreeTypeFont:
    """Return the largest font (≤ base_size) such that every line in
    ``strings`` fits within ``max_width`` when rendered with it.

    Used by the recap card title so long destinations like
    "CENTRAL PARK" don't get clipped at the column edge. Shrinks in
    8% steps from base_size down to min_size; returns min_size if even
    that still overflows (caller has to live with the clip).
    """
    size = base_size
    while size > min_size:
        font = ImageFont.truetype(font_path, size)
        widest = max(draw.textbbox((0, 0), t, font=font)[2] for t in strings)
        if widest <= max_width:
            return font
        size = max(min_size, int(size * 0.92))
    return ImageFont.truetype(font_path, min_size)


@dataclass
class _SegmentParams:
    width: int
    height: int
    fps_frac: str
    fps: float
    pix_fmt: str
    sample_rate: int
    channels: int


def _probe_kv(path: Path, stream: str, fields: list[str]) -> dict[str, str]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream,
         "-show_entries", f"stream={','.join(fields)}",
         "-of", "default=noprint_wrappers=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    result = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def probe_segment_params(path: Path) -> _SegmentParams:
    """Probe a burned segment so the cards encode with matching params."""
    v = _probe_kv(path, "v:0", ["width", "height", "r_frame_rate", "pix_fmt"])
    fps_frac = v["r_frame_rate"]
    n, d = fps_frac.split("/")
    fps = float(n) / float(d)

    a = _probe_kv(path, "a:0", ["sample_rate", "channels"])
    sr = int(a.get("sample_rate", 48000))
    ch = int(a.get("channels", 2))

    return _SegmentParams(
        width=int(v["width"]), height=int(v["height"]),
        fps_frac=fps_frac, fps=fps, pix_fmt=v["pix_fmt"],
        sample_rate=sr, channels=ch,
    )


def _value_height(font, text="0123456789") -> int:
    """Return the rendered glyph height for a representative numeric string."""
    bbox = font.getbbox(text)
    return bbox[3] - bbox[1]


def _draw_value_label(draw, value, label, f_value, f_label, cx, cy_value,
                      gap, value_fill, label_fill):
    """Draw value centered at (cx, cy_value); label centered below the value."""
    draw.text((cx, cy_value), value, font=f_value, fill=value_fill, anchor="mm")
    half_h = _value_height(f_value) / 2
    draw.text((cx, cy_value + half_h + gap), label, font=f_label,
              fill=label_fill, anchor="mt")


def _project_route(route, box_x, box_y, box_w, box_h, pad_x, pad_y):
    if len(route) < 2:
        return []
    lats = [r[0] for r in route]
    lons = [r[1] for r in route]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    lat_r = (max_lat - min_lat) or 1e-6
    lon_r = (max_lon - min_lon) or 1e-6
    import math
    cos_lat = math.cos(math.radians((min_lat + max_lat) / 2))
    aspect = (lon_r * cos_lat) / lat_r
    dw, dh = box_w - 2 * pad_x, box_h - 2 * pad_y
    ba = dw / dh if dh else 1
    if aspect > ba:
        xs, ys = dw, dw / aspect
        xo, yo = pad_x, pad_y + (dh - ys) / 2
    else:
        ys, xs = dh, dh * aspect
        xo, yo = pad_x + (dw - xs) / 2, pad_y
    return [
        (
            int(box_x + xo + ((lon - min_lon) / lon_r) * xs),
            int(box_y + yo + ((max_lat - lat) / lat_r) * ys),
        )
        for lat, lon in route
    ]


def _draw_intro_frame(width: int, height: int, stats: RideStats, alpha: float) -> Image.Image:
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    portrait = height > width
    sc = (width / 1920) if not portrait else (height / 1920)

    a = int(255 * alpha)
    white = (255, 255, 255, a)
    dim = (180, 180, 180, a)

    if portrait:
        f_date = ImageFont.truetype(_FONT_REG, int(46 * sc))
        f_value = ImageFont.truetype(_FONT_BOLD, int(200 * sc))
        f_label = ImageFont.truetype(_FONT_BOLD, int(40 * sc))
    else:
        f_date = ImageFont.truetype(_FONT_REG, int(40 * sc))
        f_value = ImageFont.truetype(_FONT_BOLD, int(180 * sc))
        f_label = ImageFont.truetype(_FONT_BOLD, int(34 * sc))

    if stats.date_str:
        date_text = "  ".join(stats.date_str)
        draw.text((width // 2, int(height * (0.10 if portrait else 0.16))),
                  date_text, font=f_date, fill=dim, anchor="mt")

    rows = [
        (f"{stats.distance_mi:.2f}", "MILES"),
        (f"{int(round(stats.elev_gain_ft)):,}", "FEET CLIMBED"),
        (_format_duration(stats.moving_secs), "MOVING TIME"),
    ]
    gap = int(40 * sc)

    if portrait:
        block_top = int(height * 0.28)
        block_h = int(height * 0.55)
        slot = block_h / len(rows)
        for i, (val, lbl) in enumerate(rows):
            cy = int(block_top + slot * (i + 0.5))
            _draw_value_label(draw, val, lbl, f_value, f_label,
                              width // 2, cy, gap, white, dim)
    else:
        col_w = width / len(rows)
        cy = int(height * 0.52)
        for i, (val, lbl) in enumerate(rows):
            cx = int((i + 0.5) * col_w)
            _draw_value_label(draw, val, lbl, f_value, f_label,
                              cx, cy, gap, white, dim)

    return img


def _draw_outro_frame(
    width: int,
    height: int,
    stats: RideStats,
    alpha: float,
    destination: str = "",
    subtitle: str = _DEFAULT_OUTRO_SUBTITLE,
    origin: str = _DEFAULT_ORIGIN,
    road: str = LOCKUP_ROAD,
    crew: str = LOCKUP_CREW,
) -> Image.Image:
    """Sage/mint outro card.

    Layout:
      Landscape — left column: title + stats grid (3 cols × 2 rows,
                  column-aligned, AVG WATTS under DISTANCE).
                  Right ~55%: route trace as the hero.
      Portrait — top: title block. Middle: route trace. Bottom: same
                 3×2 stats grid.
    """
    img = Image.new("RGB", (width, height), CARD_BG)
    draw = ImageDraw.Draw(img, "RGBA")
    portrait = height > width
    sc = (height / 2160) if not portrait else (height / 1920)

    def s(v):
        return max(1, int(v * sc))

    a = int(255 * alpha)
    cream = (*CREAM, a)
    muted = (*TEXT_MUTED, a)
    sage = (*SAGE, a)

    margin = s(80)

    # ── Cells (5 stats — column-aligned grid) ────────────────────
    cells = [
        ("DISTANCE",    f"{stats.distance_mi:.2f}",        "MI"),
        ("ELEVATION",   f"{int(round(stats.elev_gain_ft)):,}",   "FT"),
        ("AVG SPEED",   f"{stats.avg_speed_mph:.1f}",      "MPH"),
        ("AVG WATTS",   f"{int(round(stats.avg_power_w))}", "WATTS"),
        ("MOVING TIME", _format_duration(stats.moving_secs), "H:MM"),
    ]
    # Bottom row column-aligns to top: AVG WATTS under DISTANCE,
    # MOVING TIME under ELEVATION, nothing under AVG SPEED.
    positions = [(0, 0), (0, 1), (0, 2),
                 (1, 0), (1, 1)]

    if portrait:
        f_eyebrow = ImageFont.truetype(FONT_BODY, s(38))
        # Title font shrinks to fit the widest line within the column so
        # long destinations (e.g. "CENTRAL PARK") don't get chopped.
        title_lines = [origin]
        if destination:
            title_lines.append(f"→  {destination.upper()}")
        f_title = _fit_text_width(
            title_lines, base_size=s(120), font_path=FONT_BODY,
            draw=draw, max_width=width - 2 * margin, min_size=s(50),
        )
        f_sub = ImageFont.truetype(FONT_SERIF, s(70))
        f_pin = ImageFont.truetype(FONT_BODY, s(34))
        f_foot = ImageFont.truetype(FONT_BODY, s(28))
        f_stat_v = ImageFont.truetype(FONT_NUMERIC, s(96))
        f_stat_l = ImageFont.truetype(FONT_BODY, s(24))
        f_stat_u = ImageFont.truetype(FONT_BODY, s(24))

        ly = s(160)
        if stats.date_str:
            draw.text((margin, ly), stats.date_str, font=f_eyebrow, fill=muted)
            ly += s(70)
        draw.text((margin, ly), origin, font=f_title, fill=cream)
        ly += s(140)
        if destination:
            draw.text((margin, ly), f"→  {destination.upper()}",
                      font=f_title, fill=cream)
            ly += s(160)
        if subtitle:
            draw.text((margin, ly), subtitle, font=f_sub, fill=sage)
            ly += s(140)

        # Route trace block — middle. Trimmed label padding so the trace
        # fills the block's height instead of floating small in the center
        # (portrait frame is narrow, so the trace was width-starved by the
        # full-label horizontal pad). Block height stays at 0.36 — the stats
        # grid + footer below leave no room to grow it.
        rt_x, rt_y = margin, ly
        rt_w = width - 2 * margin
        rt_h = int(height * 0.36)
        _draw_route_block(draw, stats.route, rt_x, rt_y, rt_w, rt_h,
                          sc, cream, sage, (*MINT, a), f_pin,
                          origin, destination, end_pin=stats.end_pin,
                          label_pad_frac=0.8)
        ly = rt_y + rt_h + s(80)

        # Stats grid 3 cols × 2 rows, column-aligned
        ncols = 3
        col_w = (width - 2 * margin) // ncols
        row_h = s(220)
        for (r, c), (sl, sv, su) in zip(positions, cells):
            cx = margin + c * col_w
            cy = ly + r * row_h
            draw.text((cx, cy), sl, font=f_stat_l, fill=muted)
            draw.text((cx, cy + s(36)), sv, font=f_stat_v, fill=cream)
            if su:
                vbox = draw.textbbox((cx, cy + s(36)), sv, font=f_stat_v)
                draw.text((vbox[2] + s(8), cy + s(120)), su,
                          font=f_stat_u, fill=muted)
            draw.rectangle(
                [cx, cy + s(170), cx + col_w - s(20), cy + s(172)],
                fill=(*CREAM, max(0, a - 165)),
            )

        # Footer — no rule above (declutter)
        draw.text((margin, height - s(60)),
                  f"{road}   ·   {crew}",
                  font=f_foot, fill=muted)

    else:
        f_eyebrow = ImageFont.truetype(FONT_BODY, s(38))
        col_w = int(width * 0.45)
        # Title font shrinks to fit the title column (~45% of frame width).
        title_lines = [origin]
        if destination:
            title_lines.append(f"→  {destination.upper()}")
        f_title = _fit_text_width(
            title_lines, base_size=s(150), font_path=FONT_BODY,
            draw=draw, max_width=col_w - margin, min_size=s(60),
        )
        f_sub = ImageFont.truetype(FONT_SERIF, s(70))
        f_pin = ImageFont.truetype(FONT_BODY, s(34))
        f_foot = ImageFont.truetype(FONT_BODY, s(28))
        f_stat_v = ImageFont.truetype(FONT_NUMERIC, s(170))
        f_stat_l = ImageFont.truetype(FONT_BODY, s(30))
        f_stat_u = ImageFont.truetype(FONT_BODY, s(30))
        ly = s(120)
        if stats.date_str:
            draw.text((margin, ly), stats.date_str, font=f_eyebrow, fill=muted)
            ly += s(80)
        draw.text((margin, ly), origin, font=f_title, fill=cream)
        ly += s(170)
        if destination:
            draw.text((margin, ly), f"→  {destination.upper()}",
                      font=f_title, fill=cream)
            ly += s(190)
        if subtitle:
            draw.text((margin, ly), subtitle, font=f_sub, fill=sage)
            ly += s(140)

        # Stats grid 3 cols × 2 rows, column-aligned
        ncols = 3
        grid_col_w = (col_w - margin) // ncols
        row_h = s(280)
        for (r, c), (sl, sv, su) in zip(positions, cells):
            cx = margin + c * grid_col_w
            cy = ly + r * row_h
            draw.text((cx, cy), sl, font=f_stat_l, fill=muted)
            draw.text((cx, cy + s(44)), sv, font=f_stat_v, fill=cream)
            if su:
                vbox = draw.textbbox((cx, cy + s(44)), sv, font=f_stat_v)
                draw.text((vbox[2] + s(12), cy + s(170)), su,
                          font=f_stat_u, fill=muted)
            draw.rectangle(
                [cx, cy + s(245), cx + grid_col_w - s(40), cy + s(247)],
                fill=(*CREAM, max(0, a - 165)),
            )

        # Route trace right side
        rt_x = col_w + s(60)
        rt_y = s(180)
        rt_w = width - rt_x - margin
        rt_h = height - rt_y - s(180)
        _draw_route_block(draw, stats.route, rt_x, rt_y, rt_w, rt_h,
                          sc, cream, sage, (*MINT, a), f_pin,
                          origin, destination, end_pin=stats.end_pin)

        # Footer rule + lockup
        rule_y = height - s(70)
        draw.rectangle([margin, rule_y, width - margin, rule_y + s(2)],
                       fill=(*CREAM, max(0, a - 175)))
        draw.text((margin, rule_y + s(18)),
                  f"{road}   ·   {crew}",
                  font=f_foot, fill=muted)

    return img


def _pin_label_side(coords: list[tuple[int, int]], pin_idx: int,
                    radius_px: int) -> str:
    """Return 'left' or 'right' — whichever side of the pin has less polyline.

    Looks at polyline vertices within ``radius_px`` of the pin in
    screen space and compares how many fall on each side of the pin's
    x coordinate. The label goes on the side with fewer vertices so it
    doesn't overlap the trace. Ties resolve to 'right' (the historical
    default), so straight-line approaches keep the old behavior.

    ``radius_px`` must be passed by the caller and sized against the
    route block — using a fixed literal here would over-cover a small
    portrait block and under-cover a large landscape one.
    """
    if not coords or pin_idx < 0 or pin_idx >= len(coords):
        return "right"
    px, py = coords[pin_idx]
    r2 = radius_px * radius_px
    left = right = 0
    for i, (cx, cy) in enumerate(coords):
        if i == pin_idx:
            continue
        dx = cx - px
        dy = cy - py
        if dx * dx + dy * dy > r2:
            continue
        if dx < -2:
            left += 1
        elif dx > 2:
            right += 1
    return "left" if right > left else "right"


def _clamp_pin_side(side: str, text, dot_x: int, *,
                    font, gap: int, canvas_w: int, edge_pad: int = 0) -> str:
    """Flip ``side`` if drawing on that side would clip the canvas edge.

    ``side`` is the polyline-density hint from ``_pin_label_side``. We
    override it only when the chosen side would push the label past
    ``[edge_pad, canvas_w - edge_pad]`` and the opposite side fits.
    """
    text_w = int(font.getlength(text))
    if side == "left":
        left_edge = dot_x - gap - text_w
        if left_edge < edge_pad and dot_x + gap + text_w <= canvas_w - edge_pad:
            return "right"
    else:
        right_edge = dot_x + gap + text_w
        if right_edge > canvas_w - edge_pad \
                and dot_x - gap - text_w >= edge_pad:
            return "left"
    return side


def _draw_pin_label(draw, text, dot_x, dot_y, *, font, fill, gap, side):
    """Draw a pin label on the requested side of the dot.

    Uses Pillow's text anchor so the label aligns flush against the dot
    regardless of side. The vertical baseline matches the historical
    "top-y at ny - 20" placement.
    """
    if side == "left":
        draw.text((dot_x - gap, dot_y - gap), text, font=font, fill=fill,
                  anchor="rt")
    else:
        draw.text((dot_x + gap, dot_y - gap), text, font=font, fill=fill,
                  anchor="lt")


def _draw_route_block(draw, route, x, y, w, h, sc,
                      line_color, glow_color, end_dot_color, font_pin,
                      origin: str, destination: str,
                      end_pin: tuple[float, float] | None = None,
                      label_pad_frac: float = 1.0):
    """Render the lat/lon route inside (x, y, w, h).

    Cream line with a soft sage glow and a tight black halo. Sage start
    dot pinned to the origin label, mint turn dot pinned to the destination
    label (if provided).

    The end pin lands on the polyline vertex nearest ``end_pin`` (in
    lat/lon space). If ``end_pin`` is None, falls back to the polyline
    vertex farthest from start — the right default for out-and-back
    rides whose apex is the named destination.
    """
    if len(route) < 2:
        return

    def s(v):
        return max(1, int(v * sc))
    # Horizontal pad reserves empty room next to dots at the bbox edge so
    # their labels don't collide with the polyline OR the canvas edge.
    # Vertical pad stays small — labels go left/right, never above/below.
    r_for_dot = max(6, int(w * 0.012))
    label_gap = max(s(20), int(r_for_dot * 2.5))
    longest = max(origin or "", (destination or "").upper(), key=len)
    label_w = int(font_pin.getlength(longest)) if longest else 0
    # label_pad_frac < 1 reserves less horizontal room for labels, letting the
    # trace grow wider/taller. Safe to trim because _clamp_pin_side already
    # flips any edge-clipping label inward; pad_x only keeps the label off the
    # polyline. Portrait passes a smaller frac (narrow frame, trace was
    # width-starved); landscape keeps the full reservation.
    pad_x = max(s(40), int(label_w * label_pad_frac) + label_gap + s(20))
    pad_y = s(40)
    coords = _project_route(route, x, y, w, h, pad_x=pad_x, pad_y=pad_y)
    if not coords:
        return

    # Soft sage glow (drawn as wide low-alpha line, not blurred for speed)
    sw = max(4, int(w * 0.005))
    draw.line(coords, fill=(*glow_color[:3], 70), width=sw + s(20),
              joint="curve")
    draw.line(coords, fill=(0, 0, 0, 200), width=sw + s(8), joint="curve")
    draw.line(coords, fill=line_color, width=sw, joint="curve")

    # Start dot (origin) — sage
    sx, sy = coords[0]
    r = max(6, int(w * 0.012))
    # Label sits ~2.5× the dot radius away so it never crowds the dot or
    # its halo. Floored at s(20) so tiny dots still get a visible gap.
    label_gap = max(s(20), int(r * 2.5))
    # Label-side heuristic samples the polyline in a neighborhood sized
    # to the route block (≈12% of width) so it works on both layouts.
    side_radius = max(s(80), int(w * 0.12))
    draw.ellipse([sx - r - 3, sy - r - 3, sx + r + 3, sy + r + 3],
                 fill=line_color)
    draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=glow_color)
    canvas_w = draw.im.size[0]
    edge_pad = s(40)
    start_side = _pin_label_side(coords, 0, side_radius)
    start_side = _clamp_pin_side(start_side, origin, sx, font=font_pin,
                                 gap=label_gap, canvas_w=canvas_w,
                                 edge_pad=edge_pad)
    _draw_pin_label(draw, origin, sx, sy,
                    font=font_pin, fill=line_color, gap=label_gap,
                    side=start_side)

    # Turn dot (destination) — mint, pinned to the polyline vertex
    # nearest the supplied end_pin (point-to-point and out-and-back with
    # a known terminus), or farthest from start when no end_pin is given.
    if end_pin is not None:
        ep_lat, ep_lon = end_pin
        end_idx = min(
            range(len(route)),
            key=lambda i: (route[i][0] - ep_lat) ** 2 + (route[i][1] - ep_lon) ** 2,
        )
    else:
        end_idx = max(
            range(len(coords)),
            key=lambda i: (coords[i][0] - sx) ** 2 + (coords[i][1] - sy) ** 2,
        )
    nx, ny = coords[end_idx]
    draw.ellipse([nx - r - 3, ny - r - 3, nx + r + 3, ny + r + 3],
                 fill=line_color)
    draw.ellipse([nx - r, ny - r, nx + r, ny + r], fill=end_dot_color)
    if destination:
        end_text = destination.upper()
        end_side = _pin_label_side(coords, end_idx, side_radius)
        end_side = _clamp_pin_side(end_side, end_text, nx, font=font_pin,
                                   gap=label_gap, canvas_w=canvas_w,
                                   edge_pad=edge_pad)
        _draw_pin_label(draw, end_text, nx, ny,
                        font=font_pin, fill=line_color, gap=label_gap,
                        side=end_side)


def _render_card(
    output_path: Path,
    duration_secs: float,
    params: _SegmentParams,
    frame_fn: Callable[[float], Image.Image],
    fade_in: float = 0.4,
    fade_out: float = 0.4,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = int(round(duration_secs * params.fps))
    layout = "stereo" if params.channels >= 2 else "mono"

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{params.width}x{params.height}",
        "-r", params.fps_frac,
        "-i", "pipe:0",
        "-f", "lavfi", "-t", f"{duration_secs:.3f}",
        "-i", f"anullsrc=channel_layout={layout}:sample_rate={params.sample_rate}",
        "-c:v", "libx264", "-crf", "18", "-preset", "slow",
        "-pix_fmt", params.pix_fmt,
        "-r", params.fps_frac,
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        str(output_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in range(total):
        t = f / params.fps
        if t < fade_in:
            alpha = t / fade_in
        elif t > duration_secs - fade_out:
            alpha = max(0.0, (duration_secs - t) / fade_out)
        else:
            alpha = 1.0
        img = frame_fn(alpha)
        proc.stdin.write(img.tobytes())

    proc.stdin.close()
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {output_path.name}:\n"
            f"{stderr.decode()[-600:]}"
        )
    return output_path


def render_intro(
    output_path: Path,
    stats: RideStats,
    params: _SegmentParams,
    duration_secs: float = 2.5,
    fade_in: float = 0.4,
    fade_out: float = 0.4,
) -> Path:
    return _render_card(
        output_path, duration_secs, params,
        lambda alpha: _draw_intro_frame(params.width, params.height, stats, alpha),
        fade_in=fade_in, fade_out=fade_out,
    )


def render_outro(
    output_path: Path,
    stats: RideStats,
    params: _SegmentParams,
    duration_secs: float = 3.0,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    destination: str = "",
    subtitle: str = _DEFAULT_OUTRO_SUBTITLE,
    origin: str = _DEFAULT_ORIGIN,
    road: str = LOCKUP_ROAD,
    crew: str = LOCKUP_CREW,
    end_pin: tuple[float, float] | None = None,
) -> Path:
    """Render the recap card as a static clip.

    Defaults to no internal fade — when used with ``crossfade_outro``,
    the xfade filter handles the appearance.

    Per-ride callers may pass:
      destination — e.g. "PIERMONT" → title becomes "{origin} / → PIERMONT"
      subtitle    — italic line under the title; omitted when empty
      origin      — title + start-pin label; normally GPS-derived
      road        — left side of the footer lockup; normally GPS-derived
      crew        — right side of the footer lockup

    Anything left unset falls back to the ``lockups`` design tokens.
      end_pin     — (lat, lon) for the destination pin on the route trace.
                    Overrides any value already on ``stats``. None leaves
                    the route-block default ("farthest from start") in
                    place — appropriate for out-and-back rides.
    """
    if end_pin is not None:
        stats.end_pin = end_pin
    return _render_card(
        output_path, duration_secs, params,
        lambda alpha: _draw_outro_frame(
            params.width, params.height, stats, alpha,
            destination=destination, subtitle=subtitle, origin=origin,
            road=road, crew=crew,
        ),
        fade_in=fade_in, fade_out=fade_out,
    )


def _probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def crossfade_outro(
    last_segment_path: Path,
    outro_card_path: Path,
    output_path: Path,
    params: _SegmentParams,
    crossfade_secs: float,
) -> Path:
    """Crossfade the last burned segment into the recap card.

    Returns a re-encoded tail clip whose final ``crossfade_secs`` blend
    the last segment to black while the recap fades in over the same
    window. Concat-compatible with the other burned segments because
    the encode params match.
    """
    last_dur = _probe_duration(last_segment_path)
    card_dur = _probe_duration(outro_card_path)
    xd = max(0.1, min(crossfade_secs, last_dur, card_dur))
    offset = max(0.0, last_dur - xd)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(last_segment_path),
        "-i", str(outro_card_path),
        "-filter_complex",
        (f"[0:v][1:v]xfade=transition=fade:duration={xd:.3f}:offset={offset:.3f}[v];"
         f"[0:a][1:a]acrossfade=d={xd:.3f}[a]"),
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-crf", "18", "-preset", "slow",
        "-pix_fmt", params.pix_fmt,
        "-r", params.fps_frac,
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"xfade failed for {output_path.name}:\n{proc.stderr[-600:]}"
        )
    return output_path
