"""Burn telemetry overlay onto GoPro video.

Five overlay elements, each drawn independently:
  1. HUD         — left-side metrics stack (speed, power, HR, cadence, gradient)
  2. Mini-map    — circular heading-up OSM map with position chevron (sage)
  3. Route trace — full-ride GPS polyline with position dot (mint)
  4. Elevation   — bottom-center sparkline with position dot (mint) + altitude
  5. Lockup band — sage hairline rule, origin → destination · road,
                   odometer right-aligned to elevation bounds

Design language sourced from `design/tokens.json` — sage = chrome (chevron,
hairlines, italic), mint = live position (route + elevation dots), cream
on tight black stroke for legibility. No drop shadow; race-bib letter
tracking on labels. Designed to survive YouTube H.264 recompression at
1080p.

Supports landscape (16:9) and portrait (9:16, center-crop) formats.
"""

from __future__ import annotations

import datetime as dt
import io
import math
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Pull design tokens from the project's single source of truth.
from .design.tokens import (
    SAGE, MINT, CREAM, ZONE_COLORS,
    FONT_NUMERIC, FONT_NUMERIC_REG, FONT_BODY,
    LABEL_TRACKING_RATIO, default_lockup,
)

from . import intro_styles
from .fit_parser import RideData, RidePoint, parse_fit
from .gopro_meta import GoProClip, extract_metadata
from .sync import auto_sync, get_ride_timezone

# ─── Unit conversions ──────────────────────────────────────────
_MS_TO_MPH = 2.23694
_M_TO_MILES = 1 / 1609.344
_M_TO_FT = 3.28084

# GoPro creation_time and FIT timestamps are UTC. The configured ride
# timezone is used for the time-of-day title card.

# ─── Training zone colors (progressive cold→hot) ─────────────
# Sourced from design/tokens.json. Thresholds (fraction of FTP / max HR)
# are policy and stay here; colors come from ZONE_COLORS.
_ZONE_ALPHA = 245
_POWER_THRESHOLDS = [0.55, 0.75, 0.90, 1.05, 1.20, 9.99]
_HR_THRESHOLDS = [0.60, 0.70, 0.80, 0.90, 9.99]

_POWER_ZONES: list[tuple[float, tuple[int, int, int, int]]] = [
    (t, (*ZONE_COLORS[i], _ZONE_ALPHA)) for i, t in enumerate(_POWER_THRESHOLDS)
]
_HR_ZONES: list[tuple[float, tuple[int, int, int, int]]] = [
    (t, (*ZONE_COLORS[i], _ZONE_ALPHA)) for i, t in enumerate(_HR_THRESHOLDS)
]


def _zone_color_at(frac: float, zones: list[tuple[float, tuple[int, int, int, int]]]) -> tuple[int, int, int, int]:
    """Return the zone color for a given fraction of reference (FTP or max HR)."""
    for upper, color in zones:
        if frac <= upper:
            return color
    return zones[-1][1]


# ─── Map tile config ───────────────────────────────────────────
_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
_TILE_SIZE = 256
_TILE_ZOOM = 15
_TILE_BG = (240, 238, 233, 255)  # OSM default background color


def _s(val: float, scale: float) -> int:
    """Scale a 4K-base dimension to actual resolution."""
    return max(1, int(val * scale))


# ═══════════════════════════════════════════════════════════════
# Shared design tokens — route trace + elevation profile use these
# ═══════════════════════════════════════════════════════════════

class _TraceStyle:
    """Shared line/dot styling for route trace and elevation profile."""

    @staticmethod
    def stroke_width(frame_w: int) -> int:
        return max(4, int(frame_w * 0.002))

    @staticmethod
    def draw_line(draw: ImageDraw.Draw, pts, stroke_w: int, sc: float,
                  outer: int = 14, inner: int = 6):
        draw.line(pts, fill=(0, 0, 0, 180), width=stroke_w + _s(outer, sc),
                  joint="curve")
        draw.line(pts, fill=(0, 0, 0, 90), width=stroke_w + _s(inner, sc),
                  joint="curve")
        draw.line(pts, fill=(255, 255, 255, 240), width=stroke_w,
                  joint="curve")


# ═══════════════════════════════════════════════════════════════
# Layout geometry — all layout-varying dimensions in one place
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LayoutGeometry:
    """Layout-specific dimensions. Ints are 4K-base pixels (scaled via _s()).
    Floats ending in _frac are fractions of frame width or height."""

    # Fonts (4K-base sizes)
    font_value: int
    font_label: int
    font_dist: int
    font_dist_sm: int

    # Mini-map circle
    map_diameter_frac: float      # fraction of w
    map_margin_r_frac: float      # fraction of w
    map_margin_t_frac: float      # fraction of h

    # HUD metrics stack
    hud_margin: int               # 4K-base
    hud_gap: int
    hud_bar_gap: int
    hud_label_gap: int
    hud_bar_h: int
    hud_bar_w: int
    hud_y_top_frac: float         # top of HUD zone (fraction of h)
    hud_y_span_frac: float        # height of HUD zone (fraction of h)

    # Route trace
    route_w_frac: float           # fraction of w
    route_margin_r_frac: float    # fraction of w
    route_bottom_frac: float      # bottom extent (fraction of h)

    # Elevation profile
    elev_w_frac: float            # fraction of w
    elev_h_frac: float            # fraction of h
    elev_bottom_frac: float       # margin from bottom (fraction of h)

    # Trace line shadow sizes (4K-base)
    trace_outer: int              # outer black halo addition
    trace_inner: int              # inner black halo addition
    dot_halo: int                 # dot outer halo radius addition
    dot_glow: int                 # dot inner glow radius addition


_LANDSCAPE = LayoutGeometry(
    font_value=240, font_label=48, font_dist=96, font_dist_sm=80,
    map_diameter_frac=0.12, map_margin_r_frac=0.03, map_margin_t_frac=0.03,
    hud_margin=60, hud_gap=30, hud_bar_gap=10,
    hud_label_gap=4, hud_bar_h=16, hud_bar_w=320,
    hud_y_top_frac=0.0, hud_y_span_frac=1.0,
    route_w_frac=0.12, route_margin_r_frac=0.02, route_bottom_frac=0.80,
    elev_w_frac=0.60, elev_h_frac=0.056, elev_bottom_frac=0.02,
    trace_outer=14, trace_inner=6, dot_halo=10, dot_glow=5,
)

_PORTRAIT = LayoutGeometry(
    font_value=130, font_label=28, font_dist=72, font_dist_sm=52,
    map_diameter_frac=0.24, map_margin_r_frac=0.03, map_margin_t_frac=0.10,
    hud_margin=36, hud_gap=12, hud_bar_gap=6,
    hud_label_gap=2, hud_bar_h=10, hud_bar_w=190,
    hud_y_top_frac=0.10, hud_y_span_frac=0.55,
    route_w_frac=0.25, route_margin_r_frac=0.03, route_bottom_frac=0.82,
    elev_w_frac=0.88, elev_h_frac=0.07, elev_bottom_frac=0.10,
    trace_outer=14, trace_inner=6, dot_halo=10, dot_glow=5,
)

_LAYOUTS: dict[str, LayoutGeometry] = {
    "landscape": _LANDSCAPE,
    "portrait": _PORTRAIT,
}


# ═══════════════════════════════════════════════════════════════
# Web Mercator tile math
# ═══════════════════════════════════════════════════════════════

def _lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad))
             / math.pi) / 2.0 * n)
    return x, y


def _lat_lon_to_pixel(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    n = 2 ** zoom
    px = (lon + 180.0) / 360.0 * n * _TILE_SIZE
    lat_rad = math.radians(lat)
    py = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad))
          / math.pi) / 2.0 * n * _TILE_SIZE
    return px, py


_TILE_CACHE_DIR = Path.home() / ".cache" / "gopro-garmin" / "tiles"


class _TileCache:
    """Fetches and caches OSM map tiles (memory + disk)."""

    def __init__(self):
        self._cache: dict[tuple[int, int, int], Image.Image] = {}
        self._failed: set[tuple[int, int, int]] = set()
        _TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _disk_path(self, x: int, y: int, z: int) -> Path:
        return _TILE_CACHE_DIR / f"{z}" / f"{x}" / f"{y}.png"

    def get_tile(self, x: int, y: int, z: int) -> Image.Image | None:
        key = (x, y, z)
        if key in self._cache:
            return self._cache[key]
        if key in self._failed:
            return None

        # Check disk cache
        dp = self._disk_path(x, y, z)
        if dp.exists():
            try:
                tile = Image.open(dp).convert("RGBA")
                self._cache[key] = tile
                return tile
            except Exception:
                pass

        # Fetch from network
        url = _TILE_URL.format(z=z, x=x, y=y)
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "GoPro-Garmin-Pipeline/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = resp.read()
            tile = Image.open(io.BytesIO(data)).convert("RGBA")
            self._cache[key] = tile
            # Persist to disk
            dp.parent.mkdir(parents=True, exist_ok=True)
            dp.write_bytes(data)
            return tile
        except Exception:
            self._failed.add(key)
            return None

    def get_map_region(self, lat: float, lon: float, zoom: int,
                       output_size: int) -> Image.Image | None:
        """Square map image centered on lat/lon, precisely aligned."""
        cpx, cpy = _lat_lon_to_pixel(lat, lon, zoom)
        tx_center, ty_center = _lat_lon_to_tile(lat, lon, zoom)
        grid = 5
        stx, sty = tx_center - grid // 2, ty_center - grid // 2
        sw, sh = grid * _TILE_SIZE, grid * _TILE_SIZE
        stitched = Image.new("RGBA", (sw, sh), _TILE_BG)

        ok = False
        for dy in range(grid):
            for dx in range(grid):
                tile = self.get_tile(stx + dx, sty + dy, zoom)
                if tile:
                    if tile.size != (_TILE_SIZE, _TILE_SIZE):
                        tile = tile.resize((_TILE_SIZE, _TILE_SIZE), Image.LANCZOS)
                    stitched.paste(tile, (dx * _TILE_SIZE, dy * _TILE_SIZE))
                    ok = True
        if not ok:
            return None

        ox, oy = stx * _TILE_SIZE, sty * _TILE_SIZE
        lx, ly = cpx - ox, cpy - oy
        half = output_size / 2.0
        box = [lx - half, ly - half, lx + half, ly + half]

        # Pad if crop exceeds stitched bounds
        pl, pt = max(0, -box[0]), max(0, -box[1])
        pr, pb = max(0, box[2] - sw), max(0, box[3] - sh)
        if pl > 0 or pt > 0 or pr > 0 or pb > 0:
            exp = Image.new("RGBA", (sw + int(pl + pr + 1), sh + int(pt + pb + 1)), _TILE_BG)
            exp.paste(stitched, (int(pl), int(pt)))
            stitched = exp
            box = [box[0] + pl, box[1] + pt, box[2] + pl, box[3] + pt]

        cropped = stitched.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
        if cropped.size != (output_size, output_size):
            cropped = cropped.resize((output_size, output_size), Image.LANCZOS)
        return cropped


# ═══════════════════════════════════════════════════════════════
# Overlay Renderer
# ═══════════════════════════════════════════════════════════════

class OverlayRenderer:
    def __init__(
        self,
        synced,
        ride: RideData,
        layout: str = "landscape",
        lockup: str | None = None,
        intro_secs: float = 0.0,
    ):
        self.synced = synced
        self.clip = synced.clip
        self.ride = ride
        # Intro title card: for the first `intro_secs` seconds the footage
        # blurs → clears (ffmpeg side) while a date + time-of-day lockup
        # fades over it and the HUD fades in. 0 disables.
        self.intro_secs = intro_secs
        self.w = self.clip.width
        self.h = self.clip.height
        self.sc = self.h / 2160  # 1.0 at 4K
        self.geo = _LAYOUTS[layout]
        self._lockup = lockup or default_lockup()
        g = self.geo

        # ── Fonts (families shared, sizes from geometry) ──────
        # Numerics come from tokens (Barlow Bold). Labels + lockup come
        # from tokens (Inter).
        self.f_value = ImageFont.truetype(FONT_NUMERIC, _s(g.font_value, self.sc))
        self.f_label = ImageFont.truetype(FONT_BODY, _s(g.font_label, self.sc))
        self.f_dist = ImageFont.truetype(FONT_NUMERIC, _s(g.font_dist, self.sc))
        self.f_dist_sm = ImageFont.truetype(FONT_NUMERIC_REG, _s(g.font_dist_sm, self.sc))
        # Lockup band font — Inter at a layout-appropriate size.
        lock_size = 38 if layout == "portrait" else 48
        self.f_lock = ImageFont.truetype(FONT_BODY, _s(lock_size, self.sc))

        # ── Pre-compute ride data ──────────────────────────────
        self.route_pts = [(p.lat, p.lon) for p in ride.points if p.lat and p.lon]

        dists = [p.distance for p in ride.points if p.distance is not None]
        self.total_dist_mi = max(dists) * _M_TO_MILES if dists else 0

        raw_elev = [
            (p.distance * _M_TO_MILES, p.altitude * _M_TO_FT)
            for p in ride.points
            if p.distance is not None and p.altitude is not None
        ]
        # Drop only LEADING points if the barometer hadn't settled yet.
        # Previous version filtered the entire ride against the early median,
        # which silently dropped huge stretches on hilly rides (anything
        # >200ft from the early baseline) and produced flat gaps in the
        # rendered profile.
        if len(raw_elev) > 10:
            early_median = sorted(a for _, a in raw_elev[1:11])[5]
            while raw_elev and abs(raw_elev[0][1] - early_median) > 200:
                raw_elev.pop(0)
        self.elev_profile = raw_elev

        self.gradients: dict[int, float] = {}
        for i in range(1, len(ride.points)):
            p0, p1 = ride.points[i - 1], ride.points[i]
            if (p0.altitude is not None and p1.altitude is not None
                    and p0.distance is not None and p1.distance is not None):
                dd = p1.distance - p0.distance
                if dd > 0.5:
                    self.gradients[i] = ((p1.altitude - p0.altitude) / dd) * 100

        speeds = [p.speed * _MS_TO_MPH for p in ride.points if p.speed]
        powers = [p.power for p in ride.points if p.power]
        hrs = [p.heart_rate for p in ride.points if p.heart_rate]
        cadences = [p.cadence for p in ride.points if p.cadence]
        self.max_speed = max(speeds) if speeds else 40
        self.max_power = max(powers) if powers else 500
        self.max_hr = max(hrs) if hrs else 200
        self.max_cadence = max(cadences) if cadences else 120

        self.has_power = ride.has_power
        self.has_cadence = ride.has_cadence

        from .config import get_settings
        _cfg = get_settings()
        self.ftp = _cfg.ftp
        self.max_heart_rate = _cfg.max_heart_rate

        if len(self.route_pts) >= 2:
            lats = [p[0] for p in self.route_pts]
            lons = [p[1] for p in self.route_pts]
            self._min_lat, self._max_lat = min(lats), max(lats)
            self._min_lon, self._max_lon = min(lons), max(lons)
            self._mid_lat = (self._min_lat + self._max_lat) / 2
            self._cos_lat = math.cos(math.radians(self._mid_lat))

        # ── Layout: mini-map circle ────────────────────────────
        self._circle_diameter = int(self.w * g.map_diameter_frac)
        self._circle_radius = self._circle_diameter // 2
        margin_r = int(self.w * g.map_margin_r_frac)
        margin_t = int(self.h * g.map_margin_t_frac)
        self._circle_cx = self.w - margin_r - self._circle_radius
        self._circle_cy = margin_t + self._circle_radius
        self._circle_bottom = margin_t + self._circle_diameter

        # ── Map tile cache ─────────────────────────────────────
        self._tile_cache = _TileCache()
        self._prefetch_tiles()

        # ── Build static route trace image ─────────────────────
        self._full_route_img = self._build_route_trace()

        # ── Intro title-card fonts ─────────────────────────────
        # Fonts are always built so a renderer cached across segments can be
        # armed for the opener on just the first clip via set_intro(). Strings
        # + origin resolve lazily on the first rendered frame so the intro
        # window aligns with the clip start regardless of any start_offset
        # seek (ffmpeg resets the blend timestamp T to 0 there).
        self._intro_origin: float | None = None
        self._intro_time_str = ""
        self._intro_date_str = ""
        self.f_intro_time = ImageFont.truetype(FONT_NUMERIC, _s(240, self.sc))
        self.f_intro_date = ImageFont.truetype(FONT_BODY, _s(56, self.sc))

    def set_intro(self, intro_secs: float) -> None:
        """Arm (or disable) the opening blur→title card for the next burn.

        Resets the per-clip origin/strings so a renderer reused across
        several segments only renders the opener on the clip it's armed for.
        The matching ffmpeg blur ramp is applied by burn_overlay using the
        same intro_secs value.
        """
        self.intro_secs = intro_secs
        self._intro_origin = None
        self._intro_time_str = ""
        self._intro_date_str = ""

    # ─── Helpers ───────────────────────────────────────────────

    def _local_wall(self, video_secs: float) -> dt.datetime:
        """Wall-clock capture time at *video_secs* into the clip, in the
        ride's local timezone. GoPro creation_time is UTC."""
        wall = self.clip.creation_time + dt.timedelta(
            seconds=video_secs + self.synced.offset_secs
        )
        if wall.tzinfo is None:
            wall = wall.replace(tzinfo=dt.timezone.utc)
        return wall.astimezone(self.synced.ride_timezone or get_ride_timezone())

    def _prefetch_tiles(self):
        if len(self.route_pts) < 2:
            return
        tiles: set[tuple[int, int]] = set()
        for p in self.ride.points:
            if p.lat and p.lon:
                tx, ty = _lat_lon_to_tile(p.lat, p.lon, _TILE_ZOOM)
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        tiles.add((tx + dx, ty + dy))
        print(f"  Pre-fetching {len(tiles)} map tiles...")
        n = sum(1 for tx, ty in tiles if self._tile_cache.get_tile(tx, ty, _TILE_ZOOM))
        print(f"  Fetched {n}/{len(tiles)} tiles")

    def _get_point(self, video_secs: float) -> RidePoint | None:
        return self.ride.point_at(self.synced._adjust(video_secs))

    def _get_gradient(self, point: RidePoint) -> float:
        idx = self.ride.point_index(point)
        return self.gradients.get(idx, 0) if idx >= 0 else 0

    def _get_point_index(self, point: RidePoint) -> int:
        return self.ride.point_index(point)

    def _calc_heading(self, point: RidePoint) -> float:
        """Averaged heading from last 10 GPS pairs. Degrees CW from north."""
        idx = self._get_point_index(point)
        if idx < 1:
            return 0
        pts = [(p.lat, p.lon) for p in self.ride.points[max(0, idx - 10):idx + 1]
               if p.lat and p.lon]
        if len(pts) < 2:
            return 0
        dlat = sum(pts[i][0] - pts[i - 1][0] for i in range(1, len(pts)))
        dlon = sum((pts[i][1] - pts[i - 1][1])
                   * math.cos(math.radians((pts[i][0] + pts[i - 1][0]) / 2))
                   for i in range(1, len(pts)))
        if abs(dlat) < 1e-9 and abs(dlon) < 1e-9:
            return 0
        return math.degrees(math.atan2(dlon, dlat))

    def _text(self, draw, xy, text, font, alpha=240, anchor="lt"):
        """Cream text with a tight black stroke. No offset shadow.

        The previous version rendered both a shadow copy AND a stroked
        main copy — that read as a 3D drop-shadow on a HUD. The cleaner,
        on-brand treatment is just a stroke for legibility on bright
        backgrounds.
        """
        stroke = _s(4, self.sc)
        draw.text(
            xy, text,
            fill=(*CREAM, alpha),
            font=font, anchor=anchor,
            stroke_width=stroke,
            stroke_fill=(0, 0, 0, int(alpha * 0.85)),
        )

    def _draw_label_tracked(self, draw, xy, text, font, alpha=210):
        """Letter-spaced label — race-bib feel for MPH/W/BPM/RPM.

        Tracking defaults to 12% of font size (per design tokens).
        """
        tracking_px = max(1, int(font.size * LABEL_TRACKING_RATIO))
        x, y = xy
        for ch in text:
            self._text(draw, (x, y), ch, font, alpha=alpha, anchor="lt")
            b = font.getbbox(ch)
            x += (b[2] - b[0]) + tracking_px

    # ═══════════════════════════════════════════════════════════
    # 1. HUD — left-side metrics stack
    # ═══════════════════════════════════════════════════════════

    def _draw_hud(self, draw, point, speed_mph, gradient):
        s = self.sc
        g = self.geo
        margin = _s(g.hud_margin, s)
        gap = _s(g.hud_gap, s)
        bar_gap = _s(g.hud_bar_gap, s)
        label_gap = _s(g.hud_label_gap, s)
        bar_h = _s(g.hud_bar_h, s)
        bar_w = _s(g.hud_bar_w, s)

        # Metric rows — no icons. Power and cadence render as `--` when
        # the sensor reads 0 mid-coast (otherwise reads as missing data
        # even though the rider is just freewheeling).
        metrics: list[tuple[str, str, float, float,
                            list | None, float]] = [
            (f"{speed_mph:.0f}" if speed_mph else "--", "MPH",
             speed_mph or 0, self.max_speed, None, 0),
        ]
        if self.has_power:
            pw = point.power
            metrics.append(
                (str(pw) if pw else "--", "W",
                 pw or 0, self.ftp * 2, _POWER_ZONES, self.ftp),
            )
        metrics.append(
            (str(point.heart_rate) if point.heart_rate else "--", "BPM",
             point.heart_rate or 0, self.max_hr, _HR_ZONES, self.max_heart_rate),
        )
        if self.has_cadence:
            cd = point.cadence
            metrics.append(
                (str(cd) if cd else "--", "RPM",
                 cd or 0, self.max_cadence, None, 0),
            )
        metrics.append(
            (f"{gradient:+.0f}%", "GRADE",
             abs(gradient), 15, None, 0),
        )

        # Tighter visible-glyph height pulls the bar up under the digits
        # (was floating on the font's internal leading).
        val_box_h = draw.textbbox((0, 0), "000", font=self.f_value)[3]
        val_h = int(val_box_h * 0.74)
        lab_h = draw.textbbox((0, 0), "MPH", font=self.f_label)[3]
        unit_h = val_h + bar_gap + bar_h + label_gap + lab_h
        total = len(metrics) * unit_h + (len(metrics) - 1) * gap

        y0 = int(self.h * g.hud_y_top_frac)
        available = int(self.h * g.hud_y_span_frac)
        y0 = y0 + (available - total) // 2

        for val_str, label, raw, mx, zones, zone_ref in metrics:
            self._text(draw, (margin, y0), val_str, self.f_value,
                       alpha=250, anchor="lt")
            self._draw_progress_bar(
                draw, margin, y0 + val_h + bar_gap, raw, mx,
                bar_w, bar_h, zones=zones, zone_ref=zone_ref,
            )
            if label:
                self._draw_label_tracked(
                    draw, (margin, y0 + val_h + bar_gap + bar_h + label_gap),
                    label, self.f_label, alpha=210,
                )
            y0 += unit_h + gap

    def _draw_progress_bar(self, draw, x, y, value, max_val, bw, bh,
                           zones=None, zone_ref=0):
        s = self.sc
        sw, sg = _s(16, s), _s(4, s)
        frac = min(value / max_val, 1.0) if max_val > 0 and value else 0
        n = bw // (sw + sg)
        filled = int(n * frac)
        draw.rectangle([x - 2, y - 2, x + bw + 2, y + bh + 2], fill=(0, 0, 0, 51))
        for i in range(n):
            sx = x + i * (sw + sg)
            draw.rectangle([sx - 1, y - 1, sx + sw + 1, y + bh + 1], fill=(0, 0, 0, 180))
            if i < filled and zones and zone_ref > 0:
                seg_val = (i + 1) / n * max_val
                color = _zone_color_at(seg_val / zone_ref, zones)
            elif i < filled:
                color = (255, 255, 255, 245)
            else:
                color = (255, 255, 255, 60)
            draw.rectangle([sx, y, sx + sw, y + bh], fill=color)

    # ═══════════════════════════════════════════════════════════
    # 2. Mini-map — heading-up OSM tiles with chevron
    # ═══════════════════════════════════════════════════════════

    def _draw_minimap(self, draw, overlay, lat, lon, point):
        if len(self.route_pts) < 2:
            return
        s = self.sc
        d = self._circle_diameter
        r = self._circle_radius
        cx, cy = self._circle_cx, self._circle_cy
        heading = self._calc_heading(point)

        # Fetch oversized map for rotation headroom
        fetch = int(d * 1.5)
        mimg = self._tile_cache.get_map_region(lat, lon, _TILE_ZOOM, fetch)
        if mimg is None:
            mimg = Image.new("RGBA", (fetch, fetch), _TILE_BG)
        if mimg.size != (fetch, fetch):
            mimg = mimg.resize((fetch, fetch), Image.LANCZOS)
        mimg = mimg.convert("RGB").convert("RGBA")

        # Draw GPS route on north-up map
        cpx, cpy = _lat_lon_to_pixel(lat, lon, _TILE_ZOOM)
        mdraw = ImageDraw.Draw(mimg)
        half = fetch / 2.0
        pts = []
        for rlat, rlon in self.route_pts:
            rpx, rpy = _lat_lon_to_pixel(rlat, rlon, _TILE_ZOOM)
            lx, ly = int(rpx - cpx + half), int(rpy - cpy + half)
            if -fetch < lx < fetch * 2 and -fetch < ly < fetch * 2:
                pts.append((lx, ly))
        sw = max(4, int(self.w * 0.003))
        if len(pts) >= 2:
            mdraw.line(pts, fill=(0, 0, 0, 60), width=sw + _s(8, s), joint="curve")
            mdraw.line(pts, fill=(*SAGE, 220), width=sw, joint="curve")

        # Rotate heading-up
        rotated = mimg.rotate(heading, center=(fetch // 2, fetch // 2),
                              resample=Image.BICUBIC, expand=False,
                              fillcolor=_TILE_BG)
        co = (fetch - d) // 2
        cropped = rotated.crop((co, co, co + d, co + d))

        # Chevron at center — points UP (= direction of travel)
        cdraw = ImageDraw.Draw(cropped)
        ch = max(6, int(self.w * 0.012))
        cw = int(ch * 0.7)
        dc = d // 2
        chev = [(dc, dc - ch // 2), (dc - cw // 2, dc + ch // 2),
                (dc, dc + ch // 4), (dc + cw // 2, dc + ch // 2)]
        cdraw.ellipse([dc - ch, dc - ch, dc + ch, dc + ch], fill=(255, 255, 255, 50))
        otr = max(2, _s(3, s))
        for dx in range(-otr, otr + 1):
            for dy in range(-otr, otr + 1):
                if dx * dx + dy * dy <= otr * otr:
                    cdraw.polygon([(x + dx, y + dy) for x, y in chev], fill=(0, 0, 0, 220))
        cdraw.polygon(chev, fill=(*SAGE, 255))

        # Vignette mask — 92% inner, fade to edge
        mask = np.zeros((d, d), dtype=np.float32)
        yc, xc = np.ogrid[:d, :d]
        dist = np.sqrt((xc - d / 2.0) ** 2 + (yc - d / 2.0) ** 2)
        mask = np.clip((r - dist) / (r - r * 0.92), 0, 1) * 255
        cropped.putalpha(Image.fromarray(mask.astype(np.uint8)))

        px, py = cx - r, cy - r
        overlay.paste(cropped, (px, py), cropped)

        # Sage hairline border — matches the lockup band's visual weight
        bw = max(1, int(self.sc))
        draw.ellipse([px, py, px + d, py + d], outline=(*SAGE, 180), width=bw)

    # ═══════════════════════════════════════════════════════════
    # 3. Route trace — full-ride polyline (static) + position dot
    # ═══════════════════════════════════════════════════════════

    def _build_route_trace(self) -> Image.Image | None:
        if len(self.route_pts) < 2:
            return None
        s = self.sc
        g = self.geo
        img = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        gap = int(self.h * 0.02)
        rw = int(self.w * g.route_w_frac)
        rx0 = self.w - rw - int(self.w * g.route_margin_r_frac)
        ry0 = self._circle_bottom + gap
        ry1 = int(self.h * g.route_bottom_frac)
        rh = ry1 - ry0
        if rh < 50:
            return None
        pad = _s(12, s)

        # Project route into available box, centered
        lat_r = self._max_lat - self._min_lat or 1e-6
        lon_r = self._max_lon - self._min_lon or 1e-6
        aspect = (lon_r * self._cos_lat) / lat_r
        dw, dh = rw - 2 * pad, rh - 2 * pad
        ba = dw / dh
        if aspect > ba:
            xs, ys = dw, dw / aspect
            xo, yo = pad, pad + (dh - ys) / 2
        else:
            ys, xs = dh, dh * aspect
            xo, yo = pad + (dw - xs) / 2, pad

        self._route_proj = lambda lat, lon: (
            int(rx0 + xo + ((lon - self._min_lon) / lon_r) * xs),
            int(ry0 + yo + ((self._max_lat - lat) / lat_r) * ys),
        )

        coords = [self._route_proj(lat, lon) for lat, lon in self.route_pts]

        sw = _TraceStyle.stroke_width(self.w)
        _TraceStyle.draw_line(draw, coords, sw, s,
                              outer=g.trace_outer, inner=g.trace_inner)

        self._route_x_center = rx0 + rw // 2
        self._route_y_bottom = max(c[1] for c in coords) + _s(16, s) if coords else ry1
        return img

    def _draw_mint_dot(self, draw, x, y) -> int:
        """Live-position dot: black halo, mint glow, solid mint center.
        One rule across surfaces — mint = live position. Returns dot_r
        so callers can offset labels off the dot."""
        s = self.sc
        dot_r = max(3, int(self.w * 0.004))
        hr = dot_r + _s(self.geo.dot_halo, s)
        draw.ellipse([x - hr, y - hr, x + hr, y + hr], fill=(0, 0, 0, 140))
        gr = dot_r + _s(self.geo.dot_glow, s)
        draw.ellipse([x - gr, y - gr, x + gr, y + gr], fill=(*MINT, 110))
        draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r],
                     fill=(*MINT, 255))
        return dot_r

    def _draw_route_dot(self, draw, lat, lon):
        if not hasattr(self, '_route_proj'):
            return
        px, py = self._route_proj(lat, lon)
        self._draw_mint_dot(draw, px, py)

    # ═══════════════════════════════════════════════════════════
    # 4. Elevation profile — bottom-center sparkline
    # ═══════════════════════════════════════════════════════════

    def _draw_elevation(self, draw, dist_mi, alt_ft):
        s = self.sc
        g = self.geo
        ew = int(self.w * g.elev_w_frac)
        eh = int(self.h * g.elev_h_frac)
        ex = (self.w - ew) // 2
        # Lift the elevation profile by ~60px so the lockup band fits below.
        ey = self.h - int(self.h * g.elev_bottom_frac) - eh - _s(60, s)

        # Publish elevation x-bounds so _draw_lockup can align to them.
        self._elev_x_left = ex
        self._elev_x_right = ex + ew

        ds = [p[0] for p in self.elev_profile]
        als = [p[1] for p in self.elev_profile]
        mind, maxd = min(ds), max(ds)
        mina, maxa = min(als), max(als)
        rd, ra = maxd - mind or 1, maxa - mina or 1

        def exy(d, a):
            return int(ex + (d - mind) / rd * ew), int(ey + eh - (a - mina) / ra * eh)

        pts = [exy(d, a) for d, a in self.elev_profile]
        sw = _TraceStyle.stroke_width(self.w)
        _TraceStyle.draw_line(draw, pts, sw, s,
                              outer=g.trace_outer, inner=g.trace_inner)

        if dist_mi is not None and alt_ft is not None:
            mx, my = exy(dist_mi, alt_ft)
            dot_r = self._draw_mint_dot(draw, mx, my)
            self._text(draw, (mx + dot_r + _s(g.dot_halo + 6, s), my),
                       f"{int(alt_ft)} FT", self.f_dist_sm, alpha=220, anchor="lm")

    # ═══════════════════════════════════════════════════════════
    # 5. Lockup band — sage hairline rule + lockup left + odometer right
    # ═══════════════════════════════════════════════════════════

    def _draw_lockup(self, draw, dist_mi):
        """Bottom band: sage hairline rule, lockup at left, odometer at right.

        Both edges align to the elevation profile bounds (published by
        _draw_elevation). Falls back to a conservative margin if elevation
        didn't run this frame (e.g. ride without altitude).
        """
        s = self.sc
        x_left = getattr(self, "_elev_x_left", _s(60, s))
        x_right = getattr(self, "_elev_x_right", self.w - _s(60, s))

        rule_y = self.h - _s(72, s)
        # Sage hairline rule (1–2px), confined to elevation-profile bounds
        rule_h = max(1, _s(2, s))
        draw.rectangle(
            [x_left, rule_y, x_right, rule_y + rule_h],
            fill=(*SAGE, 180),
        )
        ly = rule_y + _s(14, s)

        # Lockup left
        self._text(draw, (x_left, ly), self._lockup, self.f_lock,
                   alpha=235, anchor="lt")

        # Odometer right — same Barlow Bold for both numbers, alpha alone
        # carries hierarchy. Two decimals.
        if dist_mi is not None:
            cur = f"{dist_mi:.2f}"
            tot = f"/ {self.total_dist_mi:.2f} MI"
            odo_font = ImageFont.truetype(
                FONT_NUMERIC, self.f_lock.size + _s(8, s),
            )
            cur_w = draw.textbbox((0, 0), cur, font=odo_font)[2]
            tot_w = draw.textbbox((0, 0), tot, font=odo_font)[2]
            x_tot = x_right - tot_w
            x_cur = x_tot - _s(12, s) - cur_w
            self._text(draw, (x_cur, ly - _s(8, s)), cur, odo_font,
                       alpha=240, anchor="lt")
            self._text(draw, (x_tot, ly - _s(8, s)), tot, odo_font,
                       alpha=170, anchor="lt")

    # ═══════════════════════════════════════════════════════════
    # Render frame — composites all five elements
    # ═══════════════════════════════════════════════════════════

    def _intro_alphas(self, video_secs: float) -> tuple[float, float] | None:
        """(title_alpha, hud_alpha) for the intro window, or None once past it.

        Title fades in fast, holds, fades out; the HUD stays hidden while the
        title is up and fades in as the footage sharpens over the back half.
        """
        if self.intro_secs <= 0:
            return None
        if self._intro_origin is None:
            self._intro_origin = video_secs
            local = self._local_wall(video_secs)
            # Windows does not support the POSIX %-I / %-M strftime flags,
            # so format the hour/minute explicitly to keep the timestamp
            # readable as "3:06 PM" instead of "03:06 PM".
            hour_12 = local.hour % 12 or 12
            self._intro_time_str = f"{hour_12}:{local.minute:02d} {local.strftime('%p')}"
            self._intro_date_str = local.strftime("%b %d, %Y").upper()
        rel = video_secs - self._intro_origin
        if rel >= self.intro_secs:
            return None
        p = rel / self.intro_secs

        def ramp(x, a, b):
            return max(0.0, min(1.0, (x - a) / (b - a)))

        # Title is on from the very first frame, holds, then fades out as the
        # footage sharpens and the HUD fades in over the back third.
        title = 1.0 - ramp(p, 0.60, 0.85)
        hud = ramp(p, 0.62, 1.0)
        return title, hud

    def _draw_intro_card(self, draw: ImageDraw.ImageDraw, alpha: float) -> None:
        """Centered time-of-day + date lockup for the opening blur.

        Matches the endcap recap-card vocabulary: a muted, letter-spaced
        Inter date eyebrow, a sage hairline rule, and the time-of-day as a
        cream Barlow Condensed hero numeric. Black stroke throughout so it
        stays legible over the live footage (the endcap sits on a dark card).
        """
        a = int(255 * alpha)
        if a <= 0:
            return
        cx, cy = self.w // 2, self.h // 2

        # Date eyebrow — Inter, cream (theme white), generously letter-spaced.
        spaced = "  ".join(self._intro_date_str)
        self._text(draw, (cx, cy - _s(215, self.sc)), spaced,
                   self.f_intro_date, alpha=a, anchor="mm")

        # Sage hairline rule — the endcap's signature divider.
        rule_w = _s(340, self.sc)
        rule_h = max(1, _s(3, self.sc))
        ry = cy - _s(150, self.sc)
        draw.rectangle([cx - rule_w // 2, ry, cx + rule_w // 2, ry + rule_h],
                       fill=(*SAGE, a))

        # Time-of-day — cream Barlow Condensed hero.
        self._text(draw, (cx, cy), self._intro_time_str,
                   self.f_intro_time, alpha=a, anchor="mm")

    def render_frame(self, video_secs: float) -> Image.Image:
        overlay = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        intro = self._intro_alphas(video_secs)

        point = self._get_point(video_secs)
        if point is None:
            # No telemetry yet — still show the intro title card if active.
            if intro is not None:
                self._draw_intro_card(draw, intro[0])
            return overlay

        gradient = self._get_gradient(point)
        dist_mi = point.distance * _M_TO_MILES if point.distance else None
        alt_ft = point.altitude * _M_TO_FT if point.altitude else None
        speed_mph = point.speed * _MS_TO_MPH if point.speed else None

        # 1. HUD
        self._draw_hud(draw, point, speed_mph, gradient)

        # 2. Mini-map
        if point.lat and point.lon:
            self._draw_minimap(draw, overlay, point.lat, point.lon, point)

        # 3. Route trace (static image + dynamic dot)
        if self._full_route_img:
            overlay.paste(self._full_route_img, (0, 0), self._full_route_img)
        if point.lat and point.lon:
            self._draw_route_dot(draw, point.lat, point.lon)

        # 4. Elevation profile (publishes _elev_x_left / _elev_x_right)
        if len(self.elev_profile) > 1:
            self._draw_elevation(draw, dist_mi, alt_ft)

        # 5. Lockup band — sage hairline + lockup left + odometer right,
        # both edges aligned to the elevation profile bounds.
        self._draw_lockup(draw, dist_mi)

        # Intro: dim the whole HUD as it fades in, then lay the title card
        # (date + time-of-day) over the top at full title alpha.
        if intro is not None:
            title_alpha, hud_alpha = intro
            if hud_alpha < 1.0:
                dimmed = overlay.getchannel("A").point(
                    lambda v: int(v * hud_alpha))
                overlay.putalpha(dimmed)
            self._draw_intro_card(draw, title_alpha)

        return overlay


# ═══════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# FFmpeg encode presets
# ═══════════════════════════════════════════════════════════════

ENCODE_PREVIEW = "preview"
ENCODE_MASTER = "master"


def _encode_args(preset: str) -> list[str]:
    """Return ffmpeg encoding arguments for the given preset.

    preview: a fast default that stays compatible with the current OS. On macOS
    the native VideoToolbox encoder is available, but Windows/Linux must use a
    cross-platform encoder such as libx264 or ffmpeg rejects the command.
    master:  libx264 software encoder, high quality, +faststart for web.
    """
    if preset == ENCODE_MASTER:
        return [
            "-c:v", "libx264", "-crf", "18", "-preset", "slow",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
    if sys.platform.startswith("darwin"):
        return [
            "-c:v", "h264_videotoolbox", "-q:v", "65",
            "-c:a", "copy", "-pix_fmt", "yuv420p",
        ]
    return [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]


def build_renderer(
    video_path: str | Path,
    fit_path: str | Path,
    offset: float = 0.0,
    layout: str = "landscape",
    ride: RideData | None = None,
    clip: "GoProClip | None" = None,
    lockup: str | None = None,
    intro_secs: float = 0.0,
) -> tuple["OverlayRenderer", "GoProClip", RideData]:
    """Build an OverlayRenderer, reusing a pre-parsed ride if provided.

    Returns (renderer, clip, ride) so callers can reuse the ride across
    multiple segments from the same source clip.

    If *clip* is provided (e.g. from extract_all with chapter-adjusted
    timestamps), it is used directly. Otherwise extract_metadata is called,
    which does NOT apply chapter offsets for multi-file GoPro recordings.

    *lockup* is the bottom-band string, built from the ``lockups`` design
    tokens when omitted. Per-ride callers pass the GPS-derived form, e.g.
    "HUDSON YARDS → PIERMONT · 9W".
    """

    video_path = Path(video_path)
    if clip is None:
        clip = extract_metadata(video_path)
    if ride is None:
        ride = parse_fit(fit_path)
    synced = auto_sync(clip, ride, offset)

    src_h = clip.height
    if layout == "portrait":
        out_w = int(src_h * 9 / 16)
        out_w = out_w - (out_w % 2)
        out_h = src_h
        render_clip = replace(clip, width=out_w, height=out_h)
        render_synced = auto_sync(render_clip, ride, offset)
        renderer = OverlayRenderer(render_synced, ride, layout=layout,
                                   lockup=lockup, intro_secs=intro_secs)
    else:
        renderer = OverlayRenderer(synced, ride, layout=layout,
                                   lockup=lockup, intro_secs=intro_secs)

    return renderer, clip, ride


def burn_overlay(
    video_path: str | Path,
    fit_path: str | Path,
    output_path: str | Path,
    offset: float = 0.0,
    layout: str = "landscape",
    start_offset: float = 0.0,
    trim_duration: float = 0.0,
    renderer: OverlayRenderer | None = None,
    ride: RideData | None = None,
    encode_preset: str = ENCODE_PREVIEW,
    portrait_crop_bias: float = 0.0,
    lockup: str | None = None,
    intro_secs: float = 0.0,
    intro_style: str = intro_styles.DEFAULT_STYLE,
    intro_reveal_secs: float = 0.0,
    grade: str = "",
) -> Path:
    """Burn telemetry overlay onto video.

    Args:
        video_path: Source GoPro .mp4 file.
        fit_path: Garmin .fit file.
        output_path: Output .mp4 path.
        offset: Sync offset in seconds (FIT - GoPro).
        layout: "landscape" (16:9) or "portrait" (9:16 center-crop).
        start_offset: Video-time offset in seconds — if burning a segment,
            this is the position in the source file where the segment starts.
            The renderer uses this to look up the correct telemetry.
        trim_duration: If > 0, only process this many seconds starting at
            start_offset. If 0, process the entire video.
        renderer: Pre-built OverlayRenderer to reuse across segments.
            If None, builds a new one (parses FIT + metadata).
        ride: Pre-parsed RideData to avoid re-parsing FIT.
        encode_preset: "preview" (VideoToolbox, fast) or "master"
            (libx264, high quality, +faststart).
        grade: ffmpeg filter chain from grade.build_filter(), applied to the
            footage before the HUD composites. Empty string = no grade.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build or reuse renderer
    if renderer is None:
        print("Parsing metadata...")
        renderer, clip, ride = build_renderer(
            video_path, fit_path, offset, layout, ride=ride, lockup=lockup,
            intro_secs=intro_secs,
        )
    else:
        clip = extract_metadata(video_path)

    # Arm/disable the opener for this specific burn. Keeps a renderer that is
    # cached across segments from leaking the intro onto later clips.
    renderer.set_intro(intro_secs)

    # Determine output dimensions and frame count
    src_w, src_h = clip.width, clip.height
    if layout == "portrait":
        out_w = int(src_h * 9 / 16)
        out_w = out_w - (out_w % 2)
        out_h = src_h
    else:
        out_w, out_h = src_w, src_h

    if trim_duration > 0:
        render_secs = trim_duration
    else:
        render_secs = clip.duration_secs

    total_frames = int(render_secs * clip.fps)
    print(f"Video: {src_w}x{src_h} -> {out_w}x{out_h} @ {clip.fps:.1f}fps, "
          f"{render_secs:.1f}s ({total_frames} frames)")

    # Build ffmpeg command — -ss before input for fast keyframe seek.
    # Add a tiny epsilon (half a frame) so ffmpeg lands past the keyframe
    # boundary, avoiding the black pre-roll frame at the start.
    cmd = ["ffmpeg", "-y"]

    # Input 0: video file (with optional seek)
    if start_offset > 0:
        epsilon = 0.5 / clip.fps  # half a frame duration
        cmd += ["-ss", f"{start_offset + epsilon:.4f}"]
    cmd += ["-i", str(video_path)]

    # Input 1: overlay pipe
    cmd += [
        "-f", "rawvideo", "-pix_fmt", "rgba",
        "-s", f"{out_w}x{out_h}",
        "-r", str(clip.fps),
        "-i", "pipe:0",
    ]

    # Filter: crop (if portrait) → optional intro blur→clear → overlay.
    # The video source label ("base") is [0:v] for landscape, or a cropped
    # branch for portrait. The RGBA overlay (input 1) always composites last.
    filters = []
    if layout == "portrait":
        # portrait_crop_bias: -1.0 = left edge, 0.0 = center, +1.0 = right edge
        max_x = clip.width - out_w
        center_x = max_x / 2
        crop_x = int(center_x + portrait_crop_bias * center_x)
        crop_x = max(0, min(crop_x, max_x))
        filters.append(f"[0:v]crop={out_w}:{out_h}:{crop_x}:0[base]")
        base = "[base]"
    else:
        base = "[0:v]"

    # Grade the footage *before* the HUD composites, never after: the overlay's
    # white-with-black-stroke is a fixed design language and must not inherit the
    # look's contrast curve or colour cast.
    if grade:
        filters.append(f"{base}{grade}[graded]")
        base = "[graded]"

    if intro_secs > 0:
        # Opening reveal. The footage resolves over `reveal`, which is
        # decoupled from intro_secs: the title lockup can hold for the full
        # intro window, but obscured footage past ~2s is where viewers swipe.
        reveal = intro_reveal_secs if intro_reveal_secs > 0 else \
            intro_styles.default_reveal(intro_style, intro_secs)
        filters += intro_styles.build_intro_filter(
            intro_style, base, "[introbase]",
            secs=reveal, width=out_w, height=out_h, fps=clip.fps,
        )
        base = "[introbase]"

    filters.append(f"{base}[1:v]overlay=0:0:format=auto")
    fc = ";".join(filters)

    cmd += ["-filter_complex", fc]

    # Output duration limit (must come after inputs, before output path)
    if trim_duration > 0:
        cmd += ["-t", f"{trim_duration:.2f}"]

    # Encoding
    cmd += _encode_args(encode_preset)
    cmd.append(str(output_path))

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    for f in range(total_frames):
        overlay = renderer.render_frame(start_offset + f / clip.fps)
        proc.stdin.write(overlay.tobytes())
        if (f + 1) % 30 == 0 or f == total_frames - 1:
            print(f"\r  Frame {f + 1}/{total_frames} ({(f + 1) / total_frames * 100:.0f}%)",
                  end="", flush=True)

    proc.stdin.close()
    _, stderr = proc.communicate()
    print()

    if proc.returncode != 0:
        print(f"FFmpeg error:\n{stderr.decode()[-500:]}")
        raise RuntimeError("FFmpeg failed")

    print(f"Output: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path
