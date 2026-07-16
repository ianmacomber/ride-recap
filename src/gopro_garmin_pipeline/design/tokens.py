"""Python loader for design/tokens.json — single source of truth.

Use the semantic names (TEXT_PRIMARY, ACCENT_PRIMARY, etc.) when meaning
matters more than hex. Tokens are cached at import time; restart the
process if you edit tokens.json.

If a token you need isn't exposed here yet, add it to tokens.json first,
then re-export. Never hard-code a hex/font path in a module — it dilutes
the system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

# Repo root (fonts live in repo-level assets/, not package data):
# design/ → gopro_garmin_pipeline/ → src/ → repo root.
REPO = Path(__file__).resolve().parents[3]
TOKENS_PATH = Path(__file__).resolve().parent / "tokens.json"

with TOKENS_PATH.open() as _f:
    _T = json.load(_f)


# ─── Hex helpers ──────────────────────────────────────────────
def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def hex_to_rgba(h: str, a: int = 255) -> Tuple[int, int, int, int]:
    return (*hex_to_rgb(h), a)


# ─── Colors ───────────────────────────────────────────────────
_PRIM = _T["color"]["primitive"]
_SEM = _T["color"]["semantic"]

PAPER       = hex_to_rgb(_PRIM["paper"])
INK         = hex_to_rgb(_PRIM["ink"])
GRAPHITE    = hex_to_rgb(_PRIM["graphite"])
CREAM       = hex_to_rgb(_PRIM["cream"])
NEAR_BLACK  = hex_to_rgb(_PRIM["near_black"])
WARM_DARK   = hex_to_rgb(_PRIM["warm_dark"])
RULE_DIM    = hex_to_rgb(_PRIM["rule_dim"])
SAGE        = hex_to_rgb(_PRIM["sage"])
MINT        = hex_to_rgb(_PRIM["mint"])
WHITE       = hex_to_rgb(_PRIM["white"])
BLACK       = hex_to_rgb(_PRIM["black"])

CARD_BG          = hex_to_rgb(_SEM["card_bg"])
TEXT_PRIMARY     = hex_to_rgb(_SEM["text_primary"])
TEXT_MUTED       = hex_to_rgb(_SEM["text_muted"])
ACCENT_PRIMARY   = hex_to_rgb(_SEM["accent_primary"])
ACCENT_SECONDARY = hex_to_rgb(_SEM["accent_secondary"])
STROKE           = hex_to_rgb(_SEM["stroke"])

# Effort zones — cold-to-hot palette for power and HR bars.
_Z = _T["color"]["zones"]
ZONE_COLORS = [hex_to_rgb(_Z[k]) for k in ("z1", "z2", "z3", "z4", "z5", "z6")]


# ─── Fonts ────────────────────────────────────────────────────
def _font_path(rel: str) -> str:
    """Resolve a font path. Absolute paths returned as-is; relative paths
    are resolved against the repo root."""
    if rel.startswith("/"):
        return rel
    return str(REPO / rel)


def _first_existing(candidates: list[str], fallback: str) -> str:
    """First candidate font that exists on this machine, else *fallback*.

    The italic serif is the one face the repo does not ship — Times New Roman
    is not redistributable, so it is sourced from the OS. macOS and most Linux
    distros keep it (or a metric-compatible clone) in different places, and a
    machine with none of them still has to render the outro card, so we fall
    back to the bundled body face rather than raising at import time.
    """
    for c in candidates:
        if c and Path(c).exists():
            return c
    return fallback


_FONTS = _T["typography"]["fonts"]
FONT_NUMERIC      = _font_path(_FONTS["numeric"]["file"])
FONT_NUMERIC_REG  = _font_path(_FONTS["numeric_regular"]["file"])
FONT_BODY         = _font_path(_FONTS["body"]["file"])

_SERIF_CANDIDATES = [
    _font_path(_FONTS["serif"]["file"]),                        # configured
    "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",  # macOS
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",  # Debian
    "/usr/share/fonts/liberation-serif/LiberationSerif-Italic.ttf",     # Fedora
    "/usr/share/fonts/TTF/LiberationSerif-Italic.ttf",                  # Arch
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",          # fallback
]
FONT_SERIF = _first_existing(_SERIF_CANDIDATES, FONT_BODY)

STROKE_RATIO_TO_FONT   = _T["typography"]["stroke_ratio_to_font"]
LABEL_TRACKING_RATIO   = _T["typography"]["label_tracking_ratio_to_font"]


# ─── Lockups ──────────────────────────────────────────────────
LOCKUP_ORIGIN   = _T["lockups"]["origin_default"]
LOCKUP_ROAD     = _T["lockups"]["road_default"]
LOCKUP_CREW     = _T["lockups"]["crew"]
LOCKUP_SUBTITLE = _T["lockups"]["outro_subtitle"]


def default_lockup() -> str:
    """The bottom-band lockup string used when no per-ride override is set."""
    return f"{LOCKUP_ORIGIN}   ·   {LOCKUP_ROAD}"


def lockup_with_destination(destination: str) -> str:
    """e.g. lockup_with_destination('PIERMONT') → '{ORIGIN} → PIERMONT · {ROAD}'."""
    if not destination:
        return default_lockup()
    return f"{LOCKUP_ORIGIN} → {destination.upper()}   ·   {LOCKUP_ROAD}"


# ─── Voice ────────────────────────────────────────────────────
VOICE_TAGLINE = _T["voice"]["tagline"]
SUBTITLE_EXAMPLES = list(_T["voice"]["subtitle_examples"])


def all_tokens() -> dict:
    """Raw parsed tokens.json — for debugging or one-off lookups."""
    return _T
