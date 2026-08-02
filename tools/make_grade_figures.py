"""Regenerate the colour-grading figures in the README.

Every panel is a real frame from the sample ride, graded by the same
`measure_shot` -> `build_filter` path `process` uses. Nothing here is a mockup
or a hand-written filter chain: if a look changes in grade.py, rerun this and
the README changes with it.

The sample video is not in the repo (it is 14 GB on Hugging Face — see
samples/README.md). Point this at wherever you pulled the full-resolution
chapters:

    python tools/make_grade_figures.py path/to/full

Writes grade-correction.jpg, grade-looks.jpg and grade-strength.jpg into
assets/images/.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from gopro_garmin_pipeline.design.tokens import CARD_BG, CREAM, MINT
from gopro_garmin_pipeline.grade import LOOKS, build_filter, measure_shot

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "assets" / "images"
FONTS = REPO / "src" / "gopro_garmin_pipeline" / "assets" / "fonts"
DEFAULT_FOOTAGE = REPO / "build" / "hf_stage" / "full"

BOLD = str(FONTS / "BarlowCondensed-Bold.ttf")
REG = str(FONTS / "BarlowCondensed-Regular.ttf")
# Any monospace will do; the numbers just need to line up column-wise.
MONO = next((p for p in ("/System/Library/Fonts/Menlo.ttc",
                         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
             if Path(p).exists()), REG)

PANEL_W = 760          # per-panel width while composing
GUTTER = 14
MARGIN = 26
MUTED = (150, 143, 133)
MAX_W = 1760           # GitHub renders README images ~850px wide; 2x for retina

FOOTAGE = DEFAULT_FOOTAGE


def f(path, size):
    return ImageFont.truetype(path, size)


def frame(clip: str, at: float, vf: str = "") -> Image.Image:
    """One real frame, optionally through a real grade filter chain."""
    chain = f"{vf}," if vf else ""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{at:.3f}", "-i", str(FOOTAGE / clip),
           "-frames:v", "1", "-vf", f"{chain}scale={PANEL_W}:-2",
           "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True, check=True)
    return Image.open(io.BytesIO(p.stdout)).convert("RGB")


def label(d, xy, text, font, fill=CREAM, track=0.0):
    """Draw text, optionally letter-spaced (the overlay's label idiom)."""
    x, y = xy
    if not track:
        d.text((x, y), text, font=font, fill=fill)
        return
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + track


def tag(img: Image.Image, text: str, accent: bool = False) -> Image.Image:
    """Burn a corner tag on, so a panel still says what it is once cropped."""
    d = ImageDraw.Draw(img, "RGBA")
    ft = f(BOLD, 27)
    w = d.textlength(text, font=ft) + 26
    d.rectangle([0, 0, w, 40], fill=(10, 10, 10, 215))
    d.text((13, 5), text, font=ft, fill=MINT if accent else CREAM)
    return img


def canvas(w: int, h: int):
    im = Image.new("RGB", (w, h), CARD_BG)
    return im, ImageDraw.Draw(im)


def header(d, y: int, title: str, subtitle: str) -> int:
    label(d, (MARGIN, y), title.upper(), f(BOLD, 38), CREAM, track=2.2)
    d.text((MARGIN, y + 47), subtitle, font=f(REG, 27), fill=MUTED)
    return y + 92


def wrap(d, text: str, font, max_w: float) -> list[str]:
    """Greedy wrap, so recipe strings stay inside their own panel."""
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if d.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def save(im: Image.Image, name: str) -> None:
    if im.size[0] > MAX_W:
        h = round(im.size[1] * MAX_W / im.size[0])
        im = im.resize((MAX_W, h), Image.LANCZOS)
    path = OUT / name
    im.save(path, quality=84, optimize=True, progressive=True)
    print(f"  {name}  {im.size[0]}x{im.size[1]}  {path.stat().st_size / 1024:.0f} KB")


# ── Figure 1: per-shot correction across the ride ────────────────────
# Three shots from three different hours. The point is that the measured
# corrections point in OPPOSITE directions — which is the whole argument for
# correcting per shot instead of once for the ride.
SHOTS = [
    ("GX010336.MP4", 23.0, "0:25", "low morning sun, GWB in view",
     "warms, pulls down"),
    ("GX010342.MP4", 102.0, "1:29", "9W under heavy canopy",
     "lifts, pulls the green cast — both gains pinned at the ±7% clamp"),
    ("GX010348.MP4", 127.0, "2:41", "open afternoon sky, Riverside Drive",
     "cools, pulls down hardest"),
]


def fig_correction() -> None:
    rows = []
    for clip, at, tstamp, note, verdict in SHOTS:
        bal = measure_shot(FOOTAGE / clip, at)
        vf = build_filter(bal, look="none")
        rows.append((tstamp, note, verdict, bal,
                     frame(clip, at), frame(clip, at, vf)))

    ph = rows[0][4].size[1]
    w = MARGIN * 2 + PANEL_W * 2 + GUTTER
    im, d = canvas(w, 116 + (ph + 116) * len(rows) + 70)

    y = header(d, MARGIN, "Per-shot correction  ·  --wb shot",
               "Same ride, three hours apart. Frames are real; the correction "
               "is measured by grade.py at each clip's anchor.")

    for tstamp, note, verdict, bal, raw, cor in rows:
        d.text((MARGIN, y - 4), f"{tstamp}  ·  {note}", font=f(REG, 26), fill=MINT)
        y += 32
        im.paste(tag(raw, "AS SHOT"), (MARGIN, y))
        im.paste(tag(cor, "CORRECTED", accent=True), (MARGIN + PANEL_W + GUTTER, y))
        y += ph + 10
        # The measured numbers, so a reader can check the claim rather than
        # take the pixels on faith.
        stat = (f"black {bal.b:.3f}   white {bal.w:.3f}   "
                f"R×{bal.gr:.3f}  B×{bal.gb:.3f}   exposure {bal.ev:+.2f} EV")
        d.text((MARGIN, y), stat, font=f(MONO, 21), fill=MUTED)
        d.text((MARGIN, y + 32), verdict, font=f(REG, 24), fill=CREAM)
        y += 74

    d.text((MARGIN, y - 6),
           "Three shots, three different directions: warm down, lift, cool down. "
           "The exposure nudges alone span 0.74 EV.",
           font=f(REG, 25), fill=CREAM)
    d.text((MARGIN, y + 30),
           "That spread is the argument for correcting per shot — no single "
           "grade fits hour 1 and hour 3.",
           font=f(REG, 25), fill=MUTED)
    save(im, "grade-correction.jpg")


# ── Figure 2: the look shelf ─────────────────────────────────────────
LOOK_ORDER = ["none", "house", "warm-afternoon", "cool-morning",
              "soft-film", "overcast-lift"]


def fig_looks() -> None:
    clip, at = "GX010338.MP4", 60.0     # GWB reveal: canopy, sky, road, stone
    bal = measure_shot(FOOTAGE / clip, at)
    cells = [(name, frame(clip, at, build_filter(bal, look=name, strength=0.35)))
             for name in LOOK_ORDER]

    ph = cells[0][1].size[1]
    cols, rows_n, cap_h = 3, 2, 86
    w = MARGIN * 2 + PANEL_W * cols + GUTTER * (cols - 1)
    im, d = canvas(w, 116 + (ph + cap_h) * rows_n + 54)

    y = header(d, MARGIN, "The look shelf  ·  --look",
               "One frame, one --wb shot correction, five creative looks at "
               "the default --look-strength 35.")

    mono = f(MONO, 19)
    for i, (name, img) in enumerate(cells):
        cx = MARGIN + (i % cols) * (PANEL_W + GUTTER)
        cy = y + (i // cols) * (ph + cap_h)
        im.paste(img, (cx, cy))
        recipe = LOOKS[name]
        txt = "as corrected — no look" if not recipe else "  ".join(
            f"{k} {v:+g}" for k, v in recipe.items())
        label(d, (cx, cy + ph + 8), name.upper(), f(BOLD, 28),
              MINT if name != "none" else CREAM, track=1.6)
        for j, line in enumerate(wrap(d, txt, mono, PANEL_W)[:2]):
            d.text((cx, cy + ph + 44 + j * 22), line, font=mono, fill=MUTED)

    d.text((MARGIN, y + (ph + cap_h) * rows_n - 2),
           "Recipes are written at full strength in grade.py and land scaled. "
           "They lean on vibrance over saturation and pull highlights down "
           "rather than pushing exposure up — a blown sky is the one thing "
           "this footage cannot recover.",
           font=f(REG, 25), fill=MUTED)
    save(im, "grade-looks.jpg")


# ── Figure 3: strength ramp ──────────────────────────────────────────
def fig_strength() -> None:
    clip, at, look = "GX010346.MP4", 1.0, "warm-afternoon"
    bal = measure_shot(FOOTAGE / clip, at)
    steps = [0, 25, 35, 50, 75, 100]
    cells = [(s, frame(clip, at, build_filter(bal, look=look, strength=s / 100)))
             for s in steps]

    ph = cells[0][1].size[1]
    cols = 3
    w = MARGIN * 2 + PANEL_W * cols + GUTTER * (cols - 1)
    im, d = canvas(w, 116 + (ph + 52) * 2 + 54)

    y = header(d, MARGIN, "Strength is the whole dial  ·  --look-strength",
               f"The same '{look}' recipe from 0 to 100. 35 is the default.")

    notes = {0: "  correction only", 35: "  default", 100: "  full recipe"}
    for i, (s, img) in enumerate(cells):
        cx = MARGIN + (i % cols) * (PANEL_W + GUTTER)
        cy = y + (i // cols) * (ph + 52)
        im.paste(img, (cx, cy))
        accent = s == 35
        label(d, (cx, cy + ph + 8), f"{s}", f(BOLD, 28),
              MINT if accent else CREAM, track=1.6)
        d.text((cx + 46, cy + ph + 13), notes.get(s, ""), font=f(MONO, 19),
               fill=MINT if accent else MUTED)

    d.text((MARGIN, y + (ph + 52) * 2 - 2),
           "Past roughly 50 it stops reading as a grade and starts reading as "
           "a filter. The default is deliberately conservative.",
           font=f(REG, 25), fill=MUTED)
    save(im, "grade-strength.jpg")


def main() -> None:
    global FOOTAGE
    if len(sys.argv) > 1:
        FOOTAGE = Path(sys.argv[1]).expanduser().resolve()
    if not FOOTAGE.is_dir():
        raise SystemExit(
            f"No footage at {FOOTAGE}.\n"
            "Pull the sample ride's full-resolution chapters from Hugging Face "
            "(see samples/README.md) and pass the directory:\n"
            "    python tools/make_grade_figures.py path/to/full")

    needed = {c for c, *_ in SHOTS} | {"GX010338.MP4", "GX010346.MP4"}
    missing = sorted(needed - {p.name for p in FOOTAGE.glob("*.MP4")})
    if missing:
        raise SystemExit(f"Missing chapters in {FOOTAGE}: {', '.join(missing)}")

    print(f"Rendering grading figures from {FOOTAGE}:")
    fig_correction()
    fig_looks()
    fig_strength()


if __name__ == "__main__":
    main()
