# ride-recap

**TL;DR:** Turn hours of raw GoPro footage + a `.fit` file into a 60-second highlight reel with ride telemetry burned in. Every second of the ride is ranked and edited by a LLM. The whole thing costs about $0.04 per ride and takes 10 minutes.

*Alternate **TL;DR:** How I managed to turn 3 hours of boring GoPro footage into 30 seconds of boring GoPro footage.*

![A frame from a finished landscape reel](assets/images/overlay-landscape.jpg)

## Summary

I ride most weekends, usually out of Manhattan and up 9W. By the end of a ride I have hours of GoPro footage and one `.fit` file with per-second speed, power, heart rate, cadence, and GPS. Absolutely no one wants to watch 3 hours of being stuck behind Citibikes on West Side Highway. It's fun to look through past footage, identify the fun parts, and put together a narrative to remember. But since cycling is already time consuming, manually editing a highlight reel edit per ride is a nonstarter. So I built and open-sourced this repo.

One command, about ten minutes, roughly four cents of `gemini-3.5-flash`:

```bash
ride-recap process data/raw/2026-07-10/
```

The outputs: `highlight_landscape.mp4` (60s, 16:9) and `highlight_portrait.mp4` (30s, 9:16). The clips are always in ride order with telemetry burned in.

**▶ Click the image below to watch a finished highlight** (River Road to 9W Market):

[![Watch a ride-recap reel on YouTube](https://img.youtube.com/vi/ZBrneOOYmG0/maxresdefault.jpg)](https://www.youtube.com/watch?v=ZBrneOOYmG0)

Here's my long write-up of how it works and everything I got wrong building it: **[Teaching LLMs Taste](https://iandmacomber.com/blog/gopro-garmin-gemini-ride-recap)**.

---

## The Architecture

![Pipeline architecture](assets/images/architecture.jpg)

We identify compelling moments from four separate sources:

1. Garmin telemetry via `.fit` file (speed, HR, power spikes, sprints, climbs)
2. Strava via API (popular segments)
3. Gemini vision scan + rating of individual frames
4. (optional) hand-labels via Streamlit app

Here's an example of one moment being scored across all sources:

![One moment, end to end](assets/images/moment-trace.jpg)
  
The fusion step works like this:

1. “Must include” manual labels are picked first
2. A Gemini LLM narrative pass picks 20 clips to best tell the story of the ride, boosting “cross-source agreements” (if a human label + telemetry + Strava + Gemini all agree that a clip is interesting)
3. Greedy re-ranking with a crowding penalty to avoid clips too close to something already selected
   1. First, a “quality” phase that picks best-first until the best remaining candidate is net-negative with crowding penalty
   2. Second, a “coverage” fill that guarantees the exact clip count by placing the liveliest available clips into the biggest timeline holes.

Generally, the visual rubric is the score of record, and telemetry is a tiebreaker.

---

## Setup

**You need:** Python 3.11+, [ffmpeg](https://ffmpeg.org/download.html), and a Gemini API key. You also need a road bike mounted with a GoPro and a Garmin.

```bash
git clone https://github.com/ianmacomber/ride-recap.git
cd ride-recap
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# macOS: brew install ffmpeg     Debian/Ubuntu: sudo apt install ffmpeg

cp .env.example .env
```

Then open `.env` and set three things:

```env
GEMINI_API_KEY=your-key-here    # https://aistudio.google.com/apikey

FTP=240                          # ← YOUR functional threshold power
MAX_HEART_RATE=196               # ← YOUR max heart rate
```

**Set FTP and MAX_HEART_RATE.** These numbers drive three things: the spike thresholds that decide what counts as a hard effort, the HUD's zone colors, and the power zones written into the Gemini prompt so the model knows whether 216W means "cruising" or "peak effort" for you.

Verify it runs:

```bash
ride-recap --help
pytest                    # smoke tests, no video or API keys needed
```

---

## Your first ride

Put everything for one ride in a single date folder — GoPro `.MP4`s and the Garmin `.fit` together:

```
data/raw/2026-07-10/
├── GX010123.MP4
├── GX010124.MP4
├── GL010123.LRV     ← copy these too, see below
├── GL010124.LRV
└── ride.fit
```

The FIT comes off the Edge over USB, or `ride-recap garmin-download --date 2026-07-10 --output-dir data/raw/2026-07-10`.

**Copy the `.LRV` files off the SD card.** GoPro already writes an 848×480 H.264 proxy next to every recording. The scan downscales to 480px anyway, so the proxy is exactly the resolution needed and decodes 10-20x faster — frame extraction drops from ~25 minutes to ~30 seconds. The pipeline auto-detects them and still burns from the full-res `.MP4`. Look for the artifact the hardware already produces before optimizing the computation.

Then:

```bash
ride-recap process data/raw/2026-07-10/
```

It'll ask five questions (start, destination, road, a saying, who you rode with, all with GPS-derived defaults, just hit Enter), sync the clocks, scan the footage, and open a Streamlit reviewer at `:8501` with 5-second preview clips.

**`process` stops at the reviewer.** Pick your clips, then run the command it prints:

```bash
ride-recap compose-selected data/raw/2026-07-10/selected_candidates.json \
    data/raw/2026-07-10 data/raw/2026-07-10/ride.fit
```

Or skip the human entirely:

```bash
ride-recap process data/raw/2026-07-10/ --skip-review
```

That command runs end to end. The autonomous output is usually good.

---

## Degrading gracefully

Only ffmpeg, a FIT file, and at least one `.MP4` are actually required. Everything else turns itself off politely:

| Missing | What happens |
|---|---|
| `GEMINI_API_KEY` | Prints a warning, falls back to telemetry + Strava candidates and greedy selection. Runs, but this is the part that finds the good clips. |
| Strava credentials | Only runs if you pass `--strava-activity` anyway. Warns and continues. |
| Garmin credentials | Only needed for `garmin-download`. `process` never touches them. |
| `OSM_CONTACT_EMAIL` | No GPS-derived place names; falls back to your `--origin`/`--road` flags or the design tokens. |
| rclone | Skips the Drive upload. Files stay in `<date>/highlights/`. |
| `.LRV` proxies | Falls back to the `.MP4` with keyframe-only decode. Slower, same result. |
| Labels | Fully optional. This is the default mode. |
| ffmpeg | Hard stop, up front, with install instructions. |

---

## Tuning it to your eye

**`src/gopro_garmin_pipeline/prompts/gemini_scan/v10.md`** is the most important file in the repo. It's the system instruction for the vision scan: a five-dimension rubric (light, composition, motion, scenery, subject, each 1–10) with anchored examples and a set of named rules. Each rule is traceable to a specific clip that Gemini skipped and I loved, or Gemini included and I hated. My taste is legislated in there: the openness gate exists because a rail-trail tunnel scored 7.2 and I think tunnels look like being stuck in a concrete tube. **You may disagree.** If your taste is different, write `v11.md`.

Prompts are immutable and versioned: never edit v10, write v11. The version string is baked into the scan's cache key, so a bump invalidates exactly the affected results and nothing else. Frontmatter requires a rationale explaining what failure prompted the version.

**`src/gopro_garmin_pipeline/design/tokens.json`** owns the design: colors, fonts, and the lockup defaults. Rebrand there, not in the drawing code.

If you want to teach it your taste systematically: label a ride, scan it, diff your ratings against the model's, change the prompt.

```bash
ride-recap extract-frames <fit> <dir>   # once per ride, ~2 min
ride-recap label <fit> <dir> --offset <secs>
ride-recap compare <date-folder>        # your labels vs the scan
ride-recap eval-prompt <folder>...      # feeds the diff back for suggestions
```

This is basically coaching a video intern by showing them the cuts you didn't like.

---

## Commands

```bash
ride-recap process <date-folder>            # the main one
ride-recap process <date-folder> --skip-review

ride-recap compose <video-dir> <fit>        # compose without the reviewer
ride-recap compose-selected <sel> <dir> <fit>
ride-recap review-candidates <video-dir> <fit>
ride-recap review <video> <fit>             # single-clip overlay preview (Flask)
ride-recap burn <video> <fit> -o out.mp4    # overlay a single clip

ride-recap inspect-fit <fit>                # ride stats, no video needed
ride-recap find-highlights <fit>            # telemetry highlights only
ride-recap garmin-download --date 2026-07-10
ride-recap strava-segments <activity-id>
```

`gopro-garmin` works as an alias for `ride-recap` everywhere (it's the original name, still used in the [write-up](https://iandmacomber.com/blog/gopro-garmin-gemini-ride-recap) and its figures).

---

## Layout

```
src/gopro_garmin_pipeline/
  cli.py              # Click CLI — every command above
  fit_parser.py       # .fit → RideData (power, HR, speed, GPS)
  gpmf_sync.py        # GoPro GPS-time → FIT offset. The clock sync that matters.
  gemini_scan.py      # Two-pass vision scan (coarse regions → fine rubric)
  composer.py         # Candidate generation, fusion, selection, composition
  burn_overlay.py     # The 5-element HUD, landscape + portrait
  intro_outro.py      # Blur→title opener, outro recap card
  route_metadata.py   # Start / Far / Road from GPS via OSM + Overpass
  highlights.py       # Telemetry spike detection
  strava.py           # Segment efforts + star counts
  candidate_review.py # Streamlit reviewer
  labeler.py          # Streamlit labeler (ground truth for prompt work)
  web/                # Flask single-clip overlay preview (review)
  design/tokens.json  # All color, type, and lockup tokens
  prompts/            # Versioned, immutable LLM prompts (ship with the package)
  assets/fonts/       # Bundled OFL faces (ship with the package)
tests/                # Smoke tests
```

---

## Caveats

* **Built for one setup**: GoPro Hero 13 + Garmin Edge 540 + road cycling. Other cameras and head units should work (GPMF sync reads the standard telemetry track), but are untested.
* **I only tested on macOS**: should work for Linux but I haven't tested yet.
* **The learned ranker is WIP.** `learned_ranker.py` logs training data on every compose and will fit a logistic regression at ~100 examples. I never reached the threshold before the feature schema went stale. It's honest to call it unfinished.
* Video and FIT files are gitignored.

## License

MIT — see [LICENSE](LICENSE). Bundled fonts (Barlow Condensed, Inter) are SIL OFL 1.1; see [src/gopro_garmin_pipeline/assets/fonts/OFL.txt](src/gopro_garmin_pipeline/assets/fonts/OFL.txt).

Contributions welcome, but this is a personal project I use most weekends and I have a real job. If you fork it and make it yours, tell me! I'd love to see your reel.
