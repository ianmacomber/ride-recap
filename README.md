# ride-recap

**Turn three hours of raw GoPro footage and a Garmin FIT file into a 30-second highlight reel with the telemetry burned in.**

![A frame from a finished landscape reel](assets/images/overlay-landscape.jpg)

I ride ~3 hours most weekends, usually out of Manhattan and up 9W. By the end of a ride I have twenty-odd GoPro clips and one FIT file with per-second speed, power, heart rate, cadence, and GPS. Nobody wants to watch three hours of cycling footage — not even me. But somewhere in there are 30 good seconds.

This finds them. One command, about ten minutes, roughly four cents of Gemini:

```bash
gopro-garmin process data/raw/2026-07-10/
```

Out the other end: `highlight_landscape.mp4` (60s, 16:9) and `highlight_portrait.mp4` (30s, 9:16), clips always in ride order, HUD burned in.

There's a long write-up of how it works and everything I got wrong building it: **[Teaching LLMs Taste](https://iandmacomber.com/blog/gopro-garmin-gemini-ride-recap)**.

---

## The idea

Three sensors that don't know about each other, fused into one chronological story.

![Pipeline architecture](assets/images/architecture.jpg)

Four sources propose candidate moments independently — Garmin telemetry (power spikes, sprints, climbs), Strava (popular segments), a Gemini vision scan of the actual footage, and optional hand-labels. Fusion merges them, scores them, drops the boring ones, and hands ~90 candidates to a reviewer. You pick; it burns.

**The central bet is that telemetry alone can't find the good parts.** My first scorer flagged anything above 350W and produced sixty seconds of me grinding on flat, featureless roads. High wattage on a suburban straightaway looks identical to high wattage on the George Washington Bridge, and the Garmin can't tell the difference. Meanwhile the best moment of a ride is often this:

![One moment, end to end](assets/images/moment-trace.jpg)

139 watts. No spike, no sprint, no starred segment — to the power meter that moment does not exist. The vision scan caught it and it shipped as clip 3. That's the whole reason there's a model in the loop.

The corollary, learned the hard way when an 858W spike behind a delivery truck outranked a river-skyline climb: **the visual rubric is the score of record, and telemetry is a tiebreaker.** Telemetry-only candidates are capped at 3.0/10 as "visually blind."

---

## Setup

**You need:** Python 3.11+, [ffmpeg](https://ffmpeg.org/download.html), and a Gemini API key. Everything else is optional.

```bash
git clone https://github.com/ianmacomber/ride-recap.git
cd ride-recap
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# macOS: brew install ffmpeg     Debian/Ubuntu: sudo apt install ffmpeg

cp .env.example .env
```

Then open `.env` and set two things:

```env
GEMINI_API_KEY=your-key-here    # https://aistudio.google.com/apikey

FTP=240                          # ← YOUR functional threshold power
MAX_HEART_RATE=196               # ← YOUR max heart rate
```

**Set FTP and MAX_HEART_RATE.** They ship with my numbers and they are load-bearing, driving three things: the spike thresholds that decide what counts as a hard effort, the HUD's zone colors, and the power zones written into the Gemini prompt so the model knows whether 216W means "cruising" or "peak effort" *for you*. That last one is why they matter more than they look — Gemini once called a 216W park climb "peak effort" because it was guessing from raw numbers. Leave mine in and the pipeline will systematically misjudge your riding: too many candidates if your FTP is lower than mine, too few if it's higher. This is the one bit of config that isn't optional.

Verify it runs:

```bash
gopro-garmin --help
pytest                    # 11 smoke tests, no video or API keys needed
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

The FIT comes off the Edge over USB, or `gopro-garmin garmin-download --date 2026-07-10 --output-dir data/raw/2026-07-10`.

**Copy the `.LRV` files off the SD card.** GoPro already writes an 848×480 H.264 proxy next to every recording. The scan downscales to 480px anyway, so the proxy is exactly the resolution needed and decodes 10-20x faster — frame extraction drops from ~25 minutes to ~30 seconds. The pipeline auto-detects them and still burns from the full-res `.MP4`. Look for the artifact the hardware already produces before optimizing the computation.

Then:

```bash
gopro-garmin process data/raw/2026-07-10/
```

It'll ask five questions (start, destination, road, a saying, who you rode with — all with GPS-derived defaults, just hit Enter), sync the clocks, scan the footage, and open a Streamlit reviewer at `:8501` with 5-second preview clips.

**`process` stops at the reviewer.** Pick your clips, then run the command it prints:

```bash
gopro-garmin compose-selected data/raw/2026-07-10/selected_candidates.json \
    data/raw/2026-07-10 data/raw/2026-07-10/ride.fit
```

Or skip the human entirely:

```bash
gopro-garmin process data/raw/2026-07-10/ --skip-review
```

That's the one command that goes end to end. The autonomous output is usually good.

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

One thing worth knowing: in June my Gemini billing lapsed, every call 403'd, the scan cached the emptiness as a valid result, and the pipeline quietly degraded to telemetry-only. The reel was mediocre and I spent an evening blaming the prompt. Now failures are never cached and a `DEGRADED` banner prints instead. **Silent degradation is strictly worse than a crash — you end up debugging the wrong system.**

---

## Tuning it to your eye

The interesting config isn't in `.env`.

**`prompts/gemini_scan/v10.md`** is the most important file in the repo. It's the system instruction for the vision scan — a five-dimension rubric (light, composition, motion, scenery, subject, each 1–10) with anchored examples and a set of named rules, each one traceable to a specific clip I hated. My taste is legislated in there: the openness gate exists because a rail-trail tunnel scored 7.2 and I think tunnels look like being stuck in a concrete tube. **You may disagree.** Yours is a different file — write `v11.md`.

Prompts are immutable and versioned: never edit v10, write v11. The version string is baked into the scan's cache key, so a bump invalidates exactly the affected results and nothing else. Frontmatter requires a rationale explaining what failure prompted the version.

**`src/gopro_garmin_pipeline/design/tokens.json`** owns the entire look — colors, fonts, and the lockup defaults. Rebrand there, not in the drawing code.

If you want to teach it your taste systematically, there's a loop for that: label a ride, scan it, diff your ratings against the model's, fix the prompt.

```bash
gopro-garmin extract-frames <fit> <dir>   # once per ride, ~2 min
gopro-garmin label <fit> <dir> --offset <secs>
gopro-garmin compare <date-folder>        # your labels vs the scan
gopro-garmin eval-prompt <folder>...      # feeds the diff back for suggestions
```

It took me an evening and produced prompt versions 3 through 5. It isn't retraining a model; it's coaching a video editor by showing them the cuts you didn't like.

---

## Commands

```bash
gopro-garmin process <date-folder>            # the main one
gopro-garmin process <date-folder> --skip-review

gopro-garmin compose <video-dir> <fit>        # compose without the reviewer
gopro-garmin compose-selected <sel> <dir> <fit>
gopro-garmin review-candidates <video-dir> <fit>
gopro-garmin burn <video> <fit> -o out.mp4    # overlay a single clip

gopro-garmin inspect-fit <fit>                # ride stats, no video needed
gopro-garmin find-highlights <fit>            # telemetry highlights only
gopro-garmin garmin-download --date 2026-07-10
gopro-garmin strava-segments <activity-id>
```

`ride-recap` works as an alias for `gopro-garmin` everywhere.

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
  design/tokens.json  # All color, type, and lockup tokens
prompts/              # Versioned, immutable LLM prompts
tests/                # Smoke tests
```

Two design notes, if you're reading the code. `LayoutGeometry` holds every layout-varying dimension, so there are zero landscape/portrait conditionals in the drawing code — adding a layout is one new dataclass instance. And overlay rendering pipes raw RGBA frames straight to ffmpeg's stdin; portrait center-crops 16:9→9:16 in the same encoding pass.

---

## Caveats

- **Built for one setup**: GoPro Hero 13 + Garmin Edge 540 + road cycling. Other GoPros should work (GPMF sync reads the standard telemetry track); other sports are untested and the prompt is full of cycling assumptions.
- **macOS is the tested platform.** It should run on Linux — encoder selection and font fallback both probe rather than assume — but I haven't tested it there. If it breaks, that's a bug worth filing.
- **The learned ranker is scaffolding.** `learned_ranker.py` logs training data on every compose and will fit a logistic regression at ~100 examples. I never reached the threshold before the feature schema went stale. It's honest to call it unfinished.
- Video and FIT files are gitignored. Don't commit your rides.

## License

MIT — see [LICENSE](LICENSE). Bundled fonts (Barlow Condensed, Inter) are SIL OFL 1.1; see [assets/fonts/OFL.txt](assets/fonts/OFL.txt).

Contributions welcome, but this is a personal project I use most weekends — I'd rather it stay small and opinionated than grow into a framework. If you fork it and make it yours, tell me; I'd like to see your reel.
