# Sample ride — 2026-07-10

A real ride you can point the pipeline at. 44.7 miles, 3h09m elapsed, Manhattan
up 9W and back, recorded on a handlebar GoPro with a Garmin Edge alongside.
19 chapters, 10,381 GPS points.

The sidecars live here in the repo (~116 KB). The video lives on Hugging Face,
because it is 14 GB.

## What's in this folder

| File | What it is |
|---|---|
| `ride_labels.json` | 9 clips I rated by hand. The ground truth. |
| `.gemini_cache/` | Gemini's `v10` ratings for all 19 chapters. The baseline. |
| `sync.json` | Per-chapter video↔ride-time offsets from GPMF. |
| `moments.json` | Merged scan output — what `process` produced from the above. |

Those four are enough to reproduce the scoring comparison **without downloading
any video and without a Gemini API key**. That's the point of committing them.

## Getting the video

The dataset is tiered — pull only what you need:

- `sidecars/` (~1 MB) — same files as this folder plus the Garmin `.fit`, for
  people who didn't clone. (The Gemini cache is `gemini_cache/` there, without
  the leading dot, because the Hub dislikes dot-paths.)
- `clips/` (6.3 GB) — the 8 chapters my labels reference, full resolution,
  plus their LRV proxies. This is the tier issue #2 needs.
- `full/` (14 GB) — all 19 chapters plus the `.fit`, for testing `process`
  end to end.

To get a runnable ride folder (swap `clips` for `full` if you want everything):

```bash
pip install huggingface_hub
hf download iandmacomber/ride-recap-sample-2026-07-10 --repo-type dataset \
    --include "clips/*" "sidecars/*" --local-dir hf_ride
mkdir -p data/raw
mv hf_ride/clips data/raw/2026-07-10
mv hf_ride/sidecars/gemini_cache data/raw/2026-07-10/.gemini_cache
mv hf_ride/sidecars/* data/raw/2026-07-10/
```

`data/raw/2026-07-10/` is the path every command in the main README already
uses, so everything there now runs verbatim.

## The two scoring scales

This trips people up, so read it before you conclude your adapter is broken.

**My labels** (`ride_labels.json`) use two dimensions, `visual` and `action`,
each 1–10, tagged `scale_version: 2`. Labels written before that version used
1–5 and are doubled on read by `normalize_label_scale`.

**Gemini's ratings** (`.gemini_cache/`) use the five independent dimensions from
`prompts/gemini_scan/v10.md` — `light`, `composition`, `motion`, `scenery`,
`subject` — each 1–10.

`compare` reconciles them by folding the rubric onto the label axes:
`visual` = mean(composition, scenery), `action` = mean(motion, subject).
`light` is dropped, matching `composer._rubric_score`, which excludes it
because it doesn't discriminate between clips.

If your model emits a different rubric, `_hit_scores` in `prompt_eval.py` is
the one place that needs to know about it.

## What the rubric is asking for

`v10` frames the task as calibrating a distribution, not clearing a bar. From
the prompt:

> A "good ride" generates maybe 3-8 clips that pass this bar. The other 50+
> clips you'll see are filler, transit footage, or "fine but forgettable." That
> is normal.

So a model that rates everything a 7 is failing, even though nothing it said
was wrong. Dispersion is the thing being measured. If you're testing another
model, that's the first place to look — my guess is most of them flatten toward
the middle, but I haven't checked, which is the whole reason issue #2 is open.

## Using this for issue #2

The `.gemini_cache/` ratings are the baseline. Run your adapter over the same
19 chapters, then compare both against `ride_labels.json`. The 9 labels are
sparse and they're one rider's taste on one ride — treat them as a smell test,
not a benchmark. Findings like "model X over-scores tunnels" are useful on
their own; you don't need a PR.

Five labels are marked `must_include: true` — the GWB, the Hudson stretch with
speed, a deer, a bridge, and Hudson Yards. If a model misses those, that's more
interesting than any aggregate score. The deer is two seconds long and is
probably the hardest single thing in this dataset to catch.

## Note on the data

This is unmodified ride data. The FIT and the GPMF track inside every MP4 and
LRV carry the full per-second GPS trace, including where the ride started and
ended. Published deliberately, but you should know it's there.

Shot on a single ride, so it's one rider, one bike, one summer morning, one
camera angle. Don't over-fit to it.
