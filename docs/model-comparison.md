# Vision-model comparison for issue #2

This is a small, reproducible comparison of the unchanged `v10` vision rubric
across three providers:

- `gemini-3.5-flash`
- `gpt-4.1-mini`
- `mlx-community/Qwen3-VL-8B-Instruct-3bit`, served locally by MLX-VLM

The test uses the eight video chapters in the `clips/` tier of the
`iandmacomber/ride-recap-sample-2026-07-10` dataset. Those chapters contain all
nine hand labels, including the five `must_include` moments. The hand labels
are sparse and represent one rider's taste on one ride, so the numbers are a
smell test rather than a benchmark.

## Method

Each provider ran the same two-pass scan and versioned prompts. Provider cache
directories are isolated, and non-Gemini cache filenames include a model
fingerprint. Comparisons were restricted to the eight labeled chapters, so
Gemini's cached ratings for the other eleven chapters did not inflate its
model-only count. Gemini retains legacy cache filenames for compatibility.

The comparison folds the five-dimension rubric onto the two hand-label axes:

- `visual = mean(composition, scenery)`
- `action = mean(motion, subject)`
- `light` is excluded, matching the production selector

A label and model rating count as a temporal match when their ride timestamps
are within 15 seconds. Labels are injected blindly as forced fine-pass regions,
so recall does not by itself measure discovery. Coarse-pass behavior and the
semantic explanation must be interpreted alongside it.

## Results

| Model | Fine ratings | Temporal matches | Recall | Nominal precision | Visual SD (range) | Action SD (range) |
|---|---:|---:|---:|---:|---:|---:|
| Gemini 3.5 Flash | 24 | 8/9 | 89% | 33% | 1.68 (2.5-8.5) | 1.27 (2.5-7.5) |
| GPT-4.1 mini | 23 | 8/9 | 89% | 35% | 1.13 (2.5-7.0) | 0.84 (2.5-5.5) |
| Qwen3-VL 8B 3-bit | 28 | 9/9 | 100% | 32% | 1.28 (4.0-8.5) | 0.54 (4.0-6.0) |

"Precision" is nominal: an unlabeled model rating is not necessarily a false
positive because the human labels intentionally cover only nine moments.

## Findings

### Gemini has the strongest calibration

Gemini produced the widest visual and action distributions. Its one reported
miss, the George Washington Bridge label, is primarily a temporal-alignment
artifact: it recognized the bridge in two nearby ratings, but both anchors
fell just outside the 15-second matching window.

### GPT finds most labeled moments but under-scores them

GPT missed the labeled climb. On its eight temporal matches, its scores were
lower than the hand labels by an average of 1.62 visual points and 2.00 action
points. Matched ratings averaged 5.6 visual / 3.8 action, while model-only
ratings averaged 5.5 / 3.9. That near-zero separation is the clearest sign that
GPT flattens interesting and ordinary footage toward the middle.

### Local Qwen understands fine clips but does not discover regions

Qwen explicitly recognized the two-second deer and produced valid structured
rubrics for every forced fine-pass region. However, every coarse pass returned
zero hits. Its 28 ratings therefore came from hand-label timestamps, telemetry
peaks, or periodic coverage samples rather than coarse visual discovery.

Qwen also compressed action scores into the narrow range 4-6. It frequently
over-rated ordinary wooded climbs and transitions, called the Hudson a lake,
and described the 9W-sign moment without identifying the sign. Its 100% temporal
recall should not be read as 100% semantic recognition or discovery recall.

## Reproduction

The exact caches and reports behind the table are committed under
`samples/2026-07-10/`. Reproduce each report without video, credentials, or a
running local model:

```bash
MODEL_PROVIDER=gemini ride-recap compare samples/2026-07-10
MODEL_PROVIDER=openai OPENAI_MODEL=gpt-4.1-mini \
  ride-recap compare samples/2026-07-10
MODEL_PROVIDER=local LOCAL_MODEL=mlx-community/Qwen3-VL-8B-Instruct-3bit \
  ride-recap compare samples/2026-07-10
```

Reports use provider/model-specific filenames. With downloaded footage,
comparison uses the MP4 chapters present in the ride directory; in the
video-free committed sample, the labeled chapter names define the same tier.
Raw footage, `.env`, and working outputs under `data/raw/` remain ignored.
