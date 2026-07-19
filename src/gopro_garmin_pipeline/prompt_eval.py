"""Compare human labels vs Gemini scan output and drive prompt improvement.

Two workflows:

1. **compare**: Load labels + Gemini cache for a ride, align temporally,
   categorize as matched / label-only (Gemini missed) / gemini-only.
   Outputs a structured JSON report + human-readable summary.

2. **eval-prompt**: Feed the comparison report to Gemini and ask it to
   analyze systematic patterns and suggest prompt improvements.
   Accumulates across rides for cross-ride analysis.

Usage:
    from prompt_eval import compare_ride, eval_prompt
    report = compare_ride(date_folder)
    suggestions = eval_prompt(date_folder)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .utils import format_time, normalize_label_scale, rating_visual_action

# Temporal alignment window: label and Gemini hit within this many
# seconds are considered a "match"
_MATCH_WINDOW = 15.0

# Report filename
_REPORT_NAME = "prompt_eval.json"


def _hit_video_secs(hit: dict) -> float:
    """Where in its clip a hit sits. v10 renamed this to anchor_video_secs."""
    v = hit.get("anchor_video_secs")
    if v is None:
        v = hit.get("video_secs", 0)
    return v or 0.0


@dataclass
class CompareHit:
    """A single moment from either labels or Gemini, with match info."""

    ride_time_secs: float
    source: str  # "label", "gemini", or "both"
    label_type: str = ""
    label_notes: str = ""
    label_visual: int = 0
    label_action: int = 0
    gemini_visual: int = 0
    gemini_action: int = 0
    gemini_clip_type: str = ""
    gemini_reason: str = ""
    clip_name: str = ""
    video_secs: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "ride_time_secs": self.ride_time_secs,
            "ride_time_str": format_time(self.ride_time_secs),
            "source": self.source,
            "clip_name": self.clip_name,
            "video_secs": round(self.video_secs, 1),
        }
        if self.label_type:
            d["label_type"] = self.label_type
        if self.label_visual:
            d["label_visual"] = self.label_visual
        if self.label_action:
            d["label_action"] = self.label_action
        if self.label_notes:
            d["label_notes"] = self.label_notes
        if self.gemini_visual:
            d["gemini_visual"] = self.gemini_visual
        if self.gemini_action:
            d["gemini_action"] = self.gemini_action
        if self.gemini_clip_type:
            d["gemini_clip_type"] = self.gemini_clip_type
        if self.gemini_reason:
            d["gemini_reason"] = self.gemini_reason
        return d


@dataclass
class CompareReport:
    """Structured comparison of labels vs Gemini scan for one ride."""

    date_folder: str
    n_labels: int = 0
    n_gemini_hits: int = 0
    matched: list[CompareHit] = field(default_factory=list)
    label_only: list[CompareHit] = field(default_factory=list)  # Gemini missed
    gemini_only: list[CompareHit] = field(default_factory=list)  # user didn't flag
    patterns: dict = field(default_factory=dict)  # systematic observations

    @property
    def precision(self) -> float:
        """Of Gemini's hits, what fraction did the user also flag?"""
        total = len(self.matched) + len(self.gemini_only)
        return len(self.matched) / total if total else 0.0

    @property
    def recall(self) -> float:
        """Of user's labels, what fraction did Gemini also find?"""
        total = len(self.matched) + len(self.label_only)
        return len(self.matched) / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "date_folder": self.date_folder,
            "summary": {
                "n_labels": self.n_labels,
                "n_gemini_hits": self.n_gemini_hits,
                "n_matched": len(self.matched),
                "n_label_only": len(self.label_only),
                "n_gemini_only": len(self.gemini_only),
                "precision": round(self.precision, 3),
                "recall": round(self.recall, 3),
            },
            "patterns": self.patterns,
            "matched": [h.to_dict() for h in self.matched],
            "label_only": [h.to_dict() for h in self.label_only],
            "gemini_only": [h.to_dict() for h in self.gemini_only],
        }

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        print(f"\n{'='*60}")
        print(f"Label vs Gemini Comparison — {self.date_folder}")
        print(f"{'='*60}")
        print(f"  Labels:       {self.n_labels}")
        print(f"  Gemini hits:  {self.n_gemini_hits}")
        print(f"  Matched:      {len(self.matched)}")
        print(f"  Label-only:   {len(self.label_only)}  (Gemini missed)")
        print(f"  Gemini-only:  {len(self.gemini_only)}  (user didn't flag)")
        print(f"  Precision:    {self.precision:.0%}")
        print(f"  Recall:       {self.recall:.0%}")

        if self.label_only:
            print("\n  Gemini MISSED these labeled moments:")
            for h in self.label_only:
                t = format_time(h.ride_time_secs)
                print(f"    {t}  [{h.label_type}] {h.label_notes}")

        if self.gemini_only:
            print("\n  Gemini flagged but user DIDN'T label:")
            for h in sorted(self.gemini_only, key=lambda x: -max(x.gemini_visual, x.gemini_action))[:10]:
                t = format_time(h.ride_time_secs)
                score = f"v={h.gemini_visual} a={h.gemini_action}"
                print(f"    {t}  [{h.gemini_clip_type}] {score} — {h.gemini_reason}")
            if len(self.gemini_only) > 10:
                print(f"    ... and {len(self.gemini_only) - 10} more")

        if self.patterns:
            print("\n  Patterns:")
            for k, v in self.patterns.items():
                print(f"    {k}: {v}")


# ═══════════════════════════════════════════════════════════════
# Core comparison
# ═══════════════════════════════════════════════════════════════

def _load_labels(date_folder: Path) -> list[dict]:
    """Load ride_labels.json from a date folder.

    Old labels saved on the 1-5 scale are upgraded to the 1-10 scale on
    read so the comparison flow operates in a single space. Gemini hits
    reach the same space via `rating_visual_action`, which folds the v10
    five-dim rubric down to visual/action.
    """
    labels_path = date_folder / "ride_labels.json"
    if not labels_path.exists():
        return []
    return [normalize_label_scale(lab) for lab in json.loads(labels_path.read_text())]


def _active_model_id(settings) -> str:
    provider = (settings.model_provider or "gemini").lower()
    if provider == "openai":
        return settings.openai_model
    return settings.gemini_model


def _cache_files_for_active_model(cache_dir: Path, provider: str, model_id: str) -> list[Path]:
    """List cache JSON files for the active provider/model only.

    Gemini: all ``*.json`` (legacy filenames have no model fingerprint).
    OpenAI: only files whose name contains ``_m{fingerprint}`` for the
    active ``OPENAI_MODEL`` — never merge results across models.
    """
    if not cache_dir.exists():
        return []
    files = sorted(cache_dir.glob("*.json"))
    if provider == "gemini":
        return files

    from .gemini_scan import _model_fingerprint

    fp = _model_fingerprint(model_id)
    matched: list[Path] = []
    for path in files:
        stem = path.stem
        # Keys: {stem}_{ver}_m{fp}.json or {stem}_{ver}_m{fp}_l{label}.json
        if stem.endswith(f"_m{fp}") or f"_m{fp}_l" in stem:
            matched.append(path)
    return matched


def _load_gemini_hits(date_folder: Path) -> list[dict]:
    """Load vision-scan hits from the active provider's cache directory.

    Uses ``MODEL_PROVIDER`` to pick ``.{provider}_cache/``. Does **not**
    fall back across providers (one-provider rule). Gemini still accepts
    legacy ``gemini_cache.json``. OpenAI files are filtered to the active
    ``OPENAI_MODEL`` fingerprint.
    """
    from .config import get_settings
    from .models import cache_dir_name

    settings = get_settings()
    provider = (settings.model_provider or "gemini").lower()
    model_id = _active_model_id(settings)
    cache_dir = date_folder / cache_dir_name(provider)

    if not cache_dir.exists():
        if provider == "gemini":
            legacy = date_folder / "gemini_cache.json"
            if legacy.exists():
                return json.loads(legacy.read_text())
        return []

    all_hits = []
    for cache_file in _cache_files_for_active_model(cache_dir, provider, model_id):
        try:
            hits = json.loads(cache_file.read_text())
            if isinstance(hits, list):
                all_hits.extend(hits)
        except (json.JSONDecodeError, OSError):
            continue
    return all_hits


def _align_moments(
    labels: list[dict], gemini_hits: list[dict], window: float = _MATCH_WINDOW,
) -> tuple[list[CompareHit], list[CompareHit], list[CompareHit]]:
    """Align labels and Gemini hits by ride time.

    Returns (matched, label_only, gemini_only).
    """
    # Build label list with ride_time_secs
    label_moments = []
    for lab in labels:
        ride_secs = lab.get("ride_time_secs", 0)
        label_moments.append({
            "ride_time_secs": ride_secs,
            "type": lab.get("type", ""),
            "visual": lab.get("visual", 0),
            "action": lab.get("action", 0),
            "notes": lab.get("notes", ""),
            "clip_name": lab.get("clip_name", ""),
            "video_secs": lab.get("video_secs", 0),
        })

    # Build Gemini list with ride_time_secs
    # Gemini hits use video_secs within each clip — we need to map to
    # ride time. We store clip_name + video_secs as the key, but for
    # comparison we need ride-relative time. If the Gemini cache
    # doesn't have ride_time_secs, we approximate from video_secs.
    gemini_moments = []
    for hit in gemini_hits:
        video_secs = _hit_video_secs(hit)
        visual, action = rating_visual_action(hit)
        gemini_moments.append({
            "ride_time_secs": hit.get("ride_time_secs", video_secs),
            "visual": visual,
            "action": action,
            "clip_type": hit.get("clip_type", ""),
            "reason": hit.get("reason", ""),
            "clip_name": hit.get("clip_name", ""),
            "video_secs": video_secs,
        })

    # Sort both by ride time
    label_moments.sort(key=lambda x: x["ride_time_secs"])
    gemini_moments.sort(key=lambda x: x["ride_time_secs"])

    # Greedy matching: for each label, find closest unmatched Gemini hit
    matched_gemini = set()
    matched = []
    label_only = []

    for lab in label_moments:
        best_idx = None
        best_dist = window + 1
        for i, gem in enumerate(gemini_moments):
            if i in matched_gemini:
                continue
            dist = abs(lab["ride_time_secs"] - gem["ride_time_secs"])
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is not None and best_dist <= window:
            gem = gemini_moments[best_idx]
            matched_gemini.add(best_idx)
            matched.append(CompareHit(
                ride_time_secs=lab["ride_time_secs"],
                source="both",
                label_type=lab["type"],
                label_notes=lab["notes"],
                label_visual=lab["visual"],
                label_action=lab["action"],
                gemini_visual=gem["visual"],
                gemini_action=gem["action"],
                gemini_clip_type=gem["clip_type"],
                gemini_reason=gem["reason"],
                clip_name=lab["clip_name"],
                video_secs=lab["video_secs"],
            ))
        else:
            label_only.append(CompareHit(
                ride_time_secs=lab["ride_time_secs"],
                source="label",
                label_type=lab["type"],
                label_notes=lab["notes"],
                label_visual=lab["visual"],
                label_action=lab["action"],
                clip_name=lab["clip_name"],
                video_secs=lab["video_secs"],
            ))

    # Remaining Gemini hits = gemini-only
    gemini_only = []
    for i, gem in enumerate(gemini_moments):
        if i not in matched_gemini:
            gemini_only.append(CompareHit(
                ride_time_secs=gem["ride_time_secs"],
                source="gemini",
                gemini_visual=gem["visual"],
                gemini_action=gem["action"],
                gemini_clip_type=gem["clip_type"],
                gemini_reason=gem["reason"],
                clip_name=gem["clip_name"],
                video_secs=gem["video_secs"],
            ))

    return matched, label_only, gemini_only


def _analyze_patterns(
    matched: list[CompareHit],
    label_only: list[CompareHit],
    gemini_only: list[CompareHit],
) -> dict:
    """Identify systematic patterns in the comparison."""
    patterns = {}

    # What types does Gemini miss?
    if label_only:
        missed_types: dict[str, int] = {}
        for h in label_only:
            t = h.label_type or "unknown"
            missed_types[t] = missed_types.get(t, 0) + 1
        patterns["gemini_misses_by_type"] = missed_types

    # What clip_types does Gemini over-flag?
    if gemini_only:
        over_types: dict[str, int] = {}
        for h in gemini_only:
            t = h.gemini_clip_type or "unknown"
            over_types[t] = over_types.get(t, 0) + 1
        patterns["gemini_overflag_by_clip_type"] = over_types

    # Score agreement on matched moments (label vs Gemini visual/action)
    scored_matches = [h for h in matched if h.label_visual and h.gemini_visual]
    if scored_matches:
        visual_diffs = [h.gemini_visual - h.label_visual for h in scored_matches]
        action_diffs = [h.gemini_action - h.label_action for h in scored_matches]
        patterns["score_agreement"] = {
            "n": len(scored_matches),
            "visual_mean_diff": round(sum(visual_diffs) / len(visual_diffs), 2),
            "action_mean_diff": round(sum(action_diffs) / len(action_diffs), 2),
        }
        # Type agreement
        type_matches = sum(1 for h in scored_matches if h.label_type == h.gemini_clip_type)
        patterns["type_agreement"] = {
            "matching": type_matches,
            "total": len(scored_matches),
            "rate": round(type_matches / len(scored_matches), 2),
        }

    # Average Gemini scores for matched vs gemini-only
    if matched:
        avg_matched_v = sum(h.gemini_visual for h in matched) / len(matched)
        avg_matched_a = sum(h.gemini_action for h in matched) / len(matched)
        patterns["avg_matched_scores"] = {
            "visual": round(avg_matched_v, 1),
            "action": round(avg_matched_a, 1),
        }
    if gemini_only:
        avg_go_v = sum(h.gemini_visual for h in gemini_only) / len(gemini_only)
        avg_go_a = sum(h.gemini_action for h in gemini_only) / len(gemini_only)
        patterns["avg_gemini_only_scores"] = {
            "visual": round(avg_go_v, 1),
            "action": round(avg_go_a, 1),
        }

    # Score threshold analysis: what's the min score in matched?
    if matched:
        min_matched_score = min(max(h.gemini_visual, h.gemini_action) for h in matched)
        patterns["min_matched_max_score"] = min_matched_score

    return patterns


def compare_ride(date_folder: Path) -> CompareReport:
    """Compare human labels vs Gemini scan for a single ride.

    Loads labels from ride_labels.json and Gemini hits from .gemini_cache/.
    Returns a structured CompareReport.
    """
    date_folder = Path(date_folder)

    labels = _load_labels(date_folder)
    gemini_hits = _load_gemini_hits(date_folder)

    if not labels:
        print(f"No labels found in {date_folder}")
    if not gemini_hits:
        print(f"No Gemini scan results found in {date_folder}")

    matched, label_only, gemini_only = _align_moments(labels, gemini_hits)
    patterns = _analyze_patterns(matched, label_only, gemini_only)

    report = CompareReport(
        date_folder=str(date_folder),
        n_labels=len(labels),
        n_gemini_hits=len(gemini_hits),
        matched=matched,
        label_only=label_only,
        gemini_only=gemini_only,
        patterns=patterns,
    )

    # Save report
    report_path = date_folder / _REPORT_NAME
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"Saved comparison report to {report_path}")

    return report


# ═══════════════════════════════════════════════════════════════
# Gemini-assisted prompt improvement
# ═══════════════════════════════════════════════════════════════

_EVAL_SYSTEM = """You are an expert at tuning prompts for vision-language models used in video editing.

You are reviewing a Gemini prompt that scores handlebar-cam cycling video frames for a highlight reel.
The user has manually labeled moments they found interesting, and we're comparing those labels
against what the Gemini scan found.

Your job: analyze the comparison data and suggest SPECIFIC, ACTIONABLE prompt improvements.

Focus on:
1. What types of moments does the current prompt miss? (false negatives — user labeled but Gemini didn't flag)
2. What does the prompt over-flag? (false positives — Gemini flagged but user didn't care about)
3. Are there scoring calibration issues? (Gemini scores too high/low for certain types)
4. Are there missing visual cues the prompt should mention?

Be concrete. Don't say "improve the prompt" — say exactly what words to add, remove, or change."""

_EVAL_TEMPLATE = """Here is the CURRENT Gemini system instruction used for scoring:

```
{current_prompt}
```

Here is the comparison report for {n_rides} ride(s):

## Summary
- Total labels: {n_labels}
- Total Gemini hits: {n_gemini_hits}
- Matched (both agree): {n_matched}
- Label-only (Gemini missed): {n_label_only}
- Gemini-only (user didn't flag): {n_gemini_only}
- Precision: {precision:.0%}
- Recall: {recall:.0%}

## Moments Gemini MISSED (user labeled, Gemini didn't flag)
{missed_details}

## Moments Gemini OVER-FLAGGED (Gemini flagged, user didn't label)
{overflag_details}

## Pattern Analysis
{pattern_details}

---

Based on this data, suggest specific prompt changes. Return a JSON object:
{{
  "analysis": "2-3 sentence summary of the core issue",
  "prompt_changes": [
    {{
      "type": "add" | "remove" | "modify",
      "section": "which part of the prompt",
      "current_text": "existing text (for modify/remove)",
      "suggested_text": "new text (for add/modify)",
      "rationale": "why this change helps"
    }}
  ],
  "suggested_score_adjustments": {{
    "description": "any scoring formula changes"
  }}
}}"""


def _load_all_reports(date_folders: list[Path]) -> list[dict]:
    """Load prompt_eval.json from multiple ride folders."""
    reports = []
    for folder in date_folders:
        report_path = Path(folder) / _REPORT_NAME
        if report_path.exists():
            try:
                reports.append(json.loads(report_path.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
    return reports


def _aggregate_reports(reports: list[dict]) -> dict:
    """Aggregate multiple ride comparison reports into one summary."""
    agg = {
        "n_rides": len(reports),
        "n_labels": sum(r["summary"]["n_labels"] for r in reports),
        "n_gemini_hits": sum(r["summary"]["n_gemini_hits"] for r in reports),
        "n_matched": sum(r["summary"]["n_matched"] for r in reports),
        "n_label_only": sum(r["summary"]["n_label_only"] for r in reports),
        "n_gemini_only": sum(r["summary"]["n_gemini_only"] for r in reports),
        "all_label_only": [],
        "all_gemini_only": [],
        "all_patterns": [],
    }
    for r in reports:
        agg["all_label_only"].extend(r.get("label_only", []))
        # Limit gemini-only to top-scoring to keep prompt manageable
        gemini_only = r.get("gemini_only", [])
        gemini_only.sort(key=lambda x: -max(x.get("gemini_visual", 0), x.get("gemini_action", 0)))
        agg["all_gemini_only"].extend(gemini_only[:15])
        agg["all_patterns"].append(r.get("patterns", {}))

    total = agg["n_matched"] + agg["n_gemini_only"]
    agg["precision"] = agg["n_matched"] / total if total else 0.0
    total = agg["n_matched"] + agg["n_label_only"]
    agg["recall"] = agg["n_matched"] / total if total else 0.0

    return agg


def _format_missed(items: list[dict]) -> str:
    """Format label-only items for the eval prompt."""
    if not items:
        return "(none)"
    lines = []
    for item in items:
        t = item.get("ride_time_str", format_time(item.get("ride_time_secs", 0)))
        label_type = item.get("label_type", "")
        notes = item.get("label_notes", "")
        lines.append(f"  {t} [{label_type}] {notes}")
    return "\n".join(lines)


def _format_overflag(items: list[dict]) -> str:
    """Format gemini-only items for the eval prompt."""
    if not items:
        return "(none)"
    lines = []
    for item in items[:20]:  # cap for prompt length
        t = item.get("ride_time_str", format_time(item.get("ride_time_secs", 0)))
        v = item.get("gemini_visual", 0)
        a = item.get("gemini_action", 0)
        ct = item.get("gemini_clip_type", "")
        reason = item.get("gemini_reason", "")
        lines.append(f"  {t} [{ct}] v={v} a={a} — {reason}")
    if len(items) > 20:
        lines.append(f"  ... and {len(items) - 20} more")
    return "\n".join(lines)


def _format_patterns(patterns_list: list[dict]) -> str:
    """Format pattern analysis for the eval prompt."""
    if not patterns_list:
        return "(none)"
    lines = []
    for p in patterns_list:
        for k, v in p.items():
            lines.append(f"  {k}: {json.dumps(v)}")
    return "\n".join(lines)


def eval_prompt(
    date_folders: list[Path],
    current_prompt: str | None = None,
) -> dict:
    """Analyze label/scan comparison and suggest prompt improvements.

    Uses the same MODEL_PROVIDER as the vision scan — never a second provider.

    Args:
        date_folders: List of ride date folders with prompt_eval.json files
        current_prompt: The current fine-pass system instruction. If None,
            loads from gemini_scan._SYSTEM_INSTRUCTION (v10 placeholders).

    Returns:
        Dict with analysis and suggested changes.
    """
    from .config import get_settings
    from .gemini_scan import _SYSTEM_INSTRUCTION, _build_context_strings
    from .models import get_model_adapter, provider_api_key

    settings = get_settings()
    if not provider_api_key(settings):
        print("No API key for MODEL_PROVIDER; cannot run prompt eval.")
        return {}

    if current_prompt is None:
        power_zones, telemetry_fields, telemetry_examples = _build_context_strings(
            True, settings.ftp,
        )
        current_prompt = _SYSTEM_INSTRUCTION.format(
            n_frames=6,
            power_zones=power_zones,
            telemetry_fields=telemetry_fields,
            telemetry_examples=telemetry_examples,
        )

    reports = _load_all_reports(date_folders)
    if not reports:
        print("No comparison reports found. Run 'gopro-garmin compare' first.")
        return {}

    agg = _aggregate_reports(reports)

    prompt_text = _EVAL_TEMPLATE.format(
        current_prompt=current_prompt,
        n_rides=agg["n_rides"],
        n_labels=agg["n_labels"],
        n_gemini_hits=agg["n_gemini_hits"],
        n_matched=agg["n_matched"],
        n_label_only=agg["n_label_only"],
        n_gemini_only=agg["n_gemini_only"],
        precision=agg["precision"],
        recall=agg["recall"],
        missed_details=_format_missed(agg["all_label_only"]),
        overflag_details=_format_overflag(agg["all_gemini_only"]),
        pattern_details=_format_patterns(agg["all_patterns"]),
    )

    adapter = get_model_adapter(settings)
    result = adapter.complete_json(
        prompt=prompt_text,
        system=_EVAL_SYSTEM,
        temperature=0.3,
        max_output_tokens=2048,
    )
    if not isinstance(result, dict):
        result = {"raw_response": result}

    # Save eval result alongside reports
    for folder in date_folders:
        eval_path = Path(folder) / "prompt_eval_suggestions.json"
        eval_path.write_text(json.dumps(result, indent=2))
        print(f"Saved prompt improvement suggestions to {eval_path}")
        break  # save to first folder only

    return result


def enrich_gemini_hits_with_ride_time(
    date_folder: Path, offset: float = 0.0,
) -> None:
    """Add ride_time_secs to cached vision hits so comparison can align them.

    Cache stores video_secs per clip. This function maps each hit
    to ride-relative time using FIT + GoPro sync, then re-saves the cache.
    Uses the active MODEL_PROVIDER cache directory.
    """
    import datetime as dt
    from .config import get_settings
    from .fit_parser import parse_fit
    from .gopro_meta import extract_all
    from .models import cache_dir_name
    from .sync import normalize_tz, sync_all

    date_folder = Path(date_folder)

    # Find FIT + video files
    fit_files = list(date_folder.glob("*.fit")) + list(date_folder.glob("*.FIT"))
    if not fit_files:
        print("No FIT file found")
        return

    ride = parse_fit(fit_files[0])
    clips = extract_all(date_folder)
    synced_clips = sync_all(clips, ride, offset)
    sc_by_name = {sc.clip.path.name: sc for sc in synced_clips}

    settings = get_settings()
    provider = (settings.model_provider or "gemini").lower()
    model_id = _active_model_id(settings)
    cache_dir = date_folder / cache_dir_name(provider)
    if not cache_dir.exists():
        return

    for cache_file in _cache_files_for_active_model(cache_dir, provider, model_id):
        try:
            hits = json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(hits, list):
            continue

        changed = False
        for hit in hits:
            if "ride_time_secs" in hit:
                continue
            clip_name = hit.get("clip_name", "")
            video_secs = _hit_video_secs(hit)
            sc = sc_by_name.get(clip_name)
            if sc is None:
                continue

            wall = sc.clip.creation_time + dt.timedelta(seconds=video_secs)
            if ride.start_time:
                wall = normalize_tz(wall, ride.start_time)
                ride_secs = (wall - ride.start_time).total_seconds() + sc.offset_secs
                hit["ride_time_secs"] = round(ride_secs, 1)
                changed = True

        if changed:
            cache_file.write_text(json.dumps(hits, indent=2))
            print(f"  Enriched {cache_file.name} with ride_time_secs")
