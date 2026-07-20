"""Compare human labels vs vision-model scan output and drive prompt improvement.

Two workflows:

1. **compare**: Load labels + provider cache for a ride, align temporally,
   categorize as matched / label-only (model missed) / model-only.
   Outputs a structured JSON report + human-readable summary.

2. **eval-prompt**: Feed the comparison report to the active MODEL_PROVIDER
   and ask it to analyze systematic patterns and suggest prompt improvements.
   Accumulates across rides for cross-ride analysis.

Usage:
    from prompt_eval import compare_ride, eval_prompt
    report = compare_ride(date_folder)
    suggestions = eval_prompt(date_folder)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .utils import format_time, normalize_label_scale, rating_visual_action

# Temporal alignment window: label and model hit within this many
# seconds are considered a "match"
_MATCH_WINDOW = 15.0

# Legacy filename (implicitly Gemini); only used when MODEL_PROVIDER=gemini
_LEGACY_REPORT_NAME = "prompt_eval.json"

_HIT_FIELD_MAP = {
    "gemini_visual": "model_visual",
    "gemini_action": "model_action",
    "gemini_clip_type": "model_clip_type",
    "gemini_reason": "model_reason",
}

_PATTERN_KEY_MAP = {
    "gemini_misses_by_type": "model_misses_by_type",
    "gemini_overflag_by_clip_type": "model_overflag_by_clip_type",
    "avg_gemini_only_scores": "avg_model_only_scores",
}


def _hit_video_secs(hit: dict) -> float:
    """Where in its clip a hit sits. v10 renamed this to anchor_video_secs."""
    v = hit.get("anchor_video_secs")
    if v is None:
        v = hit.get("video_secs", 0)
    return v or 0.0


def _active_provider(settings) -> str:
    return (settings.model_provider or "gemini").lower()


def _active_model_id(settings) -> str:
    from .models import provider_model_id

    return provider_model_id(settings)


def _sanitize_model_id(model_id: str) -> str:
    """Make a model id safe for use as a filename component.

    Keeps alphanumerics, dots, hyphens, underscores; maps everything else
    to ``-`` and collapses runs so unusual IDs cannot escape the folder
    or collide via path separators / spaces.

    When sanitization changes the id (e.g. ``org/model:v1`` → ``org-model-v1``),
    append a short stable hash of the original so distinct raw ids cannot
    overwrite each other's reports.
    """
    raw = (model_id or "unknown").strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", raw)
    s = re.sub(r"-{2,}", "-", s).strip("-.")
    s = s or "unknown"
    if s != raw:
        fp = hashlib.sha1(raw.encode()).hexdigest()[:8]
        return f"{s}_{fp}"
    return s


def _report_filename(provider: str, model_id: str) -> str:
    return f"prompt_eval_{provider}_{_sanitize_model_id(model_id)}.json"


def _suggestions_filename(provider: str, model_id: str) -> str:
    return f"prompt_eval_suggestions_{provider}_{_sanitize_model_id(model_id)}.json"


def _local_chapter_names(date_folder: Path) -> set[str]:
    """Basenames of locally present GoPro chapter files (no ffprobe)."""
    names = {p.name for p in date_folder.glob("*.MP4")}
    names |= {p.name for p in date_folder.glob("*.mp4")}
    return names


def _filter_hits_to_local_chapters(
    hits: list[dict], local_names: set[str],
) -> list[dict]:
    """Drop hits whose clip_name is not among locally present chapters.

    When *local_names* is empty (sidecar-only), returns *hits* unchanged.
    """
    if not local_names:
        return hits
    return [h for h in hits if h.get("clip_name", "") in local_names]


def _filter_labels_to_local_chapters(
    labels: list[dict], local_names: set[str],
) -> list[dict]:
    """Drop labels whose non-empty clip_name is known absent locally.

    Labels with missing/empty ``clip_name`` are retained (legacy).
    When *local_names* is empty, returns *labels* unchanged.
    """
    if not local_names:
        return labels
    kept = []
    for lab in labels:
        clip = lab.get("clip_name") or ""
        if not clip or clip in local_names:
            kept.append(lab)
    return kept


def _normalize_hit(hit: dict) -> dict:
    """Map legacy gemini_* hit keys to model_*."""
    out = dict(hit)
    for old, new in _HIT_FIELD_MAP.items():
        if new not in out and old in out:
            out[new] = out.pop(old)
        elif old in out:
            out.pop(old)
    return out


def _normalize_patterns(patterns: dict) -> dict:
    if not patterns:
        return {}
    out = {}
    for k, v in patterns.items():
        out[_PATTERN_KEY_MAP.get(k, k)] = v
    return out


def _normalize_report(report: dict) -> dict:
    """Map legacy gemini_* report schema to model_* before aggregation.

    Call on every report read. Downstream only reads model_* keys.
    """
    r = dict(report)
    summary = dict(r.get("summary") or {})
    if "n_model_hits" not in summary and "n_gemini_hits" in summary:
        summary["n_model_hits"] = summary.pop("n_gemini_hits")
    elif "n_gemini_hits" in summary:
        summary.pop("n_gemini_hits")
    if "n_model_only" not in summary and "n_gemini_only" in summary:
        summary["n_model_only"] = summary.pop("n_gemini_only")
    elif "n_gemini_only" in summary:
        summary.pop("n_gemini_only")
    # Legacy unidentified reports are implicitly Gemini
    if "provider" not in summary and "provider" not in r:
        summary.setdefault("provider", "gemini")
    r["summary"] = summary

    if "model_only" not in r and "gemini_only" in r:
        r["model_only"] = r.pop("gemini_only")
    elif "gemini_only" in r:
        r.pop("gemini_only")

    for key in ("matched", "label_only", "model_only"):
        items = r.get(key)
        if isinstance(items, list):
            r[key] = [_normalize_hit(h) if isinstance(h, dict) else h for h in items]

    r["patterns"] = _normalize_patterns(r.get("patterns") or {})
    return r


def _resolve_report_path(date_folder: Path, provider: str, model_id: str) -> Path | None:
    """Return the report path to load for provider/model, or None.

    Prefers the scoped filename. Falls back to legacy ``prompt_eval.json``
    only when *provider* is ``gemini`` (legacy files are implicitly Gemini).
    """
    scoped = date_folder / _report_filename(provider, model_id)
    if scoped.exists():
        return scoped
    if provider == "gemini":
        legacy = date_folder / _LEGACY_REPORT_NAME
        if legacy.exists():
            return legacy
    return None


@dataclass
class CompareHit:
    """A single moment from either labels or the vision model, with match info."""

    ride_time_secs: float
    source: str  # "label", provider id, or "both"
    label_type: str = ""
    label_notes: str = ""
    label_visual: int = 0
    label_action: int = 0
    model_visual: int = 0
    model_action: int = 0
    model_clip_type: str = ""
    model_reason: str = ""
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
        if self.model_visual:
            d["model_visual"] = self.model_visual
        if self.model_action:
            d["model_action"] = self.model_action
        if self.model_clip_type:
            d["model_clip_type"] = self.model_clip_type
        if self.model_reason:
            d["model_reason"] = self.model_reason
        return d


@dataclass
class CompareReport:
    """Structured comparison of labels vs vision-model scan for one ride."""

    date_folder: str
    provider: str = "gemini"
    model: str = ""
    n_labels: int = 0
    n_model_hits: int = 0
    matched: list[CompareHit] = field(default_factory=list)
    label_only: list[CompareHit] = field(default_factory=list)  # model missed
    model_only: list[CompareHit] = field(default_factory=list)  # user didn't flag
    patterns: dict = field(default_factory=dict)  # systematic observations

    @property
    def precision(self) -> float:
        """Of the model's hits, what fraction did the user also flag?"""
        total = len(self.matched) + len(self.model_only)
        return len(self.matched) / total if total else 0.0

    @property
    def recall(self) -> float:
        """Of user's labels, what fraction did the model also find?"""
        total = len(self.matched) + len(self.label_only)
        return len(self.matched) / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "date_folder": self.date_folder,
            "summary": {
                "provider": self.provider,
                "model": self.model,
                "n_labels": self.n_labels,
                "n_model_hits": self.n_model_hits,
                "n_matched": len(self.matched),
                "n_label_only": len(self.label_only),
                "n_model_only": len(self.model_only),
                "precision": round(self.precision, 3),
                "recall": round(self.recall, 3),
            },
            "patterns": self.patterns,
            "matched": [h.to_dict() for h in self.matched],
            "label_only": [h.to_dict() for h in self.label_only],
            "model_only": [h.to_dict() for h in self.model_only],
        }

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        label = f"{self.provider} ({self.model})" if self.model else self.provider
        print(f"\n{'='*60}")
        print(f"Label vs {label} Comparison — {self.date_folder}")
        print(f"{'='*60}")
        print(f"  Labels:       {self.n_labels}")
        print(f"  Model hits:   {self.n_model_hits}")
        print(f"  Matched:      {len(self.matched)}")
        print(f"  Label-only:   {len(self.label_only)}  (model missed)")
        print(f"  Model-only:   {len(self.model_only)}  (user didn't flag)")
        print(f"  Precision:    {self.precision:.0%}")
        print(f"  Recall:       {self.recall:.0%}")

        if self.label_only:
            print("\n  Model MISSED these labeled moments:")
            for h in self.label_only:
                t = format_time(h.ride_time_secs)
                print(f"    {t}  [{h.label_type}] {h.label_notes}")

        if self.model_only:
            print("\n  Model flagged but user DIDN'T label:")
            for h in sorted(
                self.model_only,
                key=lambda x: -max(x.model_visual, x.model_action),
            )[:10]:
                t = format_time(h.ride_time_secs)
                score = f"v={h.model_visual} a={h.model_action}"
                print(f"    {t}  [{h.model_clip_type}] {score} — {h.model_reason}")
            if len(self.model_only) > 10:
                print(f"    ... and {len(self.model_only) - 10} more")

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
    read so the comparison flow operates in a single space. Model hits
    reach the same space via `rating_visual_action`, which folds the v10
    five-dim rubric down to visual/action.
    """
    labels_path = date_folder / "ride_labels.json"
    if not labels_path.exists():
        return []
    return [normalize_label_scale(lab) for lab in json.loads(labels_path.read_text())]


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


def _load_model_hits(
    date_folder: Path,
    chapter_names: set[str] | None = None,
) -> list[dict]:
    """Load vision-scan hits from the active provider's cache directory.

    Uses ``MODEL_PROVIDER`` to pick ``.{provider}_cache/``. Does **not**
    fall back across providers (one-provider rule). Gemini still accepts
    legacy ``gemini_cache.json``. OpenAI files are filtered to the active
    ``OPENAI_MODEL`` fingerprint. Hits are restricted to *chapter_names*
    when supplied; otherwise local ``*.MP4``/``*.mp4`` files define the
    restriction when present.
    """
    from .config import get_settings
    from .models import cache_dir_name

    settings = get_settings()
    provider = _active_provider(settings)
    model_id = _active_model_id(settings)
    cache_dir = date_folder / cache_dir_name(provider)

    if not cache_dir.exists():
        if provider == "gemini":
            legacy = date_folder / "gemini_cache.json"
            if legacy.exists():
                hits = json.loads(legacy.read_text())
                if isinstance(hits, list):
                    names = (
                        chapter_names
                        if chapter_names is not None
                        else _local_chapter_names(date_folder)
                    )
                    return _filter_hits_to_local_chapters(hits, names)
        return []

    all_hits = []
    for cache_file in _cache_files_for_active_model(cache_dir, provider, model_id):
        try:
            hits = json.loads(cache_file.read_text())
            if isinstance(hits, list):
                all_hits.extend(hits)
        except (json.JSONDecodeError, OSError):
            continue
    names = (
        chapter_names
        if chapter_names is not None
        else _local_chapter_names(date_folder)
    )
    return _filter_hits_to_local_chapters(all_hits, names)


# Thin alias for callers/tests that still use the old name
_load_gemini_hits = _load_model_hits


def _align_moments(
    labels: list[dict],
    model_hits: list[dict],
    *,
    provider: str,
    window: float = _MATCH_WINDOW,
) -> tuple[list[CompareHit], list[CompareHit], list[CompareHit]]:
    """Align labels and model hits by ride time.

    Returns (matched, label_only, model_only). Model-only hits use
    *provider* as ``CompareHit.source``.
    """
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

    # Model hits use video_secs within each clip — map to ride time when
    # ride_time_secs is present; otherwise approximate from video_secs.
    model_moments = []
    for hit in model_hits:
        video_secs = _hit_video_secs(hit)
        visual, action = rating_visual_action(hit)
        model_moments.append({
            "ride_time_secs": hit.get("ride_time_secs", video_secs),
            "visual": visual,
            "action": action,
            "clip_type": hit.get("clip_type", ""),
            "reason": hit.get("reason", ""),
            "clip_name": hit.get("clip_name", ""),
            "video_secs": video_secs,
        })

    label_moments.sort(key=lambda x: x["ride_time_secs"])
    model_moments.sort(key=lambda x: x["ride_time_secs"])

    matched_model = set()
    matched = []
    label_only = []

    for lab in label_moments:
        best_idx = None
        best_dist = window + 1
        for i, mom in enumerate(model_moments):
            if i in matched_model:
                continue
            dist = abs(lab["ride_time_secs"] - mom["ride_time_secs"])
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is not None and best_dist <= window:
            mom = model_moments[best_idx]
            matched_model.add(best_idx)
            matched.append(CompareHit(
                ride_time_secs=lab["ride_time_secs"],
                source="both",
                label_type=lab["type"],
                label_notes=lab["notes"],
                label_visual=lab["visual"],
                label_action=lab["action"],
                model_visual=mom["visual"],
                model_action=mom["action"],
                model_clip_type=mom["clip_type"],
                model_reason=mom["reason"],
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

    model_only = []
    for i, mom in enumerate(model_moments):
        if i not in matched_model:
            model_only.append(CompareHit(
                ride_time_secs=mom["ride_time_secs"],
                source=provider,
                model_visual=mom["visual"],
                model_action=mom["action"],
                model_clip_type=mom["clip_type"],
                model_reason=mom["reason"],
                clip_name=mom["clip_name"],
                video_secs=mom["video_secs"],
            ))

    return matched, label_only, model_only


def _analyze_patterns(
    matched: list[CompareHit],
    label_only: list[CompareHit],
    model_only: list[CompareHit],
) -> dict:
    """Identify systematic patterns in the comparison."""
    patterns = {}

    if label_only:
        missed_types: dict[str, int] = {}
        for h in label_only:
            t = h.label_type or "unknown"
            missed_types[t] = missed_types.get(t, 0) + 1
        patterns["model_misses_by_type"] = missed_types

    if model_only:
        over_types: dict[str, int] = {}
        for h in model_only:
            t = h.model_clip_type or "unknown"
            over_types[t] = over_types.get(t, 0) + 1
        patterns["model_overflag_by_clip_type"] = over_types

    scored_matches = [h for h in matched if h.label_visual and h.model_visual]
    if scored_matches:
        visual_diffs = [h.model_visual - h.label_visual for h in scored_matches]
        action_diffs = [h.model_action - h.label_action for h in scored_matches]
        patterns["score_agreement"] = {
            "n": len(scored_matches),
            "visual_mean_diff": round(sum(visual_diffs) / len(visual_diffs), 2),
            "action_mean_diff": round(sum(action_diffs) / len(action_diffs), 2),
        }
        type_matches = sum(1 for h in scored_matches if h.label_type == h.model_clip_type)
        patterns["type_agreement"] = {
            "matching": type_matches,
            "total": len(scored_matches),
            "rate": round(type_matches / len(scored_matches), 2),
        }

    if matched:
        avg_matched_v = sum(h.model_visual for h in matched) / len(matched)
        avg_matched_a = sum(h.model_action for h in matched) / len(matched)
        patterns["avg_matched_scores"] = {
            "visual": round(avg_matched_v, 1),
            "action": round(avg_matched_a, 1),
        }
    if model_only:
        avg_go_v = sum(h.model_visual for h in model_only) / len(model_only)
        avg_go_a = sum(h.model_action for h in model_only) / len(model_only)
        patterns["avg_model_only_scores"] = {
            "visual": round(avg_go_v, 1),
            "action": round(avg_go_a, 1),
        }

    if matched:
        min_matched_score = min(max(h.model_visual, h.model_action) for h in matched)
        patterns["min_matched_max_score"] = min_matched_score

    return patterns


def compare_ride(date_folder: Path) -> CompareReport:
    """Compare human labels vs the active vision-model scan for a single ride.

    Loads labels from ride_labels.json and hits from ``.{provider}_cache/``.
    Writes a provider/model-scoped ``prompt_eval_*.json`` report.
    """
    from .config import get_settings

    date_folder = Path(date_folder)
    settings = get_settings()
    provider = _active_provider(settings)
    model_id = _active_model_id(settings)

    all_labels = _load_labels(date_folder)
    local_names = _local_chapter_names(date_folder)
    # The committed sample contains no MP4s. In that case its labeled chapter
    # names define the comparison tier, preventing the 19-chapter Gemini
    # baseline from being compared against eight chapters of alternatives.
    comparison_names = local_names or {
        str(label.get("clip_name", ""))
        for label in all_labels
        if label.get("clip_name")
    }
    labels = _filter_labels_to_local_chapters(all_labels, comparison_names)
    model_hits = _load_model_hits(date_folder, comparison_names)

    if not labels:
        print(f"No labels found in {date_folder}")
    if not model_hits:
        print(f"No {provider} scan results found in {date_folder}")

    matched, label_only, model_only = _align_moments(
        labels, model_hits, provider=provider,
    )
    patterns = _analyze_patterns(matched, label_only, model_only)

    report = CompareReport(
        date_folder=str(date_folder),
        provider=provider,
        model=model_id,
        n_labels=len(labels),
        n_model_hits=len(model_hits),
        matched=matched,
        label_only=label_only,
        model_only=model_only,
        patterns=patterns,
    )

    report_path = date_folder / _report_filename(provider, model_id)
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"Saved comparison report to {report_path}")

    return report


# ═══════════════════════════════════════════════════════════════
# Model-assisted prompt improvement
# ═══════════════════════════════════════════════════════════════

_EVAL_SYSTEM = """You are an expert at tuning prompts for vision-language models used in video editing.

You are reviewing a vision-scan prompt that scores handlebar-cam cycling video frames for a highlight reel.
The user has manually labeled moments they found interesting, and we're comparing those labels
against what the configured model scan found.

Your job: analyze the comparison data and suggest SPECIFIC, ACTIONABLE prompt improvements.

Focus on:
1. What types of moments does the current prompt miss? (false negatives — user labeled but model didn't flag)
2. What does the prompt over-flag? (false positives — model flagged but user didn't care about)
3. Are there scoring calibration issues? (model scores too high/low for certain types)
4. Are there missing visual cues the prompt should mention?

Be concrete. Don't say "improve the prompt" — say exactly what words to add, remove, or change."""

_EVAL_TEMPLATE = """Here is the CURRENT vision-scan system instruction used for scoring:

```
{current_prompt}
```

Here is the comparison report for {n_rides} ride(s) (provider={provider}, model={model}):

## Summary
- Total labels: {n_labels}
- Total model hits: {n_model_hits}
- Matched (both agree): {n_matched}
- Label-only (model missed): {n_label_only}
- Model-only (user didn't flag): {n_model_only}
- Precision: {precision:.0%}
- Recall: {recall:.0%}

## Moments the model MISSED (user labeled, model didn't flag)
{missed_details}

## Moments the model OVER-FLAGGED (model flagged, user didn't label)
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
    """Load comparison reports for the active provider/model.

    Prefers scoped filenames. Legacy ``prompt_eval.json`` is only used when
    the active provider is Gemini. Every loaded report is normalized to the
    model_* schema.
    """
    from .config import get_settings

    settings = get_settings()
    provider = _active_provider(settings)
    model_id = _active_model_id(settings)

    reports = []
    for folder in date_folders:
        report_path = _resolve_report_path(Path(folder), provider, model_id)
        if report_path is None:
            continue
        try:
            raw = json.loads(report_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(raw, dict):
            continue
        # Reject scoped/legacy files whose embedded metadata mismatches
        summary = raw.get("summary") or {}
        file_provider = (summary.get("provider") or raw.get("provider") or "").lower()
        if file_provider and file_provider != provider:
            continue
        file_model = summary.get("model") or raw.get("model") or ""
        # Reject any populated mismatch — including legacy prompt_eval.json
        if file_model and file_model != model_id:
            continue
        reports.append(_normalize_report(raw))
    return reports


def _aggregate_reports(reports: list[dict]) -> dict:
    """Aggregate multiple ride comparison reports into one summary.

    Expects reports already normalized to model_* keys.
    """
    agg = {
        "n_rides": len(reports),
        "n_labels": sum(r["summary"]["n_labels"] for r in reports),
        "n_model_hits": sum(r["summary"]["n_model_hits"] for r in reports),
        "n_matched": sum(r["summary"]["n_matched"] for r in reports),
        "n_label_only": sum(r["summary"]["n_label_only"] for r in reports),
        "n_model_only": sum(r["summary"]["n_model_only"] for r in reports),
        "all_label_only": [],
        "all_model_only": [],
        "all_patterns": [],
    }
    for r in reports:
        agg["all_label_only"].extend(r.get("label_only", []))
        # Limit model-only to top-scoring to keep prompt manageable
        model_only = list(r.get("model_only", []))
        model_only.sort(
            key=lambda x: -max(x.get("model_visual", 0), x.get("model_action", 0)),
        )
        agg["all_model_only"].extend(model_only[:15])
        agg["all_patterns"].append(r.get("patterns", {}))

    total = agg["n_matched"] + agg["n_model_only"]
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
    """Format model-only items for the eval prompt."""
    if not items:
        return "(none)"
    lines = []
    for item in items[:20]:  # cap for prompt length
        t = item.get("ride_time_str", format_time(item.get("ride_time_secs", 0)))
        v = item.get("model_visual", 0)
        a = item.get("model_action", 0)
        ct = item.get("model_clip_type", "")
        reason = item.get("model_reason", "")
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
        date_folders: List of ride date folders with comparison reports
        current_prompt: The current fine-pass system instruction. If None,
            loads from gemini_scan._SYSTEM_INSTRUCTION (v10 placeholders).

    Returns:
        Dict with analysis and suggested changes.
    """
    from .config import get_settings
    from .gemini_scan import _SYSTEM_INSTRUCTION, _build_context_strings
    from .models import get_model_adapter, provider_api_key

    settings = get_settings()
    provider = _active_provider(settings)
    model_id = _active_model_id(settings)
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
        provider=provider,
        model=model_id,
        n_rides=agg["n_rides"],
        n_labels=agg["n_labels"],
        n_model_hits=agg["n_model_hits"],
        n_matched=agg["n_matched"],
        n_label_only=agg["n_label_only"],
        n_model_only=agg["n_model_only"],
        precision=agg["precision"],
        recall=agg["recall"],
        missed_details=_format_missed(agg["all_label_only"]),
        overflag_details=_format_overflag(agg["all_model_only"]),
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

    for folder in date_folders:
        eval_path = Path(folder) / _suggestions_filename(provider, model_id)
        eval_path.write_text(json.dumps(result, indent=2))
        print(f"Saved prompt improvement suggestions to {eval_path}")
        break  # save to first folder only

    return result


def enrich_model_hits_with_ride_time(
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

    fit_files = list(date_folder.glob("*.fit")) + list(date_folder.glob("*.FIT"))
    if not fit_files:
        print("No FIT file found")
        return

    ride = parse_fit(fit_files[0])
    clips = extract_all(date_folder)
    synced_clips = sync_all(clips, ride, offset)
    sc_by_name = {sc.clip.path.name: sc for sc in synced_clips}

    settings = get_settings()
    provider = _active_provider(settings)
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


# Thin alias for callers that still use the old name
enrich_gemini_hits_with_ride_time = enrich_model_hits_with_ride_time
