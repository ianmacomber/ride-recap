"""Learn candidate ranking weights from accumulated ride selections.

Collects training data during every compose run: for each candidate in the pool,
records features + whether it was selected. Once enough data accumulates (~100+
examples across 5+ rides), trains a logistic regression model that replaces
hand-tuned scoring weights.

Training data is stored in ~/.gopro-garmin/training_data.jsonl (one JSON object per
line). The trained model is stored in ~/.gopro-garmin/ranker_model.json.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import rating_visual_action


_DATA_DIR = Path.home() / ".gopro-garmin"
_TRAINING_FILE = _DATA_DIR / "training_data.jsonl"
_MODEL_FILE = _DATA_DIR / "ranker_model.json"

# Minimum examples needed before the model is used
_MIN_EXAMPLES = 100
_MIN_RIDES = 3


@dataclass
class RankerModel:
    """Simple linear model for candidate scoring."""

    weights: dict[str, float]
    intercept: float
    n_examples: int
    n_rides: int

    def predict_score(self, features: dict[str, float]) -> float:
        """Score a candidate using learned weights."""
        score = self.intercept
        for name, weight in self.weights.items():
            score += weight * features.get(name, 0.0)
        return score

    def save(self) -> None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        _MODEL_FILE.write_text(json.dumps({
            "weights": self.weights,
            "intercept": self.intercept,
            "n_examples": self.n_examples,
            "n_rides": self.n_rides,
        }, indent=2))

    @classmethod
    def load(cls) -> RankerModel | None:
        if not _MODEL_FILE.exists():
            return None
        data = json.loads(_MODEL_FILE.read_text())
        return cls(
            weights=data["weights"],
            intercept=data["intercept"],
            n_examples=data["n_examples"],
            n_rides=data["n_rides"],
        )


def _extract_features(seg_data: dict) -> dict[str, float]:
    """Extract training features from a candidate's serialized data.

    Features are designed to be available from the candidate pool (candidates.json)
    without needing the original RideData or GoPro files.
    """
    # Gemini candidates store a rubric rather than visual/action. Reading the
    # keys directly made both features constant-zero for nearly every
    # candidate, so the ranker was training on "is this a human label?"
    _visual, _action = rating_visual_action(seg_data)
    visual = float(_visual)
    action = float(_action)
    score = float(seg_data.get("score", 0))
    sources = seg_data.get("sources", [])
    source_count = float(len(sources)) if sources else 1.0

    # Source presence flags.
    # Feature name stays ``has_gemini`` for training-file / weight compat;
    # compute as any vision provider so OpenAI-scored clips share the signal.
    from .models import VISION_SOURCES
    has_gemini = float(bool(VISION_SOURCES.intersection(sources)))
    has_telemetry = float("telemetry" in sources)
    has_strava = float("strava" in sources)
    has_label = float("label" in sources)


    # Strava stars (from notes, if present)
    star_count = 0.0
    notes = seg_data.get("notes", seg_data.get("label", {}).get("notes", ""))
    if "stars)" in notes:
        try:
            star_str = notes.split("(")[-1].split("stars)")[0].strip()
            star_count = math.log(max(1, float(star_str)))
        except (ValueError, IndexError):
            pass

    return {
        "visual": visual,
        "action": action,
        "visual_x_action": visual * action,  # interaction term for synergy
        "score": score,
        "source_count": source_count,
        "has_gemini": has_gemini,
        "has_telemetry": has_telemetry,
        "has_strava": has_strava,
        "has_label": has_label,
        "strava_stars": star_count,
    }


def collect_training_data(
    all_candidates: list[dict],
    selected_indices: set[int],
    ride_id: str,
) -> None:
    """Append training examples from a single ride to the training file.

    Called after selection (either human review or auto-selection).
    Each candidate becomes one training example with label 1 (selected) or 0 (rejected).
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(_TRAINING_FILE, "a") as f:
        for i, cand in enumerate(all_candidates):
            features = _extract_features(cand)
            example = {
                "ride_id": ride_id,
                "candidate_idx": i,
                "selected": int(i in selected_indices),
                **features,
            }
            f.write(json.dumps(example) + "\n")


def collect_from_segments(
    all_candidates: list,
    selected: list,
    ride_id: str,
) -> None:
    """Collect training data from Segment objects (used during compose_highlight).

    Converts Segment objects to dicts and identifies selected indices by matching
    ride_time_secs.
    """
    selected_times = {s.ride_time_secs for s in selected}
    all_dicts = []
    selected_indices = set()

    for i, seg in enumerate(all_candidates):
        visual, action = rating_visual_action(seg.label)
        d = {
            "score": seg.score,
            "sources": seg.sources if seg.sources else [seg.source],
            "visual": visual,
            "action": action,
            "notes": seg.label.get("notes", ""),
        }
        all_dicts.append(d)
        if seg.ride_time_secs in selected_times:
            selected_indices.add(i)

    collect_training_data(all_dicts, selected_indices, ride_id)
    n_pos = len(selected_indices)
    n_neg = len(all_dicts) - n_pos
    print(f"  Training data: {n_pos} positive, {n_neg} negative examples → {_TRAINING_FILE}")


def load_training_data() -> tuple[np.ndarray, np.ndarray, int]:
    """Load all training examples from disk.

    Returns (X, y, n_rides) where X is the feature matrix and y is the label vector.
    """
    if not _TRAINING_FILE.exists():
        return np.array([]), np.array([]), 0

    examples = []
    ride_ids = set()
    for line in _TRAINING_FILE.read_text().strip().split("\n"):
        if not line:
            continue
        ex = json.loads(line)
        ride_ids.add(ex["ride_id"])
        examples.append(ex)

    if not examples:
        return np.array([]), np.array([]), 0

    feature_names = [
        "visual", "action", "visual_x_action", "score",
        "source_count", "has_gemini", "has_telemetry", "has_strava",
        "has_label", "strava_stars",
    ]
    X = np.array([[ex.get(f, 0.0) for f in feature_names] for ex in examples])
    y = np.array([ex["selected"] for ex in examples])
    return X, y, len(ride_ids)


def train() -> RankerModel | None:
    """Train a logistic regression model on accumulated training data.

    Returns None if insufficient data. Prints model diagnostics.
    """
    X, y, n_rides = load_training_data()
    n_examples = len(y)

    print(f"Training data: {n_examples} examples from {n_rides} rides")
    if n_examples > 0:
        print(f"  Positive: {int(y.sum())} ({100*y.mean():.1f}%)")
        print(f"  Negative: {int((1-y).sum())} ({100*(1-y.mean()):.1f}%)")

    if n_examples < _MIN_EXAMPLES:
        print(f"  Need at least {_MIN_EXAMPLES} examples (have {n_examples})")
        return None
    if n_rides < _MIN_RIDES:
        print(f"  Need at least {_MIN_RIDES} rides (have {n_rides})")
        return None

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("  scikit-learn not installed. Run: pip install scikit-learn")
        return None

    # Standardize features
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0  # avoid division by zero
    X_scaled = (X - mean) / std

    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    model.fit(X_scaled, y)

    # Extract weights (un-standardized for direct use)
    feature_names = [
        "visual", "action", "visual_x_action", "score",
        "source_count", "has_gemini", "has_telemetry", "has_strava",
        "has_label", "strava_stars",
    ]
    weights = {}
    for name, coef, m, s in zip(feature_names, model.coef_[0], mean, std):
        weights[name] = float(coef / s)  # un-standardize
    intercept = float(model.intercept_[0] - sum(
        coef * m / s for coef, m, s in zip(model.coef_[0], mean, std)
    ))

    # Print diagnostics
    print("\n  Learned weights (sorted by importance):")
    sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, w in sorted_weights:
        direction = "+" if w > 0 else "-"
        print(f"    {direction} {name}: {w:+.4f}")
    print(f"    intercept: {intercept:+.4f}")

    # Cross-validation accuracy
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X_scaled, y, cv=min(5, n_rides), scoring="roc_auc")
    print(f"\n  Cross-validation AUC: {scores.mean():.3f} ± {scores.std():.3f}")

    ranker = RankerModel(
        weights=weights,
        intercept=intercept,
        n_examples=n_examples,
        n_rides=n_rides,
    )
    ranker.save()
    print(f"\n  Model saved to {_MODEL_FILE}")
    return ranker


def get_learned_score(seg_data: dict) -> float | None:
    """Score a candidate using the learned model, if available.

    Returns None if no model is trained yet.
    """
    model = RankerModel.load()
    if model is None:
        return None
    features = _extract_features(seg_data)
    return model.predict_score(features)


def status() -> str:
    """Return a human-readable status of the learned ranker."""
    X, y, n_rides = load_training_data()
    n_examples = len(y)
    model = RankerModel.load()

    lines = [f"Training data: {n_examples} examples from {n_rides} rides"]
    if n_examples > 0:
        lines.append(f"  Positive: {int(y.sum())} ({100*y.mean():.1f}%)")
    lines.append(f"  Required: {_MIN_EXAMPLES} examples, {_MIN_RIDES} rides")

    if model:
        lines.append(f"\nTrained model: {model.n_examples} examples, {model.n_rides} rides")
        sorted_weights = sorted(model.weights.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, w in sorted_weights[:5]:
            lines.append(f"  {name}: {w:+.4f}")
    else:
        remaining = max(0, _MIN_EXAMPLES - n_examples)
        lines.append(f"\nNo model yet — need ~{remaining} more examples")

    return "\n".join(lines)
