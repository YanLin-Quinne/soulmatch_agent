"""Track personality consistency across Big Five feature snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
from statistics import pvariance
from typing import Any, Optional


BIG_FIVE_TRAITS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)


class PersonalityConsistencyTracker:
    """Measure how stable inferred Big Five traits remain over time."""

    def __init__(self, window_size: int = 5):
        self.feature_snapshots: list[dict[str, Any]] = []
        self.consistency_scores: list[dict[str, Any]] = []
        self.window_size = window_size

    def record_snapshot(self, turn: int, features: dict, confidences: dict):
        """Store a Big Five-only feature snapshot for a conversation turn."""
        snapshot_features: dict[str, float] = {}
        snapshot_confidences: dict[str, float] = {}

        feature_big_five = features.get("big_five", {}) if isinstance(features, dict) else {}
        confidence_big_five = confidences.get("big_five", {}) if isinstance(confidences, dict) else {}

        for trait in BIG_FIVE_TRAITS:
            value = self._coerce_float(
                (features or {}).get(f"big_five_{trait}", feature_big_five.get(trait))
            )
            confidence = self._coerce_float(
                (confidences or {}).get(f"big_five_{trait}", confidence_big_five.get(trait))
            )
            if value is None:
                continue
            snapshot_features[trait] = value
            if confidence is not None:
                snapshot_confidences[trait] = confidence

        self.feature_snapshots.append(
            {
                "turn": turn,
                "features": snapshot_features,
                "confidences": snapshot_confidences,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def compute_consistency(self) -> dict:
        """Compute per-trait stability and an aggregate consistency score."""
        return self._build_consistency_result(store=True)

    def detect_sudden_shift(self) -> Optional[dict]:
        """Detect large turn-to-turn Big Five changes."""
        if len(self.feature_snapshots) < 2:
            return None

        previous = self.feature_snapshots[-2]
        latest = self.feature_snapshots[-1]
        largest_shift: Optional[dict[str, Any]] = None

        for trait in BIG_FIVE_TRAITS:
            if trait not in previous["features"] or trait not in latest["features"]:
                continue
            old_value = previous["features"][trait]
            new_value = latest["features"][trait]
            delta = abs(new_value - old_value)
            if delta <= 0.15:
                continue

            shift = {
                "trait": trait,
                "old_value": old_value,
                "new_value": new_value,
                "delta": delta,
                "turn": latest["turn"],
            }
            if largest_shift is None or shift["delta"] > largest_shift["delta"]:
                largest_shift = shift

        return largest_shift

    def get_report(self) -> dict:
        """Return a combined report with score, evolution summary, and alerts."""
        consistency = self._build_consistency_result(store=False)
        sudden_shift = self.detect_sudden_shift()

        report = {
            **consistency,
            "evolution_summary": {
                "first_turn": self.feature_snapshots[0]["turn"] if self.feature_snapshots else None,
                "latest_turn": self.feature_snapshots[-1]["turn"] if self.feature_snapshots else None,
                "window_size": self.window_size,
                "tracked_snapshots": len(self.feature_snapshots),
                "score_history": self.consistency_scores[-self.window_size :],
            },
            "sudden_shift": sudden_shift,
            "recommendation": None,
        }

        if consistency["status"] == "volatile":
            report["recommendation"] = "persona anchoring needed"

        return report

    def _build_consistency_result(self, store: bool) -> dict:
        if len(self.feature_snapshots) < 2:
            return {"score": 1.0, "status": "insufficient_data"}

        recent_snapshots = self.feature_snapshots[-self.window_size :]
        latest_turn = self.feature_snapshots[-1]["turn"]
        per_trait: dict[str, dict[str, float]] = {}
        stability_indices: list[float] = []

        for trait in BIG_FIVE_TRAITS:
            recent_values = [
                snapshot["features"][trait]
                for snapshot in recent_snapshots
                if trait in snapshot["features"]
            ]
            variance = pvariance(recent_values) if len(recent_values) >= 2 else 0.0

            first_observation = next(
                (snapshot for snapshot in self.feature_snapshots if trait in snapshot["features"]),
                None,
            )
            latest_observation = next(
                (snapshot for snapshot in reversed(self.feature_snapshots) if trait in snapshot["features"]),
                None,
            )

            drift_rate = 0.0
            if first_observation and latest_observation:
                turn_span = max(1, latest_observation["turn"] - first_observation["turn"])
                drift_rate = abs(
                    latest_observation["features"][trait] - first_observation["features"][trait]
                ) / turn_span

            stability = max(0.0, min(1.0, 1.0 - variance))
            per_trait[trait] = {
                "variance": variance,
                "drift_rate": drift_rate,
                "stability": stability,
            }
            stability_indices.append(stability)

        score = sum(stability_indices) / len(stability_indices) if stability_indices else 1.0
        if score >= 0.85:
            status = "stable"
        elif score >= 0.65:
            status = "drifting"
        else:
            status = "volatile"

        result = {
            "score": score,
            "per_trait": per_trait,
            "status": status,
            "n_snapshots": len(self.feature_snapshots),
        }

        if store:
            self.consistency_scores.append(
                {
                    "turn": latest_turn,
                    "score": score,
                    "details": {
                        "status": status,
                        "per_trait": per_trait,
                    },
                }
            )

        return result

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
