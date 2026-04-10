"""PersonaStabilizer - active personality drift correction.

When personality consistency drops below threshold, this module:
1. Identifies personality-defining anchor moments from conversation history
2. Injects anchor memories into the persona prompt to reinforce core traits
3. Tracks stabilization interventions over time
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class PersonalityAnchor:
    """A personality-defining moment from conversation."""

    turn: int
    trait: str
    evidence: str
    trait_value: float
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PersonaStabilizer:
    """Active drift correction for personality preservation."""

    DRIFT_THRESHOLD = 0.75
    MAX_ANCHORS_PER_TRAIT = 3
    BIG_FIVE = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")

    def __init__(self):
        self.anchors: Dict[str, List[PersonalityAnchor]] = {t: [] for t in self.BIG_FIVE}
        self.stabilization_history: List[Dict[str, Any]] = []
        self._intervention_count = 0

    def record_anchor(
        self,
        turn: int,
        trait: str,
        evidence: str,
        trait_value: float,
        confidence: float,
    ):
        """Record a personality-defining moment as an anchor."""
        if trait not in self.BIG_FIVE:
            return

        anchor = PersonalityAnchor(
            turn=turn,
            trait=trait,
            evidence=evidence,
            trait_value=trait_value,
            confidence=confidence,
        )
        self.anchors[trait].append(anchor)
        self.anchors[trait].sort(key=lambda a: a.confidence, reverse=True)
        self.anchors[trait] = self.anchors[trait][: self.MAX_ANCHORS_PER_TRAIT]
        logger.debug(
            "[PersonaStabilizer] Recorded anchor for {} at turn {} (conf={:.2f})",
            trait,
            turn,
            confidence,
        )

    def should_stabilize(self, consistency_score: float) -> bool:
        """Check if stabilization intervention is needed."""
        return consistency_score < self.DRIFT_THRESHOLD

    def generate_stability_prompt(self, consistency_report: dict) -> Optional[str]:
        """Generate a personality reinforcement prompt block when drift is detected."""
        score = float(consistency_report.get("score", 1.0))
        if not self.should_stabilize(score):
            return None

        per_trait = consistency_report.get("per_trait", {})
        drifting_traits = [
            (t, info) for t, info in per_trait.items()
            if isinstance(info, dict) and info.get("stability", 1.0) < 0.7
        ]
        if not drifting_traits:
            return None

        lines = [
            "[Personality Anchor - Drift Correction Active]",
            "Your core personality traits are drifting. Stay true to these defining moments:",
        ]
        for trait, _info in drifting_traits:
            trait_anchors = self.anchors.get(trait, [])
            if trait_anchors:
                best = trait_anchors[0]
                evidence = " ".join(best.evidence.split())
                if len(evidence) > 120:
                    evidence = evidence[:120] + "..."
                lines.append(
                    f'- {trait.capitalize()} ({best.trait_value:.2f}): "{evidence}" (turn {best.turn})'
                )
            else:
                lines.append(f"- {trait.capitalize()}: maintain current level, avoid sudden shifts")

        lines.append(f"\nConsistency score: {score:.2f} - actively reinforcing personality stability.")

        self._intervention_count += 1
        self.stabilization_history.append(
            {
                "turn": consistency_report.get("n_snapshots", 0),
                "score_at_intervention": score,
                "drifting_traits": [t for t, _ in drifting_traits],
                "anchors_used": sum(len(self.anchors[t]) for t, _ in drifting_traits),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        logger.info(
            "[PersonaStabilizer] Intervention #{}: score={:.2f}, drifting={}",
            self._intervention_count,
            score,
            [t for t, _ in drifting_traits],
        )
        return "\n".join(lines)

    def extract_anchors_from_features(
        self,
        turn: int,
        features: dict,
        confidences: dict,
        conversation_excerpt: str,
    ):
        """Auto-extract anchors from high-confidence feature predictions."""
        excerpt = conversation_excerpt[-200:] if conversation_excerpt else ""
        for trait in self.BIG_FIVE:
            key = f"big_five_{trait}"
            conf = confidences.get(key, 0)
            value = features.get(key)
            if conf > 0.8 and value is not None:
                self.record_anchor(
                    turn=turn,
                    trait=trait,
                    evidence=excerpt,
                    trait_value=float(value),
                    confidence=float(conf),
                )

    def get_stabilization_report(self) -> dict:
        """Return summary of all stabilization activity."""
        return {
            "total_interventions": self._intervention_count,
            "anchors_per_trait": {t: len(a) for t, a in self.anchors.items()},
            "history": self.stabilization_history[-10:],
            "total_anchors": sum(len(a) for a in self.anchors.values()),
        }

    def get_report(self) -> dict:
        """Backward-compatible alias for stabilization summary."""
        return self.get_stabilization_report()
