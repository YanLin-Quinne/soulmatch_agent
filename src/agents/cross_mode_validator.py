"""CrossModeValidator — verify personality consistency across conversation modes.

When a user switches from dating chat to expert consulting to self-dialogue,
the underlying personality should remain consistent. This module:
1. Generates a personality fingerprint from current features
2. Validates that fingerprints match across mode transitions
3. Detects mode-specific personality distortion
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from loguru import logger


CONVERSATION_MODES = ("social", "expert_consulting", "emotional_support", "self_dialogue")

BIG_FIVE_KEYS = (
    "big_five_openness", "big_five_conscientiousness",
    "big_five_extraversion", "big_five_agreeableness", "big_five_neuroticism",
)


@dataclass
class ModeSnapshot:
    mode: str
    turn: int
    fingerprint: str
    features: Dict[str, float]
    communication_style: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CrossModeValidator:
    CONSISTENCY_TOLERANCE = 0.15

    def __init__(self):
        self.mode_snapshots: List[ModeSnapshot] = []
        self.mode_transitions: List[Dict[str, Any]] = []
        self.current_mode: str = "social"
        self._violation_count: int = 0

    def set_mode(self, mode: str):
        if mode not in CONVERSATION_MODES:
            logger.warning(f"[CrossMode] Unknown mode '{mode}', defaulting to 'social'")
            mode = "social"
        if mode != self.current_mode:
            self.mode_transitions.append({
                "from": self.current_mode,
                "to": mode,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            logger.info(f"[CrossMode] Mode transition: {self.current_mode} -> {mode}")
            self.current_mode = mode

    def capture_snapshot(self, turn: int, features: dict, communication_style: str = "casual") -> ModeSnapshot:
        big_five = {}
        for key in BIG_FIVE_KEYS:
            val = features.get(key)
            if val is not None:
                big_five[key] = float(val)
        fingerprint = self._compute_fingerprint(big_five, communication_style)
        snapshot = ModeSnapshot(
            mode=self.current_mode, turn=turn, fingerprint=fingerprint,
            features=big_five, communication_style=communication_style,
        )
        self.mode_snapshots.append(snapshot)
        return snapshot

    def validate_consistency(self) -> dict:
        latest_per_mode: Dict[str, ModeSnapshot] = {}
        for snap in self.mode_snapshots:
            latest_per_mode[snap.mode] = snap
        if len(latest_per_mode) < 2:
            return {"consistent": True, "score": 1.0, "modes_compared": list(latest_per_mode.keys()),
                    "violations": [], "message": "insufficient_modes_for_comparison"}
        modes = list(latest_per_mode.keys())
        violations = []
        all_deviations = []
        for i in range(len(modes)):
            for j in range(i + 1, len(modes)):
                snap_a = latest_per_mode[modes[i]]
                snap_b = latest_per_mode[modes[j]]
                for trait in BIG_FIVE_KEYS:
                    val_a = snap_a.features.get(trait)
                    val_b = snap_b.features.get(trait)
                    if val_a is None or val_b is None:
                        continue
                    deviation = abs(val_a - val_b)
                    all_deviations.append(deviation)
                    if deviation > self.CONSISTENCY_TOLERANCE:
                        violations.append({
                            "trait": trait, "mode_a": modes[i], "mode_b": modes[j],
                            "value_a": round(val_a, 3), "value_b": round(val_b, 3),
                            "deviation": round(deviation, 3),
                        })
        if all_deviations:
            avg_dev = sum(all_deviations) / len(all_deviations)
            score = max(0.0, 1.0 - (avg_dev / self.CONSISTENCY_TOLERANCE))
        else:
            score = 1.0
        self._violation_count += len(violations)
        return {"consistent": len(violations) == 0, "score": round(score, 3),
                "modes_compared": modes, "violations": violations,
                "mode_transitions": len(self.mode_transitions),
                "total_violations": self._violation_count}

    def get_report(self) -> dict:
        return {
            "current_mode": self.current_mode,
            "mode_transitions": self.mode_transitions,
            "consistency": self.validate_consistency(),
            "total_snapshots": len(self.mode_snapshots),
        }

    def _compute_fingerprint(self, big_five: dict, comm_style: str) -> str:
        quantized = {k: round(v * 10) / 10 for k, v in sorted(big_five.items())}
        data = json.dumps({"big_five": quantized, "style": comm_style}, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()[:12]
