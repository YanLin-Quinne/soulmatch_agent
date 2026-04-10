"""MemoryPrivacyManager — privacy-preserving memory and feature management.

Provides:
- Feature privacy tiers (public/private/sensitive)
- Selective memory forgetting
- Differential privacy noise for personality scores
- User consent tracking
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from loguru import logger


DEFAULT_PRIVACY_TIERS: Dict[str, List[str]] = {
    "public": [
        "big_five_openness",
        "big_five_conscientiousness",
        "big_five_extraversion",
        "big_five_agreeableness",
        "big_five_neuroticism",
        "communication_style",
        "relationship_goals",
        "interest_music",
        "interest_sports",
        "interest_travel",
        "interest_food",
        "interest_arts",
        "interest_tech",
        "interest_outdoors",
        "interest_books",
    ],
    "private": [
        "mbti_type",
        "attachment_style",
        "attachment_anxiety",
        "attachment_avoidance",
        "trust_score",
        "trust_velocity",
        "primary_love_language",
        "secondary_love_language",
    ],
    "sensitive": [
        "age",
        "sex",
        "orientation",
        "location",
        "religion",
        "income",
        "education",
        "ethnicity",
    ],
}


@dataclass
class ConsentRecord:
    """Tracks user consent for a specific feature or memory operation."""

    feature_or_operation: str
    consented: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reason: str = ""


class MemoryPrivacyManager:
    """Manages privacy tiers, consent, selective forgetting, and differential privacy."""

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        self.privacy_tiers: Dict[str, List[str]] = {
            tier: list(features) for tier, features in DEFAULT_PRIVACY_TIERS.items()
        }
        self.consent_log: List[ConsentRecord] = []
        self.forgotten_memories: List[Dict[str, Any]] = []
        self._feature_tier_map: Dict[str, str] = self._build_tier_map()

    def _build_tier_map(self) -> Dict[str, str]:
        """Build reverse mapping: feature_name -> tier."""
        tier_map: Dict[str, str] = {}
        for tier, features in self.privacy_tiers.items():
            for feature_name in features:
                tier_map[feature_name] = tier
        return tier_map

    def get_feature_tier(self, feature_name: str) -> str:
        """Get privacy tier for a feature. Unknown features default to private."""
        return self._feature_tier_map.get(feature_name, "private")

    def can_share(self, feature_name: str) -> bool:
        """Check if a feature can be shared externally."""
        tier = self.get_feature_tier(feature_name)
        if tier == "public":
            return True
        if tier == "private":
            return self._has_consent(feature_name)
        return False

    def can_store(self, feature_name: str) -> bool:
        """Check if a feature can be stored in memory."""
        tier = self.get_feature_tier(feature_name)
        if tier == "sensitive":
            return self._has_consent(f"store_{feature_name}")
        return True

    def forget_memory(
        self,
        memory_id: str,
        memory_content: str,
        reason: str = "user_request",
    ) -> dict:
        """Mark a memory for deletion and retain only a content hash for audit."""
        record = {
            "memory_id": memory_id,
            "content_hash": hashlib.sha256(memory_content.encode("utf-8")).hexdigest(),
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.forgotten_memories.append(record)
        logger.info(f"[MemoryPrivacy] Forgot memory {memory_id} (reason={reason})")
        return record

    def forget_feature(self, feature_name: str, reason: str = "user_request") -> dict:
        """Mark a specific feature for deletion from future sharing and retrieval."""
        record = {
            "feature": feature_name,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.forgotten_memories.append(record)
        logger.info(f"[MemoryPrivacy] Forgot feature {feature_name} (reason={reason})")
        return record

    def get_forgotten_ids(self) -> Set[str]:
        """Get the set of forgotten memory IDs."""
        return {
            record["memory_id"]
            for record in self.forgotten_memories
            if "memory_id" in record
        }

    def get_forgotten_features(self) -> Set[str]:
        """Get the set of forgotten feature names."""
        return {
            record["feature"]
            for record in self.forgotten_memories
            if "feature" in record
        }

    def add_laplace_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """Add calibrated Laplace noise and clip the result to the [0, 1] range."""
        if self.epsilon <= 0:
            return value

        scale = sensitivity / self.epsilon
        sample = random.random() - 0.5
        laplace_noise = -scale * math.copysign(1, sample) * math.log(
            1 - 2 * abs(sample) + 1e-10
        )
        noisy_value = value + laplace_noise
        return max(0.0, min(1.0, noisy_value))

    def privatize_features(self, features: dict, confidences: dict) -> dict:
        """Return a privacy-safe version of features for external sharing."""
        del confidences

        forgotten = self.get_forgotten_features()
        safe_features: Dict[str, Any] = {}

        for key, value in features.items():
            if key in forgotten:
                continue

            tier = self.get_feature_tier(key)
            if tier == "sensitive":
                continue
            if tier == "private" and not self._has_consent(key):
                continue
            if tier == "private" and isinstance(value, (int, float)):
                safe_features[key] = round(self.add_laplace_noise(float(value)), 3)
                continue
            safe_features[key] = value

        return safe_features

    def record_consent(
        self,
        feature_or_operation: str,
        consented: bool,
        reason: str = "",
    ) -> None:
        """Record the user's consent decision for a feature or operation."""
        self.consent_log.append(
            ConsentRecord(
                feature_or_operation=feature_or_operation,
                consented=consented,
                reason=reason,
            )
        )
        logger.info(
            f"[MemoryPrivacy] Consent {'granted' if consented else 'denied'} "
            f"for {feature_or_operation}"
        )

    def _has_consent(self, feature_or_operation: str) -> bool:
        """Return the most recent consent decision."""
        for record in reversed(self.consent_log):
            if record.feature_or_operation == feature_or_operation:
                return record.consented
        return False

    def get_consent_summary(self) -> dict:
        """Summarize the current consent state."""
        latest_consent: Dict[str, bool] = {}
        for record in self.consent_log:
            latest_consent[record.feature_or_operation] = record.consented

        return {
            "consented": [key for key, value in latest_consent.items() if value],
            "denied": [key for key, value in latest_consent.items() if not value],
            "total_decisions": len(self.consent_log),
        }

    def get_privacy_report(self) -> dict:
        """Return a full privacy status report."""
        if self.epsilon < 0.5:
            privacy_level = "high"
        elif self.epsilon < 2.0:
            privacy_level = "moderate"
        else:
            privacy_level = "low"

        return {
            "epsilon": self.epsilon,
            "privacy_level": privacy_level,
            "feature_tiers": {
                tier: len(features) for tier, features in self.privacy_tiers.items()
            },
            "forgotten_memories": len(self.forgotten_memories),
            "forgotten_features": list(self.get_forgotten_features()),
            "consent_summary": self.get_consent_summary(),
        }
