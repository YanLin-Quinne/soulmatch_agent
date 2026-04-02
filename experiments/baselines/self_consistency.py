"""Self-consistency baseline: multiple CoT samples with aggregation."""

import time
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from src.agents.llm_router import router

from .cot_prompting import CoTBaseline

__all__ = ["SelfConsistencyBaseline"]

# Big Five trait keys in canonical order
_BIG_FIVE_TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

_REL_TYPES = ["love", "friendship", "family", "other"]
_REL_STATUSES = ["stranger", "acquaintance", "crush", "dating", "committed"]


class SelfConsistencyBaseline:
    """Baseline that samples multiple CoT responses and aggregates results.

    For personality: Big Five scores are averaged, MBTI is majority-voted,
    and confidence is derived from cross-sample standard deviation.
    For relationships: majority vote on type and status.
    """

    def __init__(self, llm_router=None, n_samples: int = 5):
        self.n_samples = n_samples
        self.cot = CoTBaseline(llm_router=llm_router or router)

    def predict_personality(self, dialogue: List[Dict]) -> Optional[Dict]:
        """Infer personality by sampling n CoT predictions and aggregating.

        Aggregation strategy:
        - Big Five: mean of valid numeric values per trait
        - MBTI: majority vote across samples
        - Confidences: 1 - std (lower cross-sample variance = higher confidence)

        Args:
            dialogue: List of {speaker, message} dicts.

        Returns:
            Aggregated personality dict, or None if all samples fail.
        """
        t0 = time.time()
        samples = []

        for i in range(self.n_samples):
            result = self.cot.predict_personality(
                dialogue, temperature=0.7
            )
            if result and "big_five" in result:
                samples.append(result)
            else:
                logger.warning(f"[SelfConsistency] personality sample {i+1} failed")

        elapsed = time.time() - t0

        if not samples:
            logger.error("[SelfConsistency] all personality samples failed")
            return None

        # Aggregate Big Five scores: mean across samples
        big_five = {}
        confidences = {}
        for trait in _BIG_FIVE_TRAITS:
            values = []
            for s in samples:
                v = s.get("big_five", {}).get(trait)
                if v is not None and isinstance(v, (int, float)):
                    values.append(float(v))
            if values:
                big_five[trait] = round(float(np.mean(values)), 3)
                std = float(np.std(values))
                # Confidence: 1 - std (clamped to [0, 1])
                confidences[trait] = round(max(0.0, min(1.0, 1.0 - std)), 3)
            else:
                big_five[trait] = 0.5
                confidences[trait] = 0.0

        # MBTI: majority vote
        mbti_votes = [
            s.get("mbti", "").upper()
            for s in samples
            if s.get("mbti") and len(s.get("mbti", "")) == 4
        ]
        if mbti_votes:
            mbti = Counter(mbti_votes).most_common(1)[0][0]
            mbti_agreement = Counter(mbti_votes).most_common(1)[0][1] / len(mbti_votes)
            confidences["mbti"] = round(mbti_agreement, 3)
        else:
            mbti = "XXXX"
            confidences["mbti"] = 0.0

        return {
            "big_five": big_five,
            "mbti": mbti,
            "confidences": confidences,
            "n_valid_samples": len(samples),
            "elapsed_seconds": round(elapsed, 3),
            "method": "self_consistency",
        }

    def predict_relationship(
        self, dialogue: List[Dict], context: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Predict relationship by sampling n CoT predictions and majority-voting.

        Args:
            dialogue: List of {speaker, message} dicts.
            context: Optional additional context.

        Returns:
            Aggregated relationship dict, or None if all samples fail.
        """
        t0 = time.time()
        samples = []

        for i in range(self.n_samples):
            result = self.cot.predict_relationship(
                dialogue, context=context, temperature=0.7
            )
            if result and "rel_type" in result:
                samples.append(result)
            else:
                logger.warning(f"[SelfConsistency] relationship sample {i+1} failed")

        elapsed = time.time() - t0

        if not samples:
            logger.error("[SelfConsistency] all relationship samples failed")
            return None

        # Majority vote on rel_type
        type_votes = [s.get("rel_type", "other") for s in samples]
        rel_type = Counter(type_votes).most_common(1)[0][0]

        # Majority vote on rel_status
        status_votes = [s.get("rel_status", "stranger") for s in samples]
        rel_status = Counter(status_votes).most_common(1)[0][0]

        # Aggregate probability distributions by averaging
        rel_type_probs = {}
        for t in _REL_TYPES:
            values = []
            for s in samples:
                v = s.get("rel_type_probs", {}).get(t)
                if v is not None and isinstance(v, (int, float)):
                    values.append(float(v))
            rel_type_probs[t] = round(float(np.mean(values)), 3) if values else 0.0

        rel_status_probs = {}
        for st in _REL_STATUSES:
            values = []
            for s in samples:
                v = s.get("rel_status_probs", {}).get(st)
                if v is not None and isinstance(v, (int, float)):
                    values.append(float(v))
            rel_status_probs[st] = round(float(np.mean(values)), 3) if values else 0.0

        return {
            "rel_type": rel_type,
            "rel_status": rel_status,
            "rel_type_probs": rel_type_probs,
            "rel_status_probs": rel_status_probs,
            "n_valid_samples": len(samples),
            "elapsed_seconds": round(elapsed, 3),
            "method": "self_consistency",
        }
