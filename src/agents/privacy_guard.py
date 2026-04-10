"""Minimal rule-based PII detection for privacy-aware memory storage."""

from __future__ import annotations

import re
from typing import Dict, List, Pattern


class PrivacyGuard:
    """Lightweight PII detection for privacy-aware memory management.

    Detects personally identifiable information in conversation text
    and provides risk levels for memory storage decisions.
    """

    PII_PATTERNS = {
        "phone": r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
        "email": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
        "ssn": r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-.\s]?){3}\d{4}\b",
        "address": r"\b\d{1,5}\s+[A-Z][a-z]+\s+(?:St|Ave|Blvd|Dr|Rd|Ln|Way|Ct)\b",
        "name_disclosure": r"(?:my (?:real |full |legal )?name is|I'm called|call me)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?",
        "financial": r"(?:bank account|routing number|account number|salary|income)\s*(?:is|:)?\s*[\d$,]+",
    }

    RISK_LEVELS = {
        "phone": "high",
        "email": "medium",
        "ssn": "critical",
        "credit_card": "critical",
        "address": "high",
        "name_disclosure": "low",
        "financial": "high",
    }

    RISK_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

    def __init__(self) -> None:
        self._compiled: Dict[str, Pattern[str]] = {
            pii_type: re.compile(pattern, re.IGNORECASE)
            for pii_type, pattern in self.PII_PATTERNS.items()
        }

    def scan(self, text: str) -> dict:
        """Scan text for PII. Returns dict with detected PII types and overall risk."""
        detections: List[dict] = []
        max_risk = "none"

        for pii_type, pattern in self._compiled.items():
            matches = pattern.findall(text)
            if matches:
                risk = self.RISK_LEVELS[pii_type]
                detections.append(
                    {
                        "type": pii_type,
                        "count": len(matches),
                        "risk": risk,
                    }
                )
                if self.RISK_ORDER.get(risk, 0) > self.RISK_ORDER.get(max_risk, 0):
                    max_risk = risk

        return {
            "has_pii": len(detections) > 0,
            "detections": detections,
            "overall_risk": max_risk,
            "recommendation": self._recommend(max_risk),
        }

    def _recommend(self, risk: str) -> str:
        recommendations = {
            "none": "safe_to_store",
            "low": "store_with_flag",
            "medium": "store_redacted",
            "high": "require_consent",
            "critical": "do_not_store",
        }
        return recommendations.get(risk, "safe_to_store")

    def redact(self, text: str) -> str:
        """Replace detected PII with [REDACTED] placeholders."""
        result = text
        for pii_type, pattern in self._compiled.items():
            result = pattern.sub(f"[REDACTED_{pii_type.upper()}]", result)
        return result
