"""ConversationSentimentAgent — conversation-level sentiment analysis (rule-based, no extra LLM)."""

from __future__ import annotations

from typing import Dict, Any, List, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from src.agents.agent_context import AgentContext


VALENCE_MAP = {
    "joy": 1.0, "excitement": 0.8, "interest": 0.6, "trust": 0.7,
    "love": 0.9, "surprise": 0.3,
    "sadness": -0.7, "anger": -0.9, "fear": -0.6, "disgust": -0.8,
    "anxiety": -0.5,
    "neutral": 0.0,
}


class ConversationSentimentAgent:
    """Analyze overall conversation sentiment from emotion history.

    Pure rule-based: no LLM calls. Uses sliding window over emotion_history
    to compute conversation-level positive/neutral/negative + trend + hints.
    """

    def __init__(self, window_size: int = 8):
        self.window_size = window_size

    def analyze_conversation(self, ctx: "AgentContext") -> Dict[str, Any]:
        """Compute conversation-level sentiment from emotion history.

        Returns:
            {"label": str, "score": float, "trend": str, "hints": list[str]}
        """
        from src.agents.agent_context import AgentContext  # avoid circular
        assert isinstance(ctx, AgentContext)

        emotions = ctx.emotion_history
        if len(emotions) < 2:
            return {"label": "neutral", "score": 0.0, "trend": "stable", "hints": []}

        # Sliding window
        window = emotions[-self.window_size:]
        valences = [VALENCE_MAP.get(e, 0.0) for e in window]
        avg = sum(valences) / len(valences)

        # Label
        if avg > 0.25:
            label = "positive"
        elif avg < -0.25:
            label = "negative"
        else:
            label = "neutral"

        # Trend: compare first half vs second half
        trend = self._compute_trend(valences)

        # Hints
        hints = self._generate_hints(label, trend, emotions)

        # Write to context
        ctx.conversation_sentiment = label
        ctx.conversation_sentiment_score = round(avg, 3)
        ctx.conversation_sentiment_trend = trend
        ctx.conversation_sentiment_hints = hints

        logger.debug(f"[ConversationSentiment] {label} (score={avg:.2f}, trend={trend})")

        return {"label": label, "score": round(avg, 3), "trend": trend, "hints": hints}

    def _compute_trend(self, valences: List[float]) -> str:
        if len(valences) < 4:
            return "stable"
        mid = len(valences) // 2
        first_half = sum(valences[:mid]) / mid
        second_half = sum(valences[mid:]) / (len(valences) - mid)
        diff = second_half - first_half
        if diff > 0.2:
            return "improving"
        elif diff < -0.2:
            return "declining"
        return "stable"

    def _generate_hints(self, label: str, trend: str, emotions: List[str]) -> List[str]:
        hints = []
        if label == "negative" and trend == "declining":
            hints.append("The conversation mood is dropping — try a lighter topic")
        elif label == "negative" and trend == "stable":
            hints.append("Things feel tense — consider acknowledging their feelings")
        elif label == "positive" and trend == "improving":
            hints.append("Great vibe! This is a good moment to go deeper")
        elif label == "positive" and trend == "stable":
            hints.append("Conversation is flowing well — keep it up")
        elif label == "neutral" and trend == "declining":
            hints.append("Energy is fading — try asking something personal")

        # Detect emotion volatility
        if len(emotions) >= 4:
            recent = emotions[-4:]
            unique = len(set(recent))
            if unique >= 3:
                hints.append("Emotions are shifting fast — they might be processing something")

        return hints[:3]
