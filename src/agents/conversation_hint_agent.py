"""ConversationHintAgent — detects stuck conversations and suggests helpful topics."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, TYPE_CHECKING

from loguru import logger

from src.agents.llm_router import router, AgentRole

if TYPE_CHECKING:
    from src.agents.agent_context import AgentContext


# Static fallback suggestions when LLM is unavailable
_STATIC_HINTS: List[Dict[str, str]] = [
    {"text": "What's something you've been really into lately?", "type": "topic_suggestion"},
    {"text": "If you could travel anywhere tomorrow, where would you go?", "type": "topic_suggestion"},
    {"text": "What's the best meal you've had recently?", "type": "topic_suggestion"},
    {"text": "Do you have any weekend plans coming up?", "type": "topic_suggestion"},
    {"text": "What kind of music have you been listening to?", "type": "topic_suggestion"},
]

# Patterns that indicate user is explicitly asking for topic help
_TOPIC_REQUEST_PATTERNS = [
    r"what should we talk about",
    r"what (can|do) we (talk|chat) about",
    r"i don'?t know what to (say|talk)",
    r"(suggest|recommend).*(topic|something)",
    r"running out of (things|stuff) to (say|talk)",
    r"awkward silence",
    r"what('s| is) on your mind",
]


class ConversationHintAgent:
    """Detects stuck conversations and generates topic suggestions.

    Trigger conditions:
    - Last 3+ messages are very short (<15 chars each)
    - Emotion has been 'neutral' for 3+ turns
    - User explicitly asks for topic ideas
    """

    def __init__(self, short_msg_threshold: int = 15, neutral_streak_min: int = 3):
        self.short_msg_threshold = short_msg_threshold
        self.neutral_streak_min = neutral_streak_min
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in _TOPIC_REQUEST_PATTERNS
        ]

    def should_suggest(self, ctx: "AgentContext") -> bool:
        """Return True when the conversation appears stuck and hints would help."""
        # Condition 1: User explicitly asks for topics
        if ctx.user_message and self._is_topic_request(ctx.user_message):
            logger.debug("[ConversationHint] Triggered: explicit topic request")
            return True

        # Condition 2: Last 3+ messages are all very short
        recent = ctx.conversation_history[-6:]  # check last 6 entries
        user_msgs = [h["message"] for h in recent if h.get("speaker") == "user"]
        if len(user_msgs) >= 3:
            last_3 = user_msgs[-3:]
            if all(len(m.strip()) < self.short_msg_threshold for m in last_3):
                logger.debug("[ConversationHint] Triggered: 3+ short messages")
                return True

        # Condition 3: Emotion has been neutral for 3+ consecutive turns
        if len(ctx.emotion_history) >= self.neutral_streak_min:
            tail = ctx.emotion_history[-self.neutral_streak_min:]
            if all(e == "neutral" for e in tail):
                logger.debug("[ConversationHint] Triggered: neutral emotion streak")
                return True

        return False

    def generate_hints(self, ctx: "AgentContext") -> List[Dict[str, str]]:
        """Generate 3-5 topic suggestions, preferring topics that help with prediction.

        Uses LLM via router (AgentRole.QUESTION) for natural suggestions.
        Falls back to static list if LLM fails.
        """
        low_conf = ctx.low_confidence_features[:6] if ctx.low_confidence_features else []

        recent_lines = "\n".join(
            f"{h['speaker']}: {h['message']}" for h in ctx.conversation_history[-6:]
        )

        prompt = (
            "The conversation between two people on a dating app seems stuck or running low on energy.\n\n"
            f"Recent messages:\n{recent_lines}\n\n"
        )
        if low_conf:
            prompt += (
                f"We'd also like to learn more about these traits: {', '.join(low_conf)}\n"
                "Suggest topics that naturally reveal these traits without being obvious.\n\n"
            )
        prompt += (
            "Generate 4 fun, casual topic suggestions that could re-energize the chat.\n"
            "Each should feel like a natural conversation starter, NOT an interview question.\n\n"
            'Respond with JSON: {"hints":[{"text":"...","type":"topic_suggestion"}]}'
        )

        try:
            text = router.chat(
                role=AgentRole.QUESTION,
                system="You are a friendly dating-app conversation coach. Suggest engaging topics.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=300,
                json_mode=True,
            )
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)
            raw_hints = data.get("hints", [])

            valid: List[Dict[str, str]] = []
            for h in raw_hints[:5]:
                if isinstance(h, dict) and "text" in h:
                    valid.append({"text": h["text"], "type": h.get("type", "topic_suggestion")})
                elif isinstance(h, str):
                    valid.append({"text": h, "type": "topic_suggestion"})

            if valid:
                logger.info(f"[ConversationHint] Generated {len(valid)} hints via LLM")
                return valid

            # LLM returned empty — fall through to static
            logger.warning("[ConversationHint] LLM returned no valid hints, using static fallback")
        except Exception as e:
            logger.warning(f"[ConversationHint] LLM failed ({e}), using static fallback")

        # Static fallback: pick 4 from the static list
        return _STATIC_HINTS[:4]

    def _is_topic_request(self, message: str) -> bool:
        """Check if the user is explicitly asking for conversation topics."""
        for pattern in self._compiled_patterns:
            if pattern.search(message):
                return True
        return False
