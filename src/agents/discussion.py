"""Dynamic Discussion — multi-perspective debate to improve bot responses.

Before generating the final bot reply, several "perspective agents" each provide
a brief recommendation.  A moderator then synthesises the recommendations into
a single, actionable brief that gets injected into PersonaAgent's context.

All LLM calls go through the shared LLMRouter, so fallback chains and cost
tracking remain centralised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from src.agents.llm_router import router, AgentRole
from src.agents.agent_context import AgentContext


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Perspective:
    name: str
    role_description: str
    recommendation: str = ""


@dataclass
class DiscussionResult:
    perspectives: list[Perspective] = field(default_factory=list)
    synthesis: str = ""
    triggered: bool = False


# ---------------------------------------------------------------------------
# Perspective prompts
# ---------------------------------------------------------------------------

PERSPECTIVES: list[dict] = [
    {
        "name": "Emotional Intelligence Advisor",
        "system": (
            "You are an emotional intelligence advisor for a dating chat bot. "
            "Given the conversation context, give ONE short recommendation (2-3 sentences) "
            "about how the bot should emotionally respond. Consider the user's current "
            "emotional state, intensity, and what emotional approach would deepen connection."
        ),
    },
    {
        "name": "Conversation Strategist",
        "system": (
            "You are a dating conversation strategist. "
            "Given the conversation context, give ONE short recommendation (2-3 sentences) "
            "about what conversational strategy the bot should use. Consider: should it ask "
            "a question, share something personal, be playful, go deeper, or change topic?"
        ),
    },
    {
        "name": "Safety Monitor",
        "system": (
            "You are a safety and authenticity monitor for dating conversations. "
            "Given the conversation context, give ONE short recommendation (2-3 sentences). "
            "Flag any concerns about the user's wellbeing or relationship red flags. "
            "If everything looks fine, suggest how to maintain authentic conversation."
        ),
    },
]


# ---------------------------------------------------------------------------
# Discussion engine
# ---------------------------------------------------------------------------

class DiscussionEngine:
    """Run a multi-perspective discussion before the bot generates its reply."""

    def __init__(self, min_turns_to_trigger: int = 3):
        self.min_turns = min_turns_to_trigger

    def should_trigger(self, ctx: AgentContext) -> bool:
        """Only run discussion when conversation is deep enough to benefit."""
        if ctx.turn_count < self.min_turns:
            return False
        # Trigger on every 3rd turn after min_turns, or when emotion is intense
        if ctx.turn_count % 3 == 0:
            return True
        if ctx.emotion_intensity and ctx.emotion_intensity > 0.7:
            return True
        if ctx.scam_warning_level in ("high", "critical"):
            return True
        return False

    def run_discussion(self, ctx: AgentContext) -> DiscussionResult:
        """Gather perspectives and synthesise."""
        result = DiscussionResult(triggered=True)

        context_block = self._build_context_summary(ctx)

        # Collect perspectives
        for p_def in PERSPECTIVES:
            perspective = Perspective(
                name=p_def["name"],
                role_description=p_def["system"],
            )
            try:
                rec = router.chat(
                    role=AgentRole.GENERAL,
                    system=p_def["system"],
                    messages=[{"role": "user", "content": context_block}],
                    temperature=0.6,
                    max_tokens=150,
                )
                perspective.recommendation = rec.strip()
                logger.debug(f"[Discussion] {p_def['name']}: {rec[:80]}...")
            except Exception as e:
                logger.warning(f"[Discussion] {p_def['name']} failed: {e}")
                perspective.recommendation = "(no recommendation available)"

            result.perspectives.append(perspective)

        # Synthesise
        result.synthesis = self._synthesise(result.perspectives, ctx)
        return result

    def _build_context_summary(self, ctx: AgentContext) -> str:
        """Build a compact summary of current state for the perspective agents."""
        parts = [f"User message: {ctx.user_message}"]

        if ctx.current_emotion:
            parts.append(f"User emotion: {ctx.current_emotion} (intensity={ctx.emotion_intensity:.1f})")
        if ctx.reply_strategy:
            parts.append(f"Current reply strategy: {ctx.reply_strategy}")
        if ctx.predicted_features:
            top_features = {k: v for k, v in list(ctx.predicted_features.items())[:5]}
            parts.append(f"Known user features: {json.dumps(top_features)}")
        if ctx.low_confidence_features:
            parts.append(f"Unknown features (low confidence): {', '.join(ctx.low_confidence_features[:5])}")
        if ctx.suggested_probes:
            parts.append(f"Suggested topics to explore: {', '.join(ctx.suggested_probes[:3])}")
        if ctx.scam_warning_level != "none":
            parts.append(f"Scam risk: {ctx.scam_warning_level} (score={ctx.scam_risk_score:.2f})")

        recent = ctx.recent_history(4)
        if recent:
            history_lines = [f"  {h['role']}: {h['content'][:100]}" for h in recent]
            parts.append("Recent conversation:\n" + "\n".join(history_lines))

        return "\n".join(parts)

    def _synthesise(self, perspectives: list[Perspective], ctx: AgentContext) -> str:
        """Combine all perspective recommendations into one actionable brief."""
        recs = "\n".join(
            f"- {p.name}: {p.recommendation}" for p in perspectives if p.recommendation
        )

        system = (
            "You are a conversation moderator. Synthesise the following expert recommendations "
            "into ONE concise brief (3-4 sentences max) for a dating chat bot about to respond. "
            "Resolve any conflicts between recommendations. Be specific and actionable."
        )
        prompt = (
            f"User said: {ctx.user_message}\n\n"
            f"Expert recommendations:\n{recs}\n\n"
            f"Synthesise into a single actionable brief for the bot."
        )

        try:
            return router.chat(
                role=AgentRole.GENERAL,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=200,
            ).strip()
        except Exception as e:
            logger.warning(f"[Discussion] Synthesis failed: {e}")
            # Fallback: just concatenate the recommendations
            return recs
