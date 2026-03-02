"""DigitalTwinAgent -- creates an AI clone based on inferred user personality."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict

from loguru import logger

from src.agents.llm_router import router, AgentRole

if TYPE_CHECKING:
    from src.agents.agent_context import AgentContext


BIG_FIVE_TRAITS = [
    "big_five_openness",
    "big_five_conscientiousness",
    "big_five_extraversion",
    "big_five_agreeableness",
    "big_five_neuroticism",
]


class DigitalTwinAgent:
    """Builds a digital-twin personality clone from predicted features and
    provides chat-as-twin and perception-comparison capabilities."""

    # ------------------------------------------------------------------ #
    # Twin creation
    # ------------------------------------------------------------------ #

    def create_twin(self, ctx: AgentContext) -> dict:
        """Create a twin profile from the current predicted features.

        Writes the result to *ctx.digital_twin* and sets
        *ctx.digital_twin_created = True*.

        Returns:
            The twin profile dict.
        """
        features = ctx.predicted_features
        confidences = ctx.feature_confidences

        # --- Personality summary from Big Five ---
        b5_parts: list[str] = []
        for trait_key in BIG_FIVE_TRAITS:
            value = features.get(trait_key)
            conf = confidences.get(trait_key, 0.0)
            if value is not None:
                label = trait_key.replace("big_five_", "").capitalize()
                b5_parts.append(f"{label}: {value} (conf {conf:.0%})")
        personality_summary = "; ".join(b5_parts) if b5_parts else "Not enough data yet"

        # --- Interests ---
        interests: list[str] = [
            k.replace("interest_", "").replace("_", " ").title()
            for k, v in features.items()
            if k.startswith("interest_") and v and v not in (0, 0.0, "0", "none", False)
        ]

        # --- Communication style ---
        comm_style = features.get("communication_style", "casual")

        # --- Optional fields ---
        mbti = features.get("mbti") or features.get("mbti_type")
        attachment = features.get("attachment_style")

        twin_profile: Dict[str, Any] = {
            "name": f"Twin-{ctx.user_id[:8]}",
            "personality_summary": personality_summary,
            "interests": interests or ["general chat"],
            "communication_style": comm_style,
            "mbti": mbti,
            "attachment_style": attachment,
            "source_turn": ctx.turn_count,
        }

        ctx.digital_twin = twin_profile
        ctx.digital_twin_created = True
        logger.info(f"[DigitalTwin] Created twin for {ctx.user_id} at turn {ctx.turn_count}")
        return twin_profile

    # ------------------------------------------------------------------ #
    # Chat with twin
    # ------------------------------------------------------------------ #

    async def chat_with_twin(self, twin_profile: dict, message: str) -> str:
        """Use the LLM to role-play as the described personality twin.

        Args:
            twin_profile: The twin profile dict (from *create_twin*).
            message: The user's message directed at the twin.

        Returns:
            The twin's response string.
        """
        system_prompt = (
            "You are a digital twin -- an AI clone of a real person.\n"
            "Role-play AS this person based on the personality profile below.\n"
            "Stay in character. Keep responses concise (1-3 sentences).\n\n"
            f"Name: {twin_profile.get('name', 'Twin')}\n"
            f"Personality (Big Five): {twin_profile.get('personality_summary', 'unknown')}\n"
            f"Interests: {', '.join(twin_profile.get('interests', []))}\n"
            f"Communication style: {twin_profile.get('communication_style', 'casual')}\n"
        )
        if twin_profile.get("mbti"):
            system_prompt += f"MBTI: {twin_profile['mbti']}\n"
        if twin_profile.get("attachment_style"):
            system_prompt += f"Attachment style: {twin_profile['attachment_style']}\n"

        messages = [{"role": "user", "content": message}]

        try:
            response = await asyncio.to_thread(
                router.chat,
                role=AgentRole.PERSONA,
                system=system_prompt,
                messages=messages,
                temperature=0.8,
                max_tokens=200,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"[DigitalTwin] chat_with_twin failed: {e}")
            return "Hmm, I'm not sure how to respond to that right now."

    # ------------------------------------------------------------------ #
    # Perception comparison
    # ------------------------------------------------------------------ #

    def compare_perception(
        self, ctx: AgentContext, user_perception: dict
    ) -> dict:
        """Compare user's self-reported perception with predicted features.

        Args:
            ctx: The agent context (holds predicted_features).
            user_perception: Dict of {dimension: float} slider values (0-1).

        Returns:
            Dict with per-dimension comparison and overall_match_rate.
        """
        predicted = ctx.predicted_features
        comparison: Dict[str, Dict[str, Any]] = {}
        match_count = 0
        total = 0

        for dimension, user_val in user_perception.items():
            pred_val = predicted.get(dimension)
            if pred_val is None:
                continue

            # Normalise both to float for comparison
            try:
                pred_float = float(pred_val)
                user_float = float(user_val)
            except (ValueError, TypeError):
                continue

            total += 1
            is_match = abs(pred_float - user_float) <= 0.2  # tolerance band
            if is_match:
                match_count += 1

            comparison[dimension] = {
                "predicted": round(pred_float, 3),
                "user_input": round(user_float, 3),
                "match": is_match,
            }

        overall = (match_count / total) if total > 0 else 0.0

        logger.info(
            f"[DigitalTwin] Perception comparison: {match_count}/{total} match "
            f"({overall:.0%}) for {ctx.user_id}"
        )

        return {
            "dimensions": comparison,
            "overall_match_rate": round(overall, 3),
            "matched": match_count,
            "total": total,
        }
