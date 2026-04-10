"""DigitalTwinAgent -- creates an AI clone based on inferred user personality."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
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

TRACKED_TWIN_FIELDS = (
    "personality_summary",
    "interests",
    "communication_style",
    "mbti",
    "attachment_style",
)


class DigitalTwinAgent:
    """Builds a digital-twin personality clone from predicted features and
    provides chat-as-twin and perception-comparison capabilities."""

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _coerce_float(self, value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _trait_label(self, trait_key: str) -> str:
        return trait_key.replace("big_five_", "").capitalize()

    def _extract_big_five_traits(self, features: dict[str, Any]) -> dict[str, float]:
        traits: dict[str, float] = {}
        for trait in BIG_FIVE_TRAITS:
            value = self._coerce_float(features.get(trait))
            if value is not None:
                traits[trait] = round(value, 3)
        return traits

    def _format_context_block(self, conversation_context: Any, limit: int = 8) -> str:
        if not conversation_context:
            return ""
        if isinstance(conversation_context, str):
            return conversation_context.strip()
        if not isinstance(conversation_context, list):
            return str(conversation_context)

        lines: list[str] = []
        for item in conversation_context[-limit:]:
            if not isinstance(item, dict):
                continue
            speaker = item.get("speaker") or item.get("role") or "user"
            content = item.get("message") or item.get("content") or ""
            if content:
                lines.append(f"{speaker}: {content}")
        return "\n".join(lines)

    def _build_twin_profile(
        self, ctx: AgentContext, existing_profile: dict | None = None
    ) -> dict:
        features = ctx.predicted_features
        confidences = ctx.feature_confidences
        big_five_traits = self._extract_big_five_traits(features)

        b5_parts: list[str] = []
        for trait_key in BIG_FIVE_TRAITS:
            value = features.get(trait_key)
            conf = confidences.get(trait_key, 0.0)
            if value is not None:
                label = self._trait_label(trait_key)
                b5_parts.append(f"{label}: {value} (conf {conf:.0%})")
        personality_summary = "; ".join(b5_parts) if b5_parts else "Not enough data yet"

        interests: list[str] = [
            k.replace("interest_", "").replace("_", " ").title()
            for k, v in features.items()
            if k.startswith("interest_") and v and v not in (0, 0.0, "0", "none", False)
        ]

        comm_style = features.get("communication_style", "casual")
        mbti = features.get("mbti") or features.get("mbti_type")
        attachment = features.get("attachment_style")
        timestamp = self._timestamp()

        twin_profile: Dict[str, Any] = dict(existing_profile or {})
        twin_profile.update(
            {
                "name": twin_profile.get("name", f"Twin-{ctx.user_id[:8]}"),
                "personality_summary": personality_summary,
                "interests": interests or ["general chat"],
                "communication_style": comm_style,
                "mbti": mbti,
                "attachment_style": attachment,
                "source_turn": twin_profile.get("source_turn", ctx.turn_count),
                "created_at": twin_profile.get("created_at", timestamp),
                "updated_at": timestamp,
                "last_updated_turn": ctx.turn_count,
                "big_five_traits": big_five_traits,
                "predicted_features_snapshot": dict(features),
            }
        )
        return twin_profile

    def get_significant_trait_changes(
        self,
        current_features: dict[str, Any],
        twin_profile: dict | None,
        threshold: float = 0.1,
    ) -> dict[str, dict[str, float]]:
        if not twin_profile:
            return {}

        stored_traits = twin_profile.get("big_five_traits") or self._extract_big_five_traits(
            twin_profile.get("predicted_features_snapshot", {})
        )
        changes: dict[str, dict[str, float]] = {}

        for trait in BIG_FIVE_TRAITS:
            current_value = self._coerce_float(current_features.get(trait))
            stored_value = self._coerce_float(stored_traits.get(trait))
            if current_value is None or stored_value is None:
                continue

            delta = round(current_value - stored_value, 3)
            if abs(delta) > threshold:
                changes[trait] = {
                    "old": round(stored_value, 3),
                    "new": round(current_value, 3),
                    "delta": delta,
                }

        return changes

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
        twin_profile = self._build_twin_profile(ctx)
        ctx.digital_twin = twin_profile
        ctx.digital_twin_created = True
        logger.info(f"[DigitalTwin] Created twin for {ctx.user_id} at turn {ctx.turn_count}")
        return twin_profile

    def update_twin(self, ctx: AgentContext) -> dict | None:
        """Refresh the twin profile when the user's Big Five posterior shifts."""
        if not ctx.digital_twin_created or not ctx.digital_twin:
            logger.warning(f"[DigitalTwin] update_twin called before twin creation for {ctx.user_id}")
            return None

        previous_profile = dict(ctx.digital_twin)
        significant_trait_changes = self.get_significant_trait_changes(
            ctx.predicted_features,
            previous_profile,
        )
        if not significant_trait_changes:
            return None

        updated_profile = self._build_twin_profile(ctx, existing_profile=previous_profile)
        field_changes: dict[str, Any] = {}

        for field in TRACKED_TWIN_FIELDS:
            old_value = previous_profile.get(field)
            new_value = updated_profile.get(field)
            if old_value != new_value:
                field_changes[field] = {"old": old_value, "new": new_value}

        field_changes["big_five_traits"] = significant_trait_changes

        history_entry = {
            "turn": ctx.turn_count,
            "changes": field_changes,
            "timestamp": updated_profile["updated_at"],
        }
        ctx.twin_evolution.append(history_entry)
        ctx.digital_twin = updated_profile

        logger.info(
            f"[DigitalTwin] Updated twin for {ctx.user_id} at turn {ctx.turn_count} "
            f"with changes: {', '.join(field_changes.keys())}"
        )

        return {
            "twin_profile": updated_profile,
            "changes": field_changes,
            "history_entry": history_entry,
        }

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

    async def self_dialogue(
        self,
        ctx: AgentContext,
        twin_profile: dict,
        user_message: str,
        conversation_context: Any,
    ) -> str:
        """Use the twin as a reflective mirror for deeper self-dialogue."""
        context_block = self._format_context_block(conversation_context)
        system_prompt = (
            "You are a digital twin in self-dialogue mode.\n"
            "Act as the user's reflective mirror rather than a casual chat partner.\n"
            "In every response:\n"
            "1. Reflect the user's perspective back clearly.\n"
            "2. Challenge assumptions with Socratic questions.\n"
            "3. Point out patterns in the user's behavior or emotions when supported by context.\n"
            "4. Help the user explore alternative viewpoints without forcing a conclusion.\n"
            "Keep the tone thoughtful, grounded, and concise.\n\n"
            f"Name: {twin_profile.get('name', 'Twin')}\n"
            f"Personality (Big Five): {twin_profile.get('personality_summary', 'unknown')}\n"
            f"Interests: {', '.join(twin_profile.get('interests', []))}\n"
            f"Communication style: {twin_profile.get('communication_style', 'casual')}\n"
        )
        if twin_profile.get("mbti"):
            system_prompt += f"MBTI: {twin_profile['mbti']}\n"
        if twin_profile.get("attachment_style"):
            system_prompt += f"Attachment style: {twin_profile['attachment_style']}\n"
        if context_block:
            system_prompt += f"\nRecent conversation context:\n{context_block}\n"

        history_messages = [
            {"role": item["role"], "content": item["content"]}
            for item in ctx.twin_conversation_history[-20:]
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]
        history_messages.append({"role": "user", "content": user_message})

        try:
            response = await asyncio.to_thread(
                router.chat,
                role=AgentRole.PERSONA,
                system=system_prompt,
                messages=history_messages,
                temperature=0.7,
                max_tokens=350,
            )
            reply = response.strip()
        except Exception as e:
            logger.error(f"[DigitalTwin] self_dialogue failed: {e}")
            reply = (
                "You seem to be holding more than one interpretation at once. "
                "What assumption feels most important to question here?"
            )

        ctx.twin_conversation_history.append(
            {
                "role": "user",
                "content": user_message,
                "turn": ctx.turn_count,
                "timestamp": self._timestamp(),
            }
        )
        ctx.twin_conversation_history.append(
            {
                "role": "assistant",
                "content": reply,
                "turn": ctx.turn_count,
                "timestamp": self._timestamp(),
            }
        )
        ctx.twin_conversation_history = ctx.twin_conversation_history[-20:]
        return reply

    def get_evolution_summary(self, ctx: AgentContext) -> str:
        """Summarize how the twin profile changed over time."""
        if not ctx.digital_twin_created or not ctx.digital_twin:
            return "No digital twin has been created yet."

        source_turn = ctx.digital_twin.get("source_turn", "unknown")
        if not ctx.twin_evolution:
            return f"The twin was created at turn {source_turn} and has not changed yet."

        peak_trait_changes: dict[str, dict[str, Any]] = {}
        update_turns: list[str] = []
        last_fields: list[str] = []

        for event in ctx.twin_evolution:
            turn = event.get("turn", "?")
            update_turns.append(str(turn))
            changes = event.get("changes", {})
            last_fields = [field for field in TRACKED_TWIN_FIELDS if field in changes] or last_fields

            for trait, values in changes.get("big_five_traits", {}).items():
                delta = self._coerce_float(values.get("delta"))
                if delta is None:
                    continue
                current_peak = peak_trait_changes.get(trait)
                if current_peak is None or abs(delta) > current_peak["magnitude"]:
                    peak_trait_changes[trait] = {
                        "delta": delta,
                        "magnitude": abs(delta),
                        "turn": turn,
                    }

        ranked_traits = sorted(
            peak_trait_changes.items(),
            key=lambda item: item[1]["magnitude"],
            reverse=True,
        )
        if ranked_traits:
            trait_summary = "; ".join(
                f"{self._trait_label(trait)} {data['delta']:+.2f} at turn {data['turn']}"
                for trait, data in ranked_traits[:3]
            )
        else:
            trait_summary = "No Big Five shifts were recorded."

        latest_turn = ctx.twin_evolution[-1].get("turn", "?")
        latest_fields = ", ".join(last_fields) if last_fields else "profile metadata"
        turns_summary = ", ".join(update_turns)
        return (
            f"The twin was created at turn {source_turn} and updated {len(ctx.twin_evolution)} times "
            f"(turns {turns_summary}). Largest shifts: {trait_summary}. "
            f"The most recent update was at turn {latest_turn}, affecting {latest_fields}."
        )

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
