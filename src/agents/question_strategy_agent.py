"""Question Strategy Agent — generates probing topics to fill low-confidence features."""

import json
from typing import Dict, List
from loguru import logger

from src.agents.llm_router import router, AgentRole


class QuestionStrategyAgent:
    """
    Given low-confidence feature keys and conversation history, suggests 1-3 natural
    conversation topics that would help the bot infer those features.
    """

    def suggest_probes(
        self,
        low_confidence_features: List[str],
        conversation_history: List[dict],
        max_probes: int = 3,
    ) -> List[str]:
        if not low_confidence_features:
            return []

        recent = "\n".join(
            f"{m['speaker']}: {m['message']}" for m in conversation_history[-6:]
        )

        prompt = (
            f"We're chatting on a dating app and trying to naturally learn about the user.\n\n"
            f"Recent conversation:\n{recent}\n\n"
            f"We still need to learn about these traits: {', '.join(low_confidence_features[:8])}\n\n"
            f"Suggest {max_probes} natural, non-intrusive conversation topics or questions that would "
            f"help us infer those traits. Keep them casual — they should feel like normal dating chat, "
            f"NOT an interview.\n\n"
            f'Respond with JSON: {{"probes":["topic 1","topic 2","topic 3"]}}'
        )

        try:
            text = router.chat(
                role=AgentRole.QUESTION,
                system="You are a conversational strategist for a dating app.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
                json_mode=True,
            )
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            data = json.loads(text)
            probes = data.get("probes", [])
            return probes[:max_probes]
        except Exception as e:
            logger.error(f"Question strategy failed: {e}")
            return []
