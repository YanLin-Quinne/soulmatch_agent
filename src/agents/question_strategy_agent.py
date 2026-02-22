"""Question Strategy Agent — generates structured conversation strategies with hint types."""

import json
from typing import Dict, List
from loguru import logger

from src.agents.llm_router import router, AgentRole


class QuestionStrategyAgent:
    """
    Given low-confidence feature keys and conversation history, suggests 1-3 natural
    conversation strategies with different approach types:
    - direct_question: A casual but direct question
    - hint: An indirect hint that invites the user to share
    - self_disclosure: Bot shares about itself first to encourage reciprocity
    - topic_shift: A natural topic transition that leads toward the target trait
    """

    def suggest_probes(
        self,
        low_confidence_features: List[str],
        conversation_history: List[dict],
        max_probes: int = 3,
    ) -> List[str]:
        """Legacy interface: returns flat list of probe strings."""
        hints = self.suggest_hints(low_confidence_features, conversation_history, max_probes)
        return [h["text"] for h in hints]

    def suggest_hints(
        self,
        low_confidence_features: List[str],
        conversation_history: List[dict],
        max_hints: int = 3,
    ) -> List[Dict[str, str]]:
        """Return structured hints: [{"type": "hint|self_disclosure|...", "text": "..."}]"""
        if not low_confidence_features:
            return []

        recent = "\n".join(
            f"{m['speaker']}: {m['message']}" for m in conversation_history[-6:]
        )

        prompt = (
            f"We're chatting on a dating app and trying to naturally learn about the user.\n\n"
            f"Recent conversation:\n{recent}\n\n"
            f"We still need to learn about these traits: {', '.join(low_confidence_features[:8])}\n\n"
            f"Suggest {max_hints} conversation strategies using DIFFERENT approach types:\n"
            f"- direct_question: A casual question (not interview-like)\n"
            f"- hint: An indirect comment that invites sharing (e.g. 'I've been thinking about...')\n"
            f"- self_disclosure: Share something about yourself first to encourage reciprocity\n"
            f"- topic_shift: A natural segue to a new topic related to the trait\n\n"
            f"Use at least 2 different types. Keep everything casual — dating chat, NOT interview.\n\n"
            f'Respond with JSON: {{"hints":[{{"type":"hint","text":"..."}},{{"type":"self_disclosure","text":"..."}}]}}'
        )

        try:
            text = router.chat(
                role=AgentRole.QUESTION,
                system="You are a conversational strategist for a dating app.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
                json_mode=True,
            )
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            data = json.loads(text)
            hints = data.get("hints", [])
            # Validate structure
            valid = []
            for h in hints[:max_hints]:
                if isinstance(h, dict) and "text" in h:
                    h.setdefault("type", "direct_question")
                    valid.append(h)
                elif isinstance(h, str):
                    valid.append({"type": "direct_question", "text": h})
            return valid
        except Exception as e:
            logger.error(f"Question strategy failed: {e}")
            return []
