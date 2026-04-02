"""Direct prompting baseline: single LLM call for personality/relationship prediction."""

import time
from typing import Dict, List, Optional

from loguru import logger

from src.agents.llm_router import router, AgentRole

from .utils import (
    format_dialogue,
    parse_json_response,
    PERSONALITY_PROMPT_TEMPLATE,
    RELATIONSHIP_PROMPT_TEMPLATE,
)

__all__ = ["DirectPromptingBaseline"]


class DirectPromptingBaseline:
    """Baseline that uses a single direct prompt to the LLM.

    No chain-of-thought, no multi-sample aggregation. This represents the
    simplest possible approach and serves as the lower bound for comparison.
    """

    def __init__(self, llm_router=None):
        self.router = llm_router or router

    def predict_personality(self, dialogue: List[Dict]) -> Optional[Dict]:
        """Infer Big Five traits and MBTI type from a conversation.

        Args:
            dialogue: List of {speaker, message} dicts.

        Returns:
            Dict with keys: big_five, mbti, confidences, elapsed_seconds.
            None on failure.
        """
        formatted = format_dialogue(dialogue)
        prompt = PERSONALITY_PROMPT_TEMPLATE.format(dialogue=formatted)

        t0 = time.time()
        try:
            response = self.router.chat(
                role=AgentRole.FEATURE,
                system="You are a personality analysis expert. Output valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
                json_mode=True,
            )
        except Exception as e:
            logger.error(f"[DirectPrompting] personality prediction failed: {e}")
            return None
        elapsed = time.time() - t0

        result = parse_json_response(response)
        if not result:
            logger.warning("[DirectPrompting] failed to parse personality response")
            return None

        result["elapsed_seconds"] = round(elapsed, 3)
        result["method"] = "direct_prompting"
        return result

    def predict_relationship(
        self, dialogue: List[Dict], context: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Predict relationship type and status from a conversation.

        Args:
            dialogue: List of {speaker, message} dicts.
            context: Optional additional context (e.g. user profiles).

        Returns:
            Dict with keys: rel_type, rel_status, rel_type_probs,
            rel_status_probs, elapsed_seconds. None on failure.
        """
        formatted = format_dialogue(dialogue)

        context_section = ""
        if context:
            context_section = f"Additional context:\n{context}"

        prompt = RELATIONSHIP_PROMPT_TEMPLATE.format(
            dialogue=formatted, context_section=context_section
        )

        t0 = time.time()
        try:
            response = self.router.chat(
                role=AgentRole.GENERAL,
                system="You are a relationship analysis expert. Output valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
                json_mode=True,
            )
        except Exception as e:
            logger.error(f"[DirectPrompting] relationship prediction failed: {e}")
            return None
        elapsed = time.time() - t0

        result = parse_json_response(response)
        if not result:
            logger.warning("[DirectPrompting] failed to parse relationship response")
            return None

        result["elapsed_seconds"] = round(elapsed, 3)
        result["method"] = "direct_prompting"
        return result
