"""No-memory baseline that predicts from only the most recent turns."""

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

__all__ = ["NoMemoryBaseline"]


class NoMemoryBaseline:
    """Stateless baseline that only uses the last N turns."""

    def __init__(self, last_n_turns: int = 6, llm_router=None):
        self.last_n_turns = last_n_turns
        self.router = llm_router or router

    def predict_personality(self, dialogue: List[Dict]) -> Optional[Dict]:
        """Predict personality from only the last N turns."""
        recent = dialogue[-self.last_n_turns :]
        prompt = PERSONALITY_PROMPT_TEMPLATE.format(dialogue=format_dialogue(recent))

        t0 = time.time()
        try:
            response = self.router.chat(
                role=AgentRole.FEATURE,
                system="You are a personality analysis expert. Use only the recent conversation turns shown. Output valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
                json_mode=True,
            )
        except Exception as e:
            logger.error(f"[NoMemoryBaseline] personality prediction failed: {e}")
            return None
        elapsed = time.time() - t0

        result = parse_json_response(response)
        if not result:
            logger.warning("[NoMemoryBaseline] failed to parse personality response")
            return None

        result["elapsed_seconds"] = round(elapsed, 3)
        result["method"] = "no_memory_baseline"
        return result

    def predict_relationship(
        self, dialogue: List[Dict], context: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Predict relationship from only the last N turns."""
        recent = dialogue[-self.last_n_turns :]
        context_section = ""
        if context:
            context_section = f"Additional context:\n{context}"

        prompt = RELATIONSHIP_PROMPT_TEMPLATE.format(
            dialogue=format_dialogue(recent),
            context_section=context_section,
        )

        t0 = time.time()
        try:
            response = self.router.chat(
                role=AgentRole.GENERAL,
                system="You are a relationship analysis expert. Use only the recent conversation turns shown. Output valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
                json_mode=True,
            )
        except Exception as e:
            logger.error(f"[NoMemoryBaseline] relationship prediction failed: {e}")
            return None
        elapsed = time.time() - t0

        result = parse_json_response(response)
        if not result:
            logger.warning("[NoMemoryBaseline] failed to parse relationship response")
            return None

        result["elapsed_seconds"] = round(elapsed, 3)
        result["method"] = "no_memory_baseline"
        return result
