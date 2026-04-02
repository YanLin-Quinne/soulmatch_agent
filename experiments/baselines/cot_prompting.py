"""Chain-of-thought prompting baseline: structured step-by-step reasoning."""

import time
from typing import Dict, List, Optional

from loguru import logger

from src.agents.llm_router import router, AgentRole

from .utils import format_dialogue, parse_json_response

__all__ = ["CoTBaseline"]

# ---------------------------------------------------------------------------
# CoT-specific prompt templates
# ---------------------------------------------------------------------------

COT_PERSONALITY_PROMPT = """You are a personality analysis expert. Given the following conversation, infer the personality profile of the speakers.

Conversation:
{dialogue}

Let's think step by step:

Step 1 - Identify behavioral patterns: Examine the speakers' word choices, communication style, emotional expressions, and social behaviors in the conversation.

Step 2 - Map to personality dimensions: Based on the behavioral patterns, estimate where each speaker falls on the Big Five personality dimensions (openness, conscientiousness, extraversion, agreeableness, neuroticism) on a 0-1 scale.

Step 3 - Determine MBTI type: Based on the personality dimensions and behavioral evidence, determine the most likely MBTI type (e.g. INFP, ESTJ).

Step 4 - Assign confidence scores: For each dimension and the MBTI type, assign a confidence score (0-1) based on how much evidence supports the assessment.

After your analysis, output your final answer as a JSON object with this exact structure:
{{
  "big_five": {{
    "openness": <float 0-1>,
    "conscientiousness": <float 0-1>,
    "extraversion": <float 0-1>,
    "agreeableness": <float 0-1>,
    "neuroticism": <float 0-1>
  }},
  "mbti": "<4-letter MBTI type>",
  "confidences": {{
    "openness": <float 0-1>,
    "conscientiousness": <float 0-1>,
    "extraversion": <float 0-1>,
    "agreeableness": <float 0-1>,
    "neuroticism": <float 0-1>,
    "mbti": <float 0-1>
  }}
}}

Provide your step-by-step reasoning first, then the final JSON object."""

COT_RELATIONSHIP_PROMPT = """You are a relationship analysis expert. Given the following conversation, predict the relationship between the speakers.

Conversation:
{dialogue}

{context_section}

Let's think step by step:

Step 1 - Analyze communication patterns: Examine the tone, formality, frequency of exchanges, use of nicknames, and conversational dynamics between speakers.

Step 2 - Identify emotional dynamics: Look for signs of emotional intimacy, conflict, care, detachment, flirtation, or familial bonds.

Step 3 - Assess relationship indicators: Based on the communication patterns and emotional dynamics, determine the most likely relationship type (love, friendship, family, other) and status (stranger, acquaintance, crush, dating, committed).

Step 4 - Estimate probabilities: Assign probability distributions over all possible relationship types and statuses.

After your analysis, output your final answer as a JSON object with this exact structure:
{{
  "rel_type": "<one of: love, friendship, family, other>",
  "rel_status": "<one of: stranger, acquaintance, crush, dating, committed>",
  "rel_type_probs": {{
    "love": <float 0-1>,
    "friendship": <float 0-1>,
    "family": <float 0-1>,
    "other": <float 0-1>
  }},
  "rel_status_probs": {{
    "stranger": <float 0-1>,
    "acquaintance": <float 0-1>,
    "crush": <float 0-1>,
    "dating": <float 0-1>,
    "committed": <float 0-1>
  }}
}}

Provide your step-by-step reasoning first, then the final JSON object."""


class CoTBaseline:
    """Baseline that uses chain-of-thought prompting for structured reasoning.

    The LLM is asked to reason step by step before producing its final
    prediction, which typically improves accuracy over direct prompting
    at the cost of higher token usage and latency.
    """

    def __init__(self, llm_router=None):
        self.router = llm_router or router

    def predict_personality(
        self, dialogue: List[Dict], temperature: float = 0.3
    ) -> Optional[Dict]:
        """Infer Big Five traits and MBTI type using chain-of-thought reasoning.

        Args:
            dialogue: List of {speaker, message} dicts.
            temperature: Sampling temperature (default 0.3 for determinism,
                         higher values used by SelfConsistencyBaseline).

        Returns:
            Dict with keys: big_five, mbti, confidences, elapsed_seconds.
            None on failure.
        """
        formatted = format_dialogue(dialogue)
        prompt = COT_PERSONALITY_PROMPT.format(dialogue=formatted)

        t0 = time.time()
        try:
            response = self.router.chat(
                role=AgentRole.FEATURE,
                system="You are a personality analysis expert. Think step by step, then output JSON.",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1500,
                json_mode=False,  # CoT produces reasoning text before JSON
            )
        except Exception as e:
            logger.error(f"[CoTBaseline] personality prediction failed: {e}")
            return None
        elapsed = time.time() - t0

        result = parse_json_response(response)
        if not result:
            logger.warning("[CoTBaseline] failed to parse personality response")
            return None

        result["elapsed_seconds"] = round(elapsed, 3)
        result["method"] = "cot_prompting"
        return result

    def predict_relationship(
        self,
        dialogue: List[Dict],
        context: Optional[Dict] = None,
        temperature: float = 0.3,
    ) -> Optional[Dict]:
        """Predict relationship type and status using chain-of-thought reasoning.

        Args:
            dialogue: List of {speaker, message} dicts.
            context: Optional additional context.
            temperature: Sampling temperature.

        Returns:
            Dict with keys: rel_type, rel_status, rel_type_probs,
            rel_status_probs, elapsed_seconds. None on failure.
        """
        formatted = format_dialogue(dialogue)

        context_section = ""
        if context:
            context_section = f"Additional context:\n{context}"

        prompt = COT_RELATIONSHIP_PROMPT.format(
            dialogue=formatted, context_section=context_section
        )

        t0 = time.time()
        try:
            response = self.router.chat(
                role=AgentRole.GENERAL,
                system="You are a relationship analysis expert. Think step by step, then output JSON.",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1500,
                json_mode=False,
            )
        except Exception as e:
            logger.error(f"[CoTBaseline] relationship prediction failed: {e}")
            return None
        elapsed = time.time() - t0

        result = parse_json_response(response)
        if not result:
            logger.warning("[CoTBaseline] failed to parse relationship response")
            return None

        result["elapsed_seconds"] = round(elapsed, 3)
        result["method"] = "cot_prompting"
        return result
