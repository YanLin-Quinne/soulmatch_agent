"""Shared utilities and prompt templates for baseline experiments."""

import json
import re
from typing import Dict, List

__all__ = [
    "format_dialogue",
    "parse_json_response",
    "PERSONALITY_PROMPT_TEMPLATE",
    "RELATIONSHIP_PROMPT_TEMPLATE",
    "PERSONALITY_SCHEMA",
    "RELATIONSHIP_SCHEMA",
]

# ---------------------------------------------------------------------------
# Output schemas (for documentation and validation)
# ---------------------------------------------------------------------------

PERSONALITY_SCHEMA = {
    "big_five": {
        "openness": "float 0-1",
        "conscientiousness": "float 0-1",
        "extraversion": "float 0-1",
        "agreeableness": "float 0-1",
        "neuroticism": "float 0-1",
    },
    "mbti": "str, e.g. 'INFP'",
    "confidences": {
        "openness": "float 0-1",
        "conscientiousness": "float 0-1",
        "extraversion": "float 0-1",
        "agreeableness": "float 0-1",
        "neuroticism": "float 0-1",
        "mbti": "float 0-1",
    },
}

RELATIONSHIP_SCHEMA = {
    "rel_type": "one of: love, friendship, family, other",
    "rel_status": "one of: stranger, acquaintance, crush, dating, committed",
    "rel_type_probs": {
        "love": "float 0-1",
        "friendship": "float 0-1",
        "family": "float 0-1",
        "other": "float 0-1",
    },
    "rel_status_probs": {
        "stranger": "float 0-1",
        "acquaintance": "float 0-1",
        "crush": "float 0-1",
        "dating": "float 0-1",
        "committed": "float 0-1",
    },
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PERSONALITY_PROMPT_TEMPLATE = """You are a personality analysis expert. Given the following conversation, infer the personality profile of the speakers.

Conversation:
{dialogue}

Analyze the conversation and output a JSON object with the following structure:
{{
  "big_five": {{
    "openness": <float 0-1>,
    "conscientiousness": <float 0-1>,
    "extraversion": <float 0-1>,
    "agreeableness": <float 0-1>,
    "neuroticism": <float 0-1>
  }},
  "mbti": "<4-letter MBTI type, e.g. INFP>",
  "confidences": {{
    "openness": <float 0-1>,
    "conscientiousness": <float 0-1>,
    "extraversion": <float 0-1>,
    "agreeableness": <float 0-1>,
    "neuroticism": <float 0-1>,
    "mbti": <float 0-1>
  }}
}}

Return ONLY the JSON object, no extra text."""

RELATIONSHIP_PROMPT_TEMPLATE = """You are a relationship analysis expert. Given the following conversation, predict the relationship between the speakers.

Conversation:
{dialogue}

{context_section}

Analyze the conversation and output a JSON object with the following structure:
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

Return ONLY the JSON object, no extra text."""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def format_dialogue(dialogue: List[Dict]) -> str:
    """Format a list of {speaker, message} dicts into readable conversation text.

    Args:
        dialogue: List of dicts, each with 'speaker' and 'message' keys.

    Returns:
        Multi-line string like "Speaker A: Hello\\nSpeaker B: Hi there"
    """
    lines = []
    for turn in dialogue:
        speaker = turn.get("speaker", "Unknown")
        message = turn.get("message", "")
        lines.append(f"{speaker}: {message}")
    return "\n".join(lines)


def parse_json_response(text: str) -> dict:
    """Robustly extract a JSON object from LLM output.

    Handles:
    - Pure JSON strings
    - JSON wrapped in markdown code fences (```json ... ```)
    - JSON embedded in surrounding text

    Args:
        text: Raw LLM output string.

    Returns:
        Parsed dict. Returns empty dict on failure.
    """
    if not text or not isinstance(text, str):
        return {}

    text = text.strip()

    # Strip markdown code fences
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    fence_match = fence_pattern.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first { ... } block (greedy, outermost braces)
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    return {}
