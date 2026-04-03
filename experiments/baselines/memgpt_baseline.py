"""Simplified MemGPT-style baseline with paged working and disk memory."""

import math
import re
import time
from collections import Counter
from typing import Dict, List, Optional

from loguru import logger

from src.agents.llm_router import router, AgentRole

from .utils import (
    format_dialogue,
    parse_json_response,
    PERSONALITY_PROMPT_TEMPLATE,
    RELATIONSHIP_PROMPT_TEMPLATE,
)

__all__ = ["MemGPTBaseline"]

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "she",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "they",
    "this",
    "to",
    "us",
    "we",
    "were",
    "with",
    "you",
    "your",
    "yours",
}


def _tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9']+", (text or "").lower())
        if len(token) > 2 and token not in _STOPWORDS
    ]


def _cosine_similarity(text_a: str, text_b: str) -> float:
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0

    counts_a = Counter(tokens_a)
    counts_b = Counter(tokens_b)
    shared = set(counts_a) & set(counts_b)
    numerator = sum(counts_a[token] * counts_b[token] for token in shared)
    norm_a = math.sqrt(sum(value * value for value in counts_a.values()))
    norm_b = math.sqrt(sum(value * value for value in counts_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return numerator / (norm_a * norm_b)


class MemGPTBaseline:
    """Simplified MemGPT-style baseline.

    The active context window acts as "main memory" while compressed summaries
    of older turns act as "disk" memory. Predictions page old content out via
    summarization and page relevant summaries back in for the final prompt.
    """

    def __init__(self, context_window: int = 10, llm_router=None):
        self.context_window = context_window
        self.router = llm_router or router
        self.disk_memory: List[str] = []

    def _page_out(self, old_turns: List[Dict]) -> str:
        """Summarize old turns and store the result in disk memory."""
        if not old_turns:
            return ""

        prompt = f"""Compress the following conversation turns into a durable memory summary.

Conversation:
{format_dialogue(old_turns)}

Focus on facts that remain useful later:
- stable personality clues
- preferences, habits, and values
- recurring emotional patterns
- relationship signals or milestones

Return valid JSON only:
{{
  "summary": "2-4 sentence summary capturing the durable information."
}}"""

        try:
            response = self.router.chat(
                role=AgentRole.MEMORY,
                system="You are a memory manager. Compress conversation history into grounded durable summaries. Output valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300,
                json_mode=True,
            )
            result = parse_json_response(response)
            summary = str(result.get("summary", "")).strip()
            if summary:
                return summary
        except Exception as e:
            logger.warning(f"[MemGPTBaseline] page-out failed, using fallback summary: {e}")

        fallback = format_dialogue(old_turns)
        return fallback[:500] + ("..." if len(fallback) > 500 else "")

    def _page_in(self, query: str, top_k: Optional[int] = None) -> str:
        """Retrieve relevant summaries from disk memory."""
        if not self.disk_memory:
            return ""

        scored = []
        for idx, summary in enumerate(self.disk_memory):
            score = _cosine_similarity(query, summary)
            if score > 0:
                scored.append((score, idx, summary))

        if not scored:
            fallback_summaries = self.disk_memory[-min(3, len(self.disk_memory)) :]
            return "\n".join(
                f"[Disk Memory {i + 1}] {summary}" for i, summary in enumerate(fallback_summaries)
            )

        scored.sort(reverse=True)
        limit = top_k or min(5, len(scored))
        selected = [summary for _, _, summary in scored[:limit]]
        return "\n".join(
            f"[Disk Memory {i + 1}] {summary}" for i, summary in enumerate(selected)
        )

    def _run_memory_management(self, dialogue: List[Dict]) -> List[Dict]:
        """Process turns incrementally and page old content to disk."""
        self.disk_memory = []
        main_memory: List[Dict] = []
        page_size = max(1, self.context_window // 2)

        for turn in dialogue:
            main_memory.append(turn)
            if len(main_memory) > self.context_window:
                page_out_turns = main_memory[:page_size]
                summary = self._page_out(page_out_turns)
                if summary:
                    self.disk_memory.append(summary)
                main_memory = main_memory[page_size:]

        return main_memory

    def predict_personality(self, dialogue: List[Dict]) -> Optional[Dict]:
        """Process dialogue with MemGPT-style memory management, then predict."""
        main_memory = self._run_memory_management(dialogue)
        active_context = format_dialogue(main_memory)
        retrieved = self._page_in(
            "personality traits preferences communication style habits values emotional tendencies "
            + active_context
        )

        sections = []
        if retrieved:
            sections.append("Paged-in long-term memory summaries:\n" + retrieved)
        sections.append("Active context window:\n" + active_context)
        prompt = PERSONALITY_PROMPT_TEMPLATE.format(dialogue="\n\n".join(sections))

        t0 = time.time()
        try:
            response = self.router.chat(
                role=AgentRole.FEATURE,
                system="You are a personality analysis expert. Use the active context and paged-in memory summaries. Output valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
                json_mode=True,
            )
        except Exception as e:
            logger.error(f"[MemGPTBaseline] personality prediction failed: {e}")
            return None
        elapsed = time.time() - t0

        result = parse_json_response(response)
        if not result:
            logger.warning("[MemGPTBaseline] failed to parse personality response")
            return None

        result["elapsed_seconds"] = round(elapsed, 3)
        result["method"] = "memgpt_baseline"
        return result

    def predict_relationship(
        self, dialogue: List[Dict], context: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Predict relationship type using paged working and disk memory."""
        main_memory = self._run_memory_management(dialogue)
        active_context = format_dialogue(main_memory)
        retrieved = self._page_in(
            "relationship dynamics closeness status affection conflict commitment shared history "
            + active_context
        )

        context_parts = []
        if retrieved:
            context_parts.append("Paged-in long-term memory summaries:\n" + retrieved)
        if context:
            context_parts.append(f"Additional context:\n{context}")

        prompt = RELATIONSHIP_PROMPT_TEMPLATE.format(
            dialogue="Active context window:\n" + active_context,
            context_section="\n\n".join(context_parts),
        )

        t0 = time.time()
        try:
            response = self.router.chat(
                role=AgentRole.GENERAL,
                system="You are a relationship analysis expert. Use the active context and paged-in memory summaries. Output valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
                json_mode=True,
            )
        except Exception as e:
            logger.error(f"[MemGPTBaseline] relationship prediction failed: {e}")
            return None
        elapsed = time.time() - t0

        result = parse_json_response(response)
        if not result:
            logger.warning("[MemGPTBaseline] failed to parse relationship response")
            return None

        result["elapsed_seconds"] = round(elapsed, 3)
        result["method"] = "memgpt_baseline"
        return result
