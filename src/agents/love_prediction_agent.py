"""LovePredictionAgent -- predicts detailed love-relationship dynamics when rel_type is 'love'.

Produces: love_stage, compatibility dimensions, blockers, catalysts, advice.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Dict

from loguru import logger

from src.agents.llm_router import router, AgentRole

if TYPE_CHECKING:
    from src.agents.agent_context import AgentContext

LOVE_STAGES = ["attraction", "exploration", "bonding", "commitment", "deep_attachment"]

_FALLBACK: Dict[str, Any] = {
    "love_stage": "attraction",
    "stage_confidence": 0.3,
    "compatibility": {
        "emotional": 0.5,
        "intellectual": 0.5,
        "lifestyle": 0.5,
        "values": 0.5,
    },
    "blockers": ["insufficient conversation data"],
    "catalysts": ["continued engagement"],
    "advice": "Keep the conversation going to learn more about each other.",
    "can_progress": False,
}

BACKTICK3 = chr(96) * 3
_FENCE_RE = re.compile(BACKTICK3 + r"\w*\n(.*?)" + BACKTICK3, re.DOTALL)


class LovePredictionAgent:
    """Predict love-specific relationship details via LLM."""

    async def predict(self, ctx: AgentContext) -> Dict[str, Any]:
        """Run love-relationship prediction and write result to ctx.love_prediction."""
        recent = ctx.recent_history(20)
        if len(recent) < 4:
            ctx.love_prediction = dict(_FALLBACK)
            return ctx.love_prediction

        history_text = "\n".join(
            f"{m['speaker']}: {m['message']}" for m in recent
        )

        rel_status = ctx.rel_status or "stranger"
        rel_type = ctx.rel_type or "other"
        sentiment = ctx.sentiment_label or "neutral"

        prompt = (
            "You are a relationship psychologist AI.\n\n"
            f"Current relationship status: {rel_status}\n"
            f"Relationship type: {rel_type}\n"
            f"Overall sentiment: {sentiment}\n\n"
            f"Recent conversation:\n{history_text}\n\n"
            "Analyze this romantic relationship and return JSON with EXACTLY these fields:\n"
            '{\n'
            '  "love_stage": one of ["attraction","exploration","bonding","commitment","deep_attachment"],\n'
            '  "stage_confidence": float 0-1,\n'
            '  "compatibility": {"emotional": 0-1, "intellectual": 0-1, "lifestyle": 0-1, "values": 0-1},\n'
            '  "blockers": ["string list of what prevents advancement"],\n'
            '  "catalysts": ["string list of what helps the relationship"],\n'
            '  "advice": "one actionable suggestion",\n'
            '  "can_progress": bool\n'
            '}\n'
            "Be concise. Return valid JSON only."
        )

        try:
            text = router.chat(
                role=AgentRole.FEATURE,
                system="You are a relationship psychologist. Respond with valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=400,
                json_mode=True,
            )
            text = text.strip()
            # Strip markdown code fences if present
            fence_match = _FENCE_RE.search(text)
            if fence_match:
                text = fence_match.group(1).strip()

            data = json.loads(text)
            result = self._validate(data)
        except Exception as e:
            logger.error(f"[LovePredictionAgent] LLM call failed: {e}")
            result = dict(_FALLBACK)

        ctx.love_prediction = result
        return result

    # ------------------------------------------------------------------
    # Validation / sanitisation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitise LLM output into a well-typed dict with fallback defaults."""
        stage = raw.get("love_stage", "attraction")
        if stage not in LOVE_STAGES:
            stage = "attraction"

        stage_conf = float(raw.get("stage_confidence", 0.5))
        stage_conf = max(0.0, min(1.0, stage_conf))

        compat_raw = raw.get("compatibility", {})
        compatibility = {}
        for dim in ("emotional", "intellectual", "lifestyle", "values"):
            val = float(compat_raw.get(dim, 0.5))
            compatibility[dim] = max(0.0, min(1.0, val))

        blockers = raw.get("blockers", [])
        if not isinstance(blockers, list):
            blockers = [str(blockers)]
        blockers = [str(b) for b in blockers[:5]]

        catalysts = raw.get("catalysts", [])
        if not isinstance(catalysts, list):
            catalysts = [str(catalysts)]
        catalysts = [str(c) for c in catalysts[:5]]

        advice = str(raw.get("advice", _FALLBACK["advice"]))
        can_progress = bool(raw.get("can_progress", False))

        return {
            "love_stage": stage,
            "stage_confidence": round(stage_conf, 2),
            "compatibility": compatibility,
            "blockers": blockers,
            "catalysts": catalysts,
            "advice": advice,
            "can_progress": can_progress,
        }
