"""Built-in skills for dating conversation bots.

Skills:
  - ice_breaker: Fun conversation starters when things go quiet
  - deep_dive: Probe deeper on meaningful topics
  - flirt_mode: Playful/flirty responses when mood is right
  - comfort: Supportive responses when user is sad/stressed
  - date_planner: Suggest date ideas based on shared interests
  - boundary_respect: De-escalate when user sets boundaries
"""

from __future__ import annotations

import re
from typing import Any

from src.agents.skills.registry import Skill, SkillResult, skill_registry
from src.agents.llm_router import router, AgentRole


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _keywords_in(message: str, keywords: list[str]) -> bool:
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in keywords)


# ---------------------------------------------------------------------------
# Ice Breaker
# ---------------------------------------------------------------------------

def _ice_breaker_trigger(message: str, ctx: Any) -> bool:
    if not hasattr(ctx, 'turn_count'):
        return False
    # Trigger on short or low-energy messages after initial turns
    if ctx.turn_count >= 3 and len(message.split()) <= 3:
        return True
    if _keywords_in(message, ["bored", "idk", "dunno", "meh", "nothing much"]):
        return True
    return False


def _ice_breaker_execute(message: str, ctx: Any) -> SkillResult:
    return SkillResult(
        output="",
        prompt_addition=(
            "[Skill: Ice Breaker] The conversation energy is low. "
            "Inject something fun, surprising, or playful to re-energize. "
            "Try a 'would you rather' question, a fun hypothetical, or share "
            "an interesting personal anecdote. Keep it light and engaging."
        ),
    )


# ---------------------------------------------------------------------------
# Deep Dive
# ---------------------------------------------------------------------------

def _deep_dive_trigger(message: str, ctx: Any) -> bool:
    if len(message.split()) < 15:
        return False
    # Long, thoughtful messages suggest the user wants depth
    deep_keywords = ["actually", "honestly", "i think", "i feel", "important to me",
                     "believe", "value", "dream", "hope", "fear", "struggle"]
    return _keywords_in(message, deep_keywords)


def _deep_dive_execute(message: str, ctx: Any) -> SkillResult:
    return SkillResult(
        output="",
        prompt_addition=(
            "[Skill: Deep Dive] The user is sharing something meaningful. "
            "Respond with genuine depth — acknowledge what they shared, "
            "relate with your own perspective, and ask a follow-up that shows "
            "you truly listened. Avoid surface-level reactions."
        ),
    )


# ---------------------------------------------------------------------------
# Flirt Mode
# ---------------------------------------------------------------------------

def _flirt_trigger(message: str, ctx: Any) -> bool:
    flirt_keywords = ["cute", "attractive", "handsome", "beautiful", "flirt",
                      "charming", "wink", "😉", "😏", "date", "miss you"]
    if _keywords_in(message, flirt_keywords):
        return True
    if hasattr(ctx, 'current_emotion') and ctx.current_emotion in ("joy", "love", "excitement"):
        if hasattr(ctx, 'emotion_intensity') and ctx.emotion_intensity > 0.6:
            return True
    return False


def _flirt_execute(message: str, ctx: Any) -> SkillResult:
    return SkillResult(
        output="",
        prompt_addition=(
            "[Skill: Flirt Mode] The vibe is playful and flirty. "
            "Be charming and witty. Use light teasing, compliments, "
            "or playful banter. Keep it tasteful and fun, never pushy."
        ),
    )


# ---------------------------------------------------------------------------
# Comfort
# ---------------------------------------------------------------------------

def _comfort_trigger(message: str, ctx: Any) -> bool:
    sad_keywords = ["sad", "depressed", "stressed", "anxious", "lonely",
                    "upset", "crying", "tough day", "bad day", "tired of"]
    if _keywords_in(message, sad_keywords):
        return True
    if hasattr(ctx, 'current_emotion') and ctx.current_emotion in ("sadness", "fear", "anxiety", "anger"):
        if hasattr(ctx, 'emotion_intensity') and ctx.emotion_intensity > 0.5:
            return True
    return False


def _comfort_execute(message: str, ctx: Any) -> SkillResult:
    return SkillResult(
        output="",
        prompt_addition=(
            "[Skill: Comfort] The user is going through something difficult. "
            "Be warm, empathetic, and supportive. Validate their feelings first, "
            "then gently offer perspective or distraction only if appropriate. "
            "Don't minimize their experience or rush to fix things."
        ),
    )


# ---------------------------------------------------------------------------
# Date Planner
# ---------------------------------------------------------------------------

def _date_planner_trigger(message: str, ctx: Any) -> bool:
    date_keywords = ["meet up", "hang out", "get together", "go out",
                     "date idea", "where should we", "what should we do",
                     "plans", "this weekend"]
    return _keywords_in(message, date_keywords)


def _date_planner_execute(message: str, ctx: Any) -> SkillResult:
    interests = []
    if hasattr(ctx, 'predicted_features') and ctx.predicted_features:
        interests = [k.replace("interest_", "") for k in ctx.predicted_features
                     if k.startswith("interest_") and ctx.predicted_features[k] > 0.5]

    interest_hint = f" User interests: {', '.join(interests)}." if interests else ""

    return SkillResult(
        output="",
        prompt_addition=(
            f"[Skill: Date Planner] The user is interested in meeting up or planning a date. "
            f"Suggest creative, specific date ideas that match the conversation vibe.{interest_hint} "
            f"Be enthusiastic but not overeager. Suggest 1-2 concrete ideas."
        ),
    )


# ---------------------------------------------------------------------------
# Boundary Respect
# ---------------------------------------------------------------------------

def _boundary_trigger(message: str, ctx: Any) -> bool:
    boundary_keywords = ["stop", "don't", "not comfortable", "too much",
                         "slow down", "back off", "not ready", "boundaries",
                         "personal", "private", "rather not"]
    return _keywords_in(message, boundary_keywords)


def _boundary_execute(message: str, ctx: Any) -> SkillResult:
    return SkillResult(
        output="",
        prompt_addition=(
            "[Skill: Boundary Respect] The user is setting a boundary. "
            "Immediately respect it — acknowledge what they said, apologize if needed, "
            "and redirect the conversation to a comfortable topic. "
            "Do NOT push back, question their boundary, or try to convince them."
        ),
        metadata={"priority": "high"},
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_builtin_skills():
    """Register all built-in skills."""
    skills = [
        Skill(name="boundary_respect", description="Respect user boundaries",
              trigger=_boundary_trigger, execute=_boundary_execute,
              priority=100, exclusive=True),

        Skill(name="comfort", description="Support user through difficult moments",
              trigger=_comfort_trigger, execute=_comfort_execute, priority=80),

        Skill(name="ice_breaker", description="Re-energize low-energy conversations",
              trigger=_ice_breaker_trigger, execute=_ice_breaker_execute, priority=50),

        Skill(name="deep_dive", description="Go deeper on meaningful topics",
              trigger=_deep_dive_trigger, execute=_deep_dive_execute, priority=60),

        Skill(name="flirt_mode", description="Playful flirty responses",
              trigger=_flirt_trigger, execute=_flirt_execute, priority=40),

        Skill(name="date_planner", description="Suggest date ideas",
              trigger=_date_planner_trigger, execute=_date_planner_execute, priority=30),
    ]

    for skill in skills:
        skill_registry.register(skill)
