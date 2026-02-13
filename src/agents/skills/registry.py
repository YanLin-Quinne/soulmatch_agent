"""Skill registry — define, register, and match composable conversation skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from loguru import logger


@dataclass
class SkillResult:
    """Result of a skill execution."""
    activated: bool = False
    skill_name: str = ""
    output: str = ""             # text to inject into context or append
    prompt_addition: str = ""    # extra system prompt guidance
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Skill:
    """A composable skill module."""
    name: str
    description: str
    # Trigger function: (user_message, ctx) -> bool
    trigger: Callable[..., bool] = lambda msg, ctx: False
    # Execute function: (user_message, ctx) -> SkillResult
    execute: Callable[..., SkillResult] = lambda msg, ctx: SkillResult()
    # Priority (higher = checked first)
    priority: int = 0
    # Whether this skill is exclusive (blocks other skills from running)
    exclusive: bool = False


class SkillRegistry:
    """Manages available skills."""

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill):
        self._skills[skill.name] = skill
        logger.debug(f"Skill registered: {skill.name}")

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def list_skills(self) -> list[Skill]:
        return sorted(self._skills.values(), key=lambda s: -s.priority)

    def match_skills(self, message: str, ctx: Any) -> list[Skill]:
        """Return all skills whose trigger matches the current context."""
        matched = []
        for skill in self.list_skills():
            try:
                if skill.trigger(message, ctx):
                    matched.append(skill)
                    if skill.exclusive:
                        break
            except Exception as e:
                logger.warning(f"Skill trigger error ({skill.name}): {e}")
        return matched

    def execute_matched(self, message: str, ctx: Any) -> list[SkillResult]:
        """Match and execute all relevant skills."""
        matched = self.match_skills(message, ctx)
        results = []
        for skill in matched:
            try:
                result = skill.execute(message, ctx)
                result.skill_name = skill.name
                result.activated = True
                results.append(result)
                logger.info(f"Skill activated: {skill.name}")
            except Exception as e:
                logger.error(f"Skill execution error ({skill.name}): {e}")
        return results


# Module-level singleton
skill_registry = SkillRegistry()
