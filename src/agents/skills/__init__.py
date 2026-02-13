"""Composable skill modules for dating conversation bots."""

from src.agents.skills.registry import Skill, SkillResult, SkillRegistry, skill_registry
from src.agents.skills.builtin import register_builtin_skills

__all__ = [
    "Skill", "SkillResult", "SkillRegistry", "skill_registry",
    "register_builtin_skills",
]
