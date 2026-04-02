"""Agent lifecycle hook system — inspired by Claude Code's PreToolUse/PostToolUse pipeline."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

from loguru import logger


class HookAction(Enum):
    """Possible hook outcomes."""
    ALLOW = "allow"
    DENY = "deny"
    MODIFY = "modify"


@dataclass
class HookResult:
    """Result returned by a hook invocation."""
    action: HookAction = HookAction.ALLOW
    reason: str = ""


class HookDeniedError(Exception):
    """Raised when a hook denies agent execution."""
    def __init__(self, agent_name: str, reason: str):
        self.agent_name = agent_name
        self.reason = reason
        super().__init__(f"Hook denied '{agent_name}': {reason}")


@runtime_checkable
class Hook(Protocol):
    """Protocol that all hooks must satisfy."""

    async def on_pre_agent(self, agent_name: str, ctx: Any) -> HookResult:
        """Called before an agent phase runs. Return DENY to skip the phase."""
        ...

    async def on_post_agent(self, agent_name: str, ctx: Any, result: Any) -> HookResult:
        """Called after an agent phase completes."""
        ...


class HookRegistry:
    """Registry for managing agent lifecycle hooks."""

    def __init__(self):
        self._hooks: list[Hook] = []

    def register(self, hook: Hook) -> None:
        self._hooks.append(hook)
        logger.debug(f"Registered hook: {hook.__class__.__name__}")

    def unregister(self, hook: Hook) -> None:
        self._hooks = [h for h in self._hooks if h is not hook]

    @property
    def hooks(self) -> list[Hook]:
        return list(self._hooks)

    async def run_pre_hooks(self, agent_name: str, ctx: Any) -> HookResult:
        """Run all pre-agent hooks. Short-circuits on first DENY."""
        for hook in self._hooks:
            try:
                result = await hook.on_pre_agent(agent_name, ctx)
                if result.action == HookAction.DENY:
                    logger.info(f"Hook {hook.__class__.__name__} denied '{agent_name}': {result.reason}")
                    return result
            except Exception as e:
                logger.error(f"Hook {hook.__class__.__name__}.on_pre_agent failed: {e}")
                # Default to ALLOW on hook error
        return HookResult(action=HookAction.ALLOW)

    async def run_post_hooks(self, agent_name: str, ctx: Any, result: Any) -> HookResult:
        """Run all post-agent hooks."""
        for hook in self._hooks:
            try:
                hook_result = await hook.on_post_agent(agent_name, ctx, result)
                if hook_result.action == HookAction.DENY:
                    logger.warning(f"Post-hook {hook.__class__.__name__} flagged '{agent_name}': {hook_result.reason}")
                    return hook_result
            except Exception as e:
                logger.error(f"Hook {hook.__class__.__name__}.on_post_agent failed: {e}")
        return HookResult(action=HookAction.ALLOW)
