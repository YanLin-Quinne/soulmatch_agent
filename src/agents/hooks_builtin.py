"""Built-in hooks for the ai_you agent pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.agents.hooks import Hook, HookAction, HookResult


class LoggingHook:
    """Logs agent phase entry/exit with timing."""

    def __init__(self):
        self._start_times: dict[str, float] = {}

    async def on_pre_agent(self, agent_name: str, ctx: Any) -> HookResult:
        self._start_times[agent_name] = time.monotonic()
        turn = getattr(ctx, "turn_count", "?")
        logger.info(f"[Hook] >> {agent_name} starting (turn {turn})")
        return HookResult()

    async def on_post_agent(self, agent_name: str, ctx: Any, result: Any) -> HookResult:
        elapsed = time.monotonic() - self._start_times.pop(agent_name, time.monotonic())
        logger.info(f"[Hook] << {agent_name} completed ({elapsed:.3f}s)")
        return HookResult()


class CostTrackingHook:
    """Tracks LLM token cost per agent phase."""

    def __init__(self):
        self._snapshots: dict[str, dict] = {}
        self.per_agent_cost: dict[str, dict] = {}

    def _get_usage_snapshot(self) -> dict:
        try:
            from src.agents.llm_router import router
            return {
                "input_tokens": router.usage.total_input_tokens,
                "output_tokens": router.usage.total_output_tokens,
                "cost": router.usage.total_cost_usd,
            }
        except Exception:
            return {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}

    async def on_pre_agent(self, agent_name: str, ctx: Any) -> HookResult:
        self._snapshots[agent_name] = self._get_usage_snapshot()
        return HookResult()

    async def on_post_agent(self, agent_name: str, ctx: Any, result: Any) -> HookResult:
        before = self._snapshots.pop(agent_name, {"input_tokens": 0, "output_tokens": 0, "cost": 0.0})
        after = self._get_usage_snapshot()
        delta = {
            "input_tokens": after["input_tokens"] - before["input_tokens"],
            "output_tokens": after["output_tokens"] - before["output_tokens"],
            "cost": round(after["cost"] - before["cost"], 6),
        }
        if delta["input_tokens"] > 0 or delta["output_tokens"] > 0:
            self.per_agent_cost[agent_name] = self.per_agent_cost.get(agent_name, {
                "input_tokens": 0, "output_tokens": 0, "cost": 0.0, "calls": 0,
            })
            self.per_agent_cost[agent_name]["input_tokens"] += delta["input_tokens"]
            self.per_agent_cost[agent_name]["output_tokens"] += delta["output_tokens"]
            self.per_agent_cost[agent_name]["cost"] += delta["cost"]
            self.per_agent_cost[agent_name]["calls"] += 1
        return HookResult()

    def get_cost_report(self) -> dict:
        return dict(self.per_agent_cost)


class ScamRiskGateHook:
    """Denies bot_response when scam risk is critically high."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    async def on_pre_agent(self, agent_name: str, ctx: Any) -> HookResult:
        if agent_name == "bot_response":
            risk = getattr(ctx, "scam_risk_score", 0.0)
            if risk >= self.threshold:
                return HookResult(
                    action=HookAction.DENY,
                    reason=f"Scam risk {risk:.2f} >= threshold {self.threshold:.2f}",
                )
        return HookResult()

    async def on_post_agent(self, agent_name: str, ctx: Any, result: Any) -> HookResult:
        return HookResult()
