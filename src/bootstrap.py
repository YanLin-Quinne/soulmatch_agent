"""Bootstrap Graph — staged initialization inspired by Claude Code's 7-stage bootstrap."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Coroutine, Optional, Union

from loguru import logger


class BootstrapStage(IntEnum):
    """Ordered initialization stages."""
    CONFIG = 0
    DATABASE = 1
    TOOLS = 2
    PERSONAS = 3
    SESSION_MGR = 4
    BACKGROUND = 5
    READY = 6


@dataclass
class StageResult:
    """Result of executing a bootstrap stage."""
    stage: BootstrapStage
    success: bool
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class _StageEntry:
    stage: BootstrapStage
    fn: Union[Callable[[], Any], Callable[[], Coroutine]]
    depends_on: list[BootstrapStage] = field(default_factory=list)


class BootstrapGraph:
    """Manages ordered, dependency-aware initialization stages."""

    def __init__(self):
        self._stages: dict[BootstrapStage, _StageEntry] = {}
        self._results: dict[BootstrapStage, StageResult] = {}
        self._ready = False

    def register_stage(
        self,
        stage: BootstrapStage,
        fn: Union[Callable, Callable[..., Coroutine]],
        depends_on: Optional[list[BootstrapStage]] = None,
    ) -> None:
        self._stages[stage] = _StageEntry(stage=stage, fn=fn, depends_on=depends_on or [])

    async def run_all(self) -> list[StageResult]:
        """Execute all stages in order, respecting dependencies."""
        results: list[StageResult] = []
        failed_stages: set[BootstrapStage] = set()

        for stage in sorted(self._stages.keys()):
            entry = self._stages[stage]

            # Check if any dependency failed
            blocked_by = [d for d in entry.depends_on if d in failed_stages]
            if blocked_by:
                result = StageResult(
                    stage=stage, success=False,
                    error=f"Skipped: dependency {[s.name for s in blocked_by]} failed",
                )
                failed_stages.add(stage)
                self._results[stage] = result
                results.append(result)
                logger.warning(f"[Bootstrap] {stage.name}: {result.error}")
                continue

            t0 = time.monotonic()
            try:
                if asyncio.iscoroutinefunction(entry.fn):
                    await entry.fn()
                else:
                    entry.fn()
                elapsed = (time.monotonic() - t0) * 1000
                result = StageResult(stage=stage, success=True, duration_ms=round(elapsed, 1))
                logger.info(f"[Bootstrap] {stage.name} OK ({elapsed:.0f}ms)")
            except Exception as e:
                elapsed = (time.monotonic() - t0) * 1000
                result = StageResult(stage=stage, success=False, error=str(e), duration_ms=round(elapsed, 1))
                failed_stages.add(stage)
                logger.error(f"[Bootstrap] {stage.name} FAILED: {e}")

            self._results[stage] = result
            results.append(result)

        self._ready = BootstrapStage.READY not in failed_stages
        return results

    def is_ready(self) -> bool:
        return self._ready

    def get_results(self) -> dict[str, dict]:
        """Return stage results as a serializable dict."""
        return {
            s.name: {"success": r.success, "error": r.error, "duration_ms": r.duration_ms}
            for s, r in self._results.items()
        }


# ---------------------------------------------------------------------------
# Default bootstrap factory
# ---------------------------------------------------------------------------

_session_manager = None
_bot_pool = None


def create_default_bootstrap() -> BootstrapGraph:
    """Create the standard ai_you bootstrap graph."""
    graph = BootstrapGraph()

    # Stage 0: CONFIG — validate settings and API keys
    def _init_config():
        from src.config import settings
        missing = []
        # At least one LLM provider must be configured
        has_provider = any([
            settings.anthropic_api_key,
            settings.openai_api_key,
            settings.gemini_api_key,
            settings.deepseek_api_key,
            settings.qwen_api_key,
        ])
        if not has_provider:
            missing.append("No LLM API key configured (need at least one)")
        if missing:
            logger.warning(f"[Bootstrap] Config warnings: {missing}")
        logger.info(f"[Bootstrap] Config loaded: data_dir={settings.data_dir}")

    graph.register_stage(BootstrapStage.CONFIG, _init_config)

    # Stage 1: DATABASE — initialize persistence layer
    def _init_database():
        from src.persistence.database import init_database
        init_database()

    graph.register_stage(BootstrapStage.DATABASE, _init_database, depends_on=[BootstrapStage.CONFIG])

    # Stage 2: TOOLS — register built-in tools
    def _init_tools():
        from src.agents.tools.builtin import register_builtin_tools
        register_builtin_tools()

    graph.register_stage(BootstrapStage.TOOLS, _init_tools, depends_on=[BootstrapStage.CONFIG])

    # Stage 3: PERSONAS — load bot persona pool
    def _init_personas():
        global _bot_pool
        from src.agents.persona_agent import PersonaAgentPool
        pool = PersonaAgentPool()
        pool.load_from_file("./data/processed/bot_personas.json")
        _bot_pool = pool
        logger.info(f"[Bootstrap] Loaded {len(pool.get_agent_summaries())} bot personas")

    graph.register_stage(BootstrapStage.PERSONAS, _init_personas, depends_on=[BootstrapStage.CONFIG])

    # Stage 4: SESSION_MGR — wire up session manager with persona pool
    def _init_session_manager():
        global _session_manager
        from src.api.session_manager import SessionManager
        sm = SessionManager()
        if _bot_pool:
            sm.set_bot_personas_pool(_bot_pool)
        _session_manager = sm

    graph.register_stage(BootstrapStage.SESSION_MGR, _init_session_manager, depends_on=[BootstrapStage.PERSONAS])

    # Stage 5: BACKGROUND — placeholder; started by the specific entry point
    def _init_background():
        pass

    graph.register_stage(BootstrapStage.BACKGROUND, _init_background, depends_on=[BootstrapStage.SESSION_MGR])

    # Stage 6: READY — mark system ready
    def _mark_ready():
        logger.info("[Bootstrap] System ready")

    graph.register_stage(BootstrapStage.READY, _mark_ready, depends_on=[BootstrapStage.SESSION_MGR])

    return graph


def get_session_manager():
    """Get the bootstrapped session manager singleton."""
    return _session_manager


def get_bot_pool():
    """Get the bootstrapped bot persona pool."""
    return _bot_pool
