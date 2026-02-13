"""Tool executor — synchronous tool-use loop for agents."""

from __future__ import annotations

import inspect
import json
from typing import Any, Optional

from loguru import logger

from src.agents.tools.registry import ToolRegistry, ToolResult, tool_registry
from src.agents.llm_router import router, AgentRole, Provider, MODELS, _Clients, _provider_available


class ToolExecutor:
    """
    Synchronous tool-use executor.

    Two modes:
      1. Anthropic native tool_use (Claude sees tool schemas, decides when to call)
      2. Prompt-based fallback (tool descriptions in system prompt, parse JSON calls)

    All execution is synchronous — safe to call from FastAPI sync handlers.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None, max_iterations: int = 3):
        self.registry = registry or tool_registry
        self.max_iterations = max_iterations

    def chat_with_tools(
        self,
        role: AgentRole,
        system: str,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """
        Send a chat that may invoke tools. Fully synchronous.

        Tries Anthropic native tool_use first, falls back to prompt-based.
        """
        tools = self.registry.list_tools()
        if not tools:
            return router.chat(role=role, system=system, messages=messages,
                               temperature=temperature, max_tokens=max_tokens)

        schemas = [t.to_schema() for t in tools]

        # Try native tool_use via Anthropic
        if _provider_available(Provider.ANTHROPIC):
            try:
                return self._anthropic_tool_loop(system, messages, schemas, temperature, max_tokens)
            except Exception as e:
                logger.warning(f"Anthropic tool_use failed: {e}, falling back to prompt-based")

        # Fallback: prompt-based
        return self._prompt_based_tool_loop(role, system, messages, schemas, temperature, max_tokens)

    # ------------------------------------------------------------------
    # Anthropic native tool_use (synchronous)
    # ------------------------------------------------------------------

    def _anthropic_tool_loop(
        self, system: str, messages: list[dict], schemas: list[dict],
        temperature: float, max_tokens: int,
    ) -> str:
        client = _Clients.anthropic()
        tools = [{"name": s["name"], "description": s["description"], "input_schema": s["input_schema"]} for s in schemas]

        working_messages = list(messages)
        resp = None

        for iteration in range(self.max_iterations):
            resp = client.messages.create(
                model=MODELS["claude-sonnet"].model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=working_messages,
                tools=tools,
            )

            if resp.stop_reason == "tool_use":
                assistant_content = resp.content
                working_messages.append({"role": "assistant", "content": assistant_content})

                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        result = self._execute_tool_sync(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result.to_text(),
                        })
                        logger.info(f"Tool called: {block.name}({block.input}) -> {result.to_text()[:80]}")

                working_messages.append({"role": "user", "content": tool_results})
            else:
                text_parts = [b.text for b in resp.content if hasattr(b, 'text')]
                return " ".join(text_parts) if text_parts else ""

        logger.warning(f"Tool loop hit max iterations ({self.max_iterations})")
        if resp:
            text_parts = [b.text for b in resp.content if hasattr(b, 'text')]
            return " ".join(text_parts) if text_parts else "Let me think about that differently."
        return "Let me think about that differently."

    # ------------------------------------------------------------------
    # Prompt-based tool calling (any provider)
    # ------------------------------------------------------------------

    def _prompt_based_tool_loop(
        self, role: AgentRole, system: str, messages: list[dict],
        schemas: list[dict], temperature: float, max_tokens: int,
    ) -> str:
        tool_descriptions = []
        for s in schemas:
            params_desc = ", ".join(
                f"{k}: {v.get('type', 'string')}" for k, v in s["input_schema"]["properties"].items()
            )
            tool_descriptions.append(f"- {s['name']}({params_desc}): {s['description']}")

        tool_block = "\n".join(tool_descriptions)
        augmented_system = (
            f"{system}\n\n"
            f"[Available Tools]\n{tool_block}\n\n"
            f"If you need to use a tool, respond ONLY with JSON:\n"
            f'{{"tool": "tool_name", "args": {{"param": "value"}}}}\n'
            f"Otherwise, respond normally with text."
        )

        text = ""
        working_messages = list(messages)

        for iteration in range(self.max_iterations):
            text = router.chat(
                role=role, system=augmented_system, messages=working_messages,
                temperature=temperature, max_tokens=max_tokens,
            )

            tool_call = self._parse_tool_call(text)
            if tool_call:
                result = self._execute_tool_sync(tool_call["tool"], tool_call.get("args", {}))
                logger.info(f"Tool called (prompt-based): {tool_call['tool']} -> {result.to_text()[:80]}")
                working_messages = working_messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"[Tool result for {tool_call['tool']}]: {result.to_text()}"},
                ]
            else:
                return text

        return text

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _execute_tool_sync(self, name: str, args: dict) -> ToolResult:
        """Execute a tool synchronously."""
        tool = self.registry.get(name)
        if not tool:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        if not tool.handler:
            return ToolResult(success=False, error=f"Tool '{name}' has no handler")
        try:
            result = tool.handler(**args)
            return ToolResult(success=True, output=result)
        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}")
            return ToolResult(success=False, error=str(e))

    @staticmethod
    def _parse_tool_call(text: str) -> Optional[dict]:
        """Try to parse a tool call JSON from LLM output."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "tool" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        return None
