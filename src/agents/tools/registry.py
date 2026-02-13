"""Tool registry — defines the tool schema and dispatch mechanism."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from loguru import logger


@dataclass
class ToolResult:
    """Result from a tool invocation."""
    success: bool
    output: Any = None
    error: Optional[str] = None

    def to_text(self) -> str:
        if self.success:
            return str(self.output) if self.output is not None else "(no output)"
        return f"Error: {self.error}"


@dataclass
class ToolParameter:
    name: str
    type: str  # "string" | "number" | "boolean" | "array" | "object"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[list[str]] = None


@dataclass
class Tool:
    """A callable tool with a JSON-schema-style definition."""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable] = None
    category: str = "general"  # "web", "data", "dating", "general"

    def to_schema(self) -> dict:
        """Convert to JSON schema format compatible with Claude/GPT tool_use."""
        properties = {}
        required = []
        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool handler."""
        if not self.handler:
            return ToolResult(success=False, error=f"Tool '{self.name}' has no handler")
        try:
            if inspect.iscoroutinefunction(self.handler):
                result = await self.handler(**kwargs)
            else:
                result = self.handler(**kwargs)
            return ToolResult(success=True, output=result)
        except Exception as e:
            logger.error(f"Tool '{self.name}' failed: {e}")
            return ToolResult(success=False, error=str(e))


class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> list[Tool]:
        if category:
            return [t for t in self._tools.values() if t.category == category]
        return list(self._tools.values())

    def get_schemas(self, category: Optional[str] = None) -> list[dict]:
        """Get all tool schemas for passing to LLM."""
        return [t.to_schema() for t in self.list_tools(category)]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


# Module-level singleton
tool_registry = ToolRegistry()
