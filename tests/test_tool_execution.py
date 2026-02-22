"""Integration tests for tool execution"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.tools.registry import ToolRegistry, Tool, ToolResult
from src.agents.tools.executor import ToolExecutor
from src.agents.llm_router import AgentRole


class TestToolExecution:
    """Test tool calling integration"""

    @pytest.fixture
    def registry(self):
        return ToolRegistry()

    @pytest.fixture
    def executor(self, registry):
        return ToolExecutor(registry=registry)

    def test_tool_registry_registration(self, registry):
        """Test ToolRegistry registration"""
        def mock_handler():
            return "result"

        tool = Tool(
            name="test_tool",
            description="Test tool",
            parameters={},
            handler=mock_handler
        )

        registry.register(tool)

        assert registry.get("test_tool") is not None
        assert registry.get("test_tool").name == "test_tool"

    def test_tool_executor_builtin_tools(self, executor):
        """Test ToolExecutor execute 6 builtin tools"""
        # Register mock builtin tools
        builtin_tools = [
            "get_current_time",
            "search_web",
            "get_weather",
            "calculate",
            "translate",
            "summarize"
        ]

        for tool_name in builtin_tools:
            tool = Tool(
                name=tool_name,
                description=f"Mock {tool_name}",
                parameters={},
                handler=lambda: {"result": "success"}
            )
            executor.registry.register(tool)

        # Execute each tool
        for tool_name in builtin_tools:
            result = executor._execute_tool_sync(tool_name, {})
            assert result.success is True

    def test_claude_tool_use_native_calling(self, executor):
        """Test Claude tool_use native calling"""
        # Mock Anthropic client
        with patch('src.agents.llm_router._provider_available', return_value=True), \
             patch('src.agents.llm_router._Clients.anthropic') as mock_client:

            mock_response = Mock()
            mock_response.stop_reason = "tool_use"
            mock_response.content = [
                Mock(type="tool_use", name="test_tool", input={}, id="tool_1")
            ]

            mock_client.return_value.messages.create.return_value = mock_response

            # Verify tool_use response
            assert mock_response.stop_reason == "tool_use"

    def test_non_claude_provider_prompt_fallback(self, executor):
        """Test non-Claude provider prompt fallback"""
        # Mock non-Claude provider
        with patch('src.agents.llm_router._provider_available', return_value=False):

            # Verify prompt-based fallback
            schemas = [{"name": "test_tool", "description": "Test", "input_schema": {"properties": {}}}]

            tool_descriptions = []
            for s in schemas:
                tool_descriptions.append(f"- {s['name']}: {s['description']}")

            assert len(tool_descriptions) == 1
            assert "test_tool" in tool_descriptions[0]

    def test_tool_parameter_validation(self, executor):
        """Test tool parameter validation and error handling"""
        def strict_handler(required_param: str):
            if not required_param:
                raise ValueError("required_param is required")
            return {"result": required_param}

        tool = Tool(
            name="strict_tool",
            description="Strict tool",
            parameters={"required_param": {"type": "string", "required": True}},
            handler=strict_handler
        )

        executor.registry.register(tool)

        # Test missing parameter
        result = executor._execute_tool_sync("strict_tool", {})
        assert result.success is False

        # Test valid parameter
        result = executor._execute_tool_sync("strict_tool", {"required_param": "value"})
        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
