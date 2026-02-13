"""
MCP Server — exposes SoulMatch agent capabilities via Model Context Protocol.

Run standalone:
    python -m src.mcp.server

Or integrate with an MCP client (Claude Desktop, etc.) via stdio transport.
"""

import json
import asyncio
from typing import Any

from loguru import logger

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool as MCPTool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("mcp package not installed. Run: pip install mcp")


def create_mcp_server() -> Any:
    """Create and configure the MCP server with SoulMatch tools."""
    if not MCP_AVAILABLE:
        raise RuntimeError("mcp package not installed. Run: pip install mcp")

    server = Server("soulmatch")

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    @server.list_tools()
    async def list_tools() -> list[MCPTool]:
        return [
            MCPTool(
                name="analyze_emotion",
                description="Analyze the emotional content of a text message. Returns emotion category, confidence, and intensity.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "The text message to analyze"},
                    },
                    "required": ["message"],
                },
            ),
            MCPTool(
                name="predict_features",
                description="Predict personality features from a conversation history. Returns 22-dimension trait predictions with confidence scores.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "conversation": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "speaker": {"type": "string"},
                                    "message": {"type": "string"},
                                },
                            },
                            "description": "List of conversation turns [{speaker, message}]",
                        },
                    },
                    "required": ["conversation"],
                },
            ),
            MCPTool(
                name="check_scam",
                description="Check a message for potential romance scam patterns. Returns risk score and warning level.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "The message to check"},
                        "conversation_history": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Previous messages for context (optional)",
                        },
                    },
                    "required": ["message"],
                },
            ),
            MCPTool(
                name="suggest_topics",
                description="Suggest natural conversation topics that would help infer specific personality traits.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "target_traits": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Trait names to probe (e.g., ['openness', 'communication_style'])",
                        },
                        "recent_messages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Recent conversation messages for context",
                        },
                    },
                    "required": ["target_traits"],
                },
            ),
            MCPTool(
                name="get_usage_stats",
                description="Get LLM usage statistics including token counts, costs per provider, and error rates.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            if name == "analyze_emotion":
                from src.agents.emotion_agent import EmotionAgent
                agent = EmotionAgent()
                result = agent.analyze_message(arguments["message"])
                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "predict_features":
                from src.agents.feature_prediction_agent import FeaturePredictionAgent
                agent = FeaturePredictionAgent("mcp_user")
                conversation = arguments["conversation"]
                history = [{"speaker": m["speaker"], "message": m["message"], "turn": i}
                          for i, m in enumerate(conversation)]
                result = agent.predict_from_conversation(history)
                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

            elif name == "check_scam":
                from src.agents.scam_detection_agent import ScamDetectionAgent
                agent = ScamDetectionAgent(use_semantic=True)
                result = agent.analyze_message(arguments["message"])
                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "suggest_topics":
                from src.agents.question_strategy_agent import QuestionStrategyAgent
                agent = QuestionStrategyAgent()
                traits = arguments["target_traits"]
                recent = arguments.get("recent_messages", [])
                history = [{"speaker": "user", "message": m, "turn": i} for i, m in enumerate(recent)]
                probes = agent.suggest_probes(traits, history)
                return [TextContent(type="text", text=json.dumps({"probes": probes}, indent=2))]

            elif name == "get_usage_stats":
                from src.agents.llm_router import router
                report = router.get_usage_report()
                return [TextContent(type="text", text=json.dumps(report, indent=2))]

            else:
                return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

        except Exception as e:
            logger.error(f"MCP tool '{name}' failed: {e}")
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    return server


async def main():
    """Run MCP server via stdio transport."""
    if not MCP_AVAILABLE:
        print("Error: mcp package not installed. Run: pip install mcp")
        return

    server = create_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        logger.info("SoulMatch MCP server started (stdio)")
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
