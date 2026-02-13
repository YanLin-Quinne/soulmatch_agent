"""Built-in tools for the dating agent."""

import json
from datetime import datetime, timezone

from loguru import logger

from src.agents.tools.registry import Tool, ToolParameter, tool_registry


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def get_current_time(timezone_name: str = "UTC") -> str:
    """Get the current date and time."""
    now = datetime.now(timezone.utc)
    return json.dumps({
        "datetime": now.isoformat(),
        "date": now.strftime("%B %d, %Y"),
        "time": now.strftime("%I:%M %p UTC"),
        "day_of_week": now.strftime("%A"),
    })


def search_web(query: str) -> str:
    """Simulate a web search (returns placeholder — plug in real API)."""
    # In production, connect to a search API (SerpAPI, Brave, etc.)
    return json.dumps({
        "query": query,
        "note": "Web search not configured. Provide SEARCH_API_KEY in .env to enable.",
        "results": [],
    })


def get_weather(city: str) -> str:
    """Simulate weather lookup (returns placeholder — plug in real API)."""
    # In production, connect to OpenWeatherMap or similar
    return json.dumps({
        "city": city,
        "note": "Weather API not configured. Provide WEATHER_API_KEY in .env to enable.",
        "condition": "unknown",
    })


def get_dating_advice(topic: str) -> str:
    """Provide dating conversation tips for common topics."""
    advice_db = {
        "first_message": "Start with something specific from their profile. Avoid generic 'hey'. Show genuine curiosity.",
        "keeping_conversation": "Ask open-ended questions. Share related stories. Use humor naturally.",
        "red_flags": "Watch for inconsistencies, pressure to move off-platform, requests for money, love bombing.",
        "meeting_up": "Suggest a public place. Daytime coffee or walk is ideal for first meets. Tell a friend.",
        "emotional_connection": "Share vulnerabilities gradually. Active listening matters more than clever replies.",
    }
    # Fuzzy match
    result = advice_db.get(topic)
    if not result:
        for key, val in advice_db.items():
            if topic.lower() in key or key in topic.lower():
                result = val
                break
    if not result:
        result = "Be genuine, ask open questions, and listen actively. Good connections start with curiosity."
    return json.dumps({"topic": topic, "advice": result})


def analyze_compatibility(user_traits: str, partner_traits: str) -> str:
    """Analyze compatibility between two sets of personality traits."""
    return json.dumps({
        "user_traits": user_traits,
        "partner_traits": partner_traits,
        "analysis": "Compatibility analysis requires the full feature prediction pipeline. Use the chat to build trait profiles first.",
        "suggestion": "Continue the conversation to let the prediction model gather more data.",
    })


def get_conversation_stats(session_data: str = "") -> str:
    """Get statistics about the current conversation."""
    return json.dumps({
        "note": "Call this tool from within an active session to get turn count, emotion history, and feature confidence.",
        "available_metrics": [
            "turn_count", "emotion_distribution", "avg_feature_confidence",
            "top_interests", "communication_style_detected",
        ],
    })


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_builtin_tools() -> None:
    """Register all built-in tools to the global registry."""

    tool_registry.register(Tool(
        name="get_current_time",
        description="Get the current date, time, and day of the week.",
        parameters=[
            ToolParameter("timezone_name", "string", "Timezone name (e.g., 'UTC', 'US/Pacific')", required=False, default="UTC"),
        ],
        handler=get_current_time,
        category="general",
    ))

    tool_registry.register(Tool(
        name="search_web",
        description="Search the web for information. Useful when the user asks about current events, facts, or anything you don't know.",
        parameters=[
            ToolParameter("query", "string", "The search query"),
        ],
        handler=search_web,
        category="web",
    ))

    tool_registry.register(Tool(
        name="get_weather",
        description="Get the current weather for a city. Useful for making conversation about weather or planning a date.",
        parameters=[
            ToolParameter("city", "string", "City name (e.g., 'San Francisco')"),
        ],
        handler=get_weather,
        category="web",
    ))

    tool_registry.register(Tool(
        name="get_dating_advice",
        description="Get dating conversation tips for common scenarios like first messages, keeping conversations going, or spotting red flags.",
        parameters=[
            ToolParameter("topic", "string", "The dating topic to get advice on",
                         enum=["first_message", "keeping_conversation", "red_flags", "meeting_up", "emotional_connection"]),
        ],
        handler=get_dating_advice,
        category="dating",
    ))

    tool_registry.register(Tool(
        name="analyze_compatibility",
        description="Analyze compatibility between two people based on their personality traits.",
        parameters=[
            ToolParameter("user_traits", "string", "Comma-separated user personality traits"),
            ToolParameter("partner_traits", "string", "Comma-separated partner personality traits"),
        ],
        handler=analyze_compatibility,
        category="dating",
    ))

    tool_registry.register(Tool(
        name="get_conversation_stats",
        description="Get statistics and insights about the current conversation including turn count, emotion history, and feature confidence levels.",
        parameters=[
            ToolParameter("session_data", "string", "Optional session context", required=False, default=""),
        ],
        handler=get_conversation_stats,
        category="dating",
    ))

    logger.info(f"Registered {len(tool_registry)} built-in tools")
