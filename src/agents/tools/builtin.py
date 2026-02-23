"""Built-in tools for the dating agent."""

import json
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from src.agents.tools.registry import Tool, ToolParameter, tool_registry


# ---------------------------------------------------------------------------
# Tool context injection (module-level, set by orchestrator per turn)
# ---------------------------------------------------------------------------

_tool_context: dict[str, Any] = {}


def set_tool_context(ctx=None, memory_manager=None, feature_agent=None):
    """Inject live objects so tool handlers can access conversation state."""
    _tool_context.clear()
    if ctx:
        _tool_context["ctx"] = ctx
    if memory_manager:
        _tool_context["memory_manager"] = memory_manager
    if feature_agent:
        _tool_context["feature_agent"] = feature_agent


def clear_tool_context():
    """Remove references after bot response to prevent leaks."""
    _tool_context.clear()


# ---------------------------------------------------------------------------
# Helper: lazy router access (avoids circular import)
# ---------------------------------------------------------------------------

def _get_router():
    from src.agents.llm_router import router, AgentRole
    return router, AgentRole


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
    """Search the web via DuckDuckGo instant answer API, with LLM fallback."""
    import httpx

    # Try DuckDuckGo instant answer API (free, no key)
    try:
        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            timeout=5.0,
        )
        data = resp.json()
        abstract = data.get("AbstractText", "")
        related = [
            t.get("Text", "") for t in data.get("RelatedTopics", [])[:3]
            if isinstance(t, dict) and t.get("Text")
        ]
        if abstract or related:
            return json.dumps({
                "query": query,
                "abstract": abstract,
                "related": related,
                "source": data.get("AbstractSource", "DuckDuckGo"),
            })
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")

    # Fallback: ask LLM
    try:
        router, AgentRole = _get_router()
        answer = router.chat(
            role=AgentRole.GENERAL,
            system="Answer the following question concisely based on your knowledge. If unsure, say so.",
            messages=[{"role": "user", "content": query}],
            temperature=0.3,
            max_tokens=200,
        )
        return json.dumps({"query": query, "answer": answer, "source": "llm_knowledge"})
    except Exception as e:
        logger.error(f"LLM fallback for search also failed: {e}")
        return json.dumps({"query": query, "error": "Search unavailable", "results": []})


def get_weather(city: str) -> str:
    """Get current weather for a city via wttr.in."""
    import httpx

    try:
        resp = httpx.get(
            f"https://wttr.in/{city}",
            params={"format": "j1"},
            headers={"User-Agent": "soulmatch-agent/1.0"},
            timeout=5.0,
        )
        data = resp.json()
        current = data.get("current_condition", [{}])[0]
        return json.dumps({
            "city": city,
            "temp_c": current.get("temp_C"),
            "feels_like_c": current.get("FeelsLikeC"),
            "condition": current.get("weatherDesc", [{}])[0].get("value", "unknown"),
            "humidity": current.get("humidity"),
            "wind_kmph": current.get("windspeedKmph"),
        })
    except Exception as e:
        logger.warning(f"Weather lookup failed for {city}: {e}")
        return json.dumps({"city": city, "condition": "unknown", "error": str(e)})


_ADVICE_FALLBACK = {
    "first_message": "Start with something specific from their profile. Avoid generic 'hey'. Show genuine curiosity.",
    "keeping_conversation": "Ask open-ended questions. Share related stories. Use humor naturally.",
    "red_flags": "Watch for inconsistencies, pressure to move off-platform, requests for money, love bombing.",
    "meeting_up": "Suggest a public place. Daytime coffee or walk is ideal for first meets. Tell a friend.",
    "emotional_connection": "Share vulnerabilities gradually. Active listening matters more than clever replies.",
}


def get_dating_advice(topic: str) -> str:
    """Provide context-aware dating advice via LLM, with static fallback."""
    ctx = _tool_context.get("ctx")

    # Build context-aware prompt
    emotion_hint = ""
    if ctx and ctx.current_emotion:
        emotion_hint = f" The user currently feels {ctx.current_emotion}."

    try:
        router, AgentRole = _get_router()
        prompt = (
            f"Give concise, practical dating advice about: {topic}.{emotion_hint} "
            "Keep it warm, specific, and under 3 sentences."
        )
        answer = router.chat(
            role=AgentRole.GENERAL,
            system="You are a supportive dating coach. Be practical and empathetic.",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
        )
        return json.dumps({"topic": topic, "advice": answer})
    except Exception as e:
        logger.warning(f"LLM dating advice failed: {e}, using fallback")

    # Static fallback
    result = _ADVICE_FALLBACK.get(topic)
    if not result:
        for key, val in _ADVICE_FALLBACK.items():
            if topic.lower() in key or key in topic.lower():
                result = val
                break
    if not result:
        result = "Be genuine, ask open questions, and listen actively. Good connections start with curiosity."
    return json.dumps({"topic": topic, "advice": result})


def analyze_compatibility(user_traits: str, partner_traits: str) -> str:
    """Analyze compatibility using real predicted features + LLM reasoning."""
    feature_agent = _tool_context.get("feature_agent")

    # Try to get real feature data
    real_features = ""
    if feature_agent:
        try:
            summary = feature_agent.get_feature_summary()
            personality = summary.get("personality", {})
            interests = summary.get("interests", {})
            if personality or interests:
                real_features = (
                    f"\n\nPredicted user personality (Big Five): {json.dumps(personality)}"
                    f"\nPredicted user interests: {json.dumps(interests)}"
                )
        except Exception as e:
            logger.warning(f"Feature summary retrieval failed: {e}")

    try:
        router, AgentRole = _get_router()
        prompt = (
            f"Analyze dating compatibility.\n"
            f"User traits: {user_traits}\nPartner traits: {partner_traits}"
            f"{real_features}\n\n"
            "Return a brief analysis with: compatibility_level (high/medium/low), "
            "2 strengths, 2 potential challenges. Be concise."
        )
        answer = router.chat(
            role=AgentRole.GENERAL,
            system="You are a relationship compatibility analyst. Be balanced and insightful.",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=250,
        )
        return json.dumps({
            "user_traits": user_traits,
            "partner_traits": partner_traits,
            "analysis": answer,
        })
    except Exception as e:
        logger.warning(f"LLM compatibility analysis failed: {e}")
        return json.dumps({
            "user_traits": user_traits,
            "partner_traits": partner_traits,
            "analysis": "Continue chatting to build a clearer picture of compatibility.",
            "suggestion": "The more you share, the better the analysis becomes.",
        })


def get_conversation_stats(session_data: str = "") -> str:
    """Get real statistics about the current conversation from AgentContext."""
    ctx = _tool_context.get("ctx")
    feature_agent = _tool_context.get("feature_agent")

    stats: dict[str, Any] = {}

    if ctx:
        stats["turn_count"] = ctx.turn_count
        stats["current_emotion"] = ctx.current_emotion
        stats["emotion_history"] = ctx.emotion_history[-10:] if ctx.emotion_history else []
        stats["scam_risk_score"] = ctx.scam_risk_score
        stats["current_state"] = ctx.current_state
        stats["relationship_status"] = ctx.rel_status
        stats["sentiment"] = ctx.sentiment_label

    if feature_agent:
        try:
            summary = feature_agent.get_feature_summary()
            stats["avg_confidence"] = summary.get("overall_confidence", 0)
            stats["communication_style"] = feature_agent.predicted_features.get("communication_style")
            # Top interests (score > 0.5)
            interests = {
                k.replace("interest_", ""): v
                for k, v in feature_agent.predicted_features.items()
                if k.startswith("interest_") and isinstance(v, (int, float)) and v > 0.5
            }
            stats["top_interests"] = interests
            stats["low_confidence_features"] = ctx.low_confidence_features if ctx else []
        except Exception as e:
            logger.warning(f"Feature stats retrieval failed: {e}")

    if not stats:
        stats["note"] = "No active session context available."

    return json.dumps(stats)


def recall_memory(query: str) -> str:
    """Retrieve relevant memories about the user from conversation history."""
    mm = _tool_context.get("memory_manager")
    if not mm:
        return json.dumps({"query": query, "memories": [], "note": "Memory manager not available."})

    try:
        memories = mm.retrieve_relevant_memories(query, n=5)
        # Return top 3
        return json.dumps({"query": query, "memories": memories[:3]})
    except Exception as e:
        logger.warning(f"Memory recall failed: {e}")
        return json.dumps({"query": query, "memories": [], "error": str(e)})


def lookup_user_profile(aspect: str = "all") -> str:
    """Look up predicted user profile by aspect."""
    feature_agent = _tool_context.get("feature_agent")
    if not feature_agent:
        return json.dumps({"aspect": aspect, "note": "Feature agent not available."})

    try:
        summary = feature_agent.get_feature_summary()
    except Exception as e:
        logger.warning(f"Profile lookup failed: {e}")
        return json.dumps({"aspect": aspect, "error": str(e)})

    if aspect == "all":
        return json.dumps({"aspect": "all", "profile": summary})

    section = summary.get(aspect)
    if section is not None:
        return json.dumps({"aspect": aspect, "data": section})

    return json.dumps({"aspect": aspect, "note": f"No data for aspect '{aspect}'.", "available": list(summary.keys())})


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

    tool_registry.register(Tool(
        name="recall_memory",
        description="Search conversation memory for things the user mentioned before (e.g., their job, hobbies, favorite food, past experiences).",
        parameters=[
            ToolParameter("query", "string", "What to search for in memory (e.g., 'favorite food', 'their job')"),
        ],
        handler=recall_memory,
        category="dating",
    ))

    tool_registry.register(Tool(
        name="lookup_user_profile",
        description="Look up the predicted user profile including personality traits, interests, demographics, and lifestyle.",
        parameters=[
            ToolParameter("aspect", "string", "Which aspect of the profile to look up",
                         enum=["personality", "interests", "demographics", "lifestyle", "all"]),
        ],
        handler=lookup_user_profile,
        category="dating",
    ))

    logger.info(f"Registered {len(tool_registry)} built-in tools")
