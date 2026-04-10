"""
app_demo.py — AI-YOU Gradio Demo

Standalone launch:
    python app_demo.py
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Backend bootstrap
# ---------------------------------------------------------------------------

try:
    from src.bootstrap import create_default_bootstrap, get_bot_pool
    from src.agents.orchestrator import OrchestratorAgent

    _bootstrap = create_default_bootstrap()
    try:
        loop = asyncio.get_running_loop()
        loop.run_until_complete(_bootstrap.run_all())
    except RuntimeError:
        asyncio.run(_bootstrap.run_all())
    BACKEND = get_bot_pool() is not None
except Exception as _boot_err:
    print(f"[app_demo] Backend unavailable: {_boot_err}")
    BACKEND = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCENT_BLUE = "#007BFF"
SURFACE = "#F7FAFC"
CARD_BG = "#FFFFFF"
BORDER = "#D9E3F0"
TEXT = "#1F2937"
MUTED = "#5B6B82"
SOFT_BLUE = "rgba(0, 123, 255, 0.12)"

EMOTION_VALENCE = {
    "joy": 0.9,
    "excitement": 0.8,
    "love": 0.85,
    "trust": 0.6,
    "interest": 0.4,
    "surprise": 0.3,
    "neutral": 0.0,
    "anxiety": -0.5,
    "sadness": -0.7,
    "fear": -0.6,
    "anger": -0.8,
    "disgust": -0.9,
}
EMOTION_COLORS = {
    "joy": "#F59E0B",
    "excitement": "#EF4444",
    "love": "#EC4899",
    "trust": "#06B6D4",
    "interest": "#22C55E",
    "surprise": "#14B8A6",
    "neutral": "#94A3B8",
    "anxiety": "#A855F7",
    "sadness": "#3B82F6",
    "fear": "#8B5CF6",
    "anger": "#DC2626",
    "disgust": "#8B7355",
}

BIG5_KEYS = [
    "big_five_openness",
    "big_five_conscientiousness",
    "big_five_extraversion",
    "big_five_agreeableness",
    "big_five_neuroticism",
]
BIG5_LABELS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]

REL_STAGES = ["stranger", "acquaintance", "crush", "dating", "committed"]
REL_STAGE_LABELS = ["Stranger", "Acquaintance", "Crush", "Dating", "Committed"]

MODE_PERSONA_IDS = {
    "Social Chat": [f"bot_{i}" for i in range(24)],
    "Expert Consulting": ["bot_expert_psych", "bot_expert_career"],
    "Self-Dialogue": ["bot_mirror_self"],
}

MODE_EMOJI = {
    "Social Chat": "💕",
    "Expert Consulting": "🧑‍💼",
    "Self-Dialogue": "🪞",
}

SEX_EMOJI = {
    "F": "👩",
    "M": "👨",
    "female": "👩",
    "male": "👨",
}
STYLE_EMOJI = {
    "warm": "🌸",
    "direct": "⚡",
    "formal": "🎩",
    "casual": "😊",
    "serious": "🧐",
    "humorous": "😄",
    "indirect": "🌀",
    "energetic": "🔥",
}

LIGHT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color=TEXT,
    margin=dict(l=42, r=32, t=40, b=36),
)


# Final fallback personas when backend and raw persona loading both fail.
FALLBACK_PERSONAS = [
    {
        "id": "bot_0",
        "name": "Lin Wanqing",
        "age": 28,
        "sex": "F",
        "location": "Beijing",
        "communication_style": "warm",
        "personality_summary": "Government worker, thoughtful and organized",
        "interests": ["reading", "travel", "cooking"],
        "mbti": "ISFJ",
        "big_five_openness": 0.68,
        "big_five_conscientiousness": 0.82,
        "big_five_extraversion": 0.46,
        "big_five_agreeableness": 0.84,
        "big_five_neuroticism": 0.38,
    },
    {
        "id": "bot_1",
        "name": "Chen Siyu",
        "age": 32,
        "sex": "F",
        "location": "Shanghai",
        "communication_style": "direct",
        "personality_summary": "Administrative clerk, detail-oriented",
        "interests": ["music", "yoga", "art"],
        "mbti": "ESTJ",
        "big_five_openness": 0.58,
        "big_five_conscientiousness": 0.86,
        "big_five_extraversion": 0.59,
        "big_five_agreeableness": 0.73,
        "big_five_neuroticism": 0.35,
    },
    {
        "id": "bot_2",
        "name": "Zhao Yawen",
        "age": 45,
        "sex": "F",
        "location": "Nanjing",
        "communication_style": "formal",
        "personality_summary": "University teacher, intellectual and patient",
        "interests": ["literature", "history", "tea"],
        "mbti": "INFJ",
        "big_five_openness": 0.83,
        "big_five_conscientiousness": 0.74,
        "big_five_extraversion": 0.37,
        "big_five_agreeableness": 0.81,
        "big_five_neuroticism": 0.29,
    },
    {
        "id": "bot_3",
        "name": "Liu Chengyuan",
        "age": 24,
        "sex": "M",
        "location": "Hangzhou",
        "communication_style": "casual",
        "personality_summary": "Programmer, curious and energetic",
        "interests": ["tech", "gaming", "hiking"],
        "mbti": "ENTP",
        "big_five_openness": 0.77,
        "big_five_conscientiousness": 0.61,
        "big_five_extraversion": 0.69,
        "big_five_agreeableness": 0.54,
        "big_five_neuroticism": 0.42,
    },
    {
        "id": "bot_4",
        "name": "Wang Jianye",
        "age": 38,
        "sex": "M",
        "location": "Chengdu",
        "communication_style": "serious",
        "personality_summary": "Business manager, strategic thinker",
        "interests": ["finance", "sports", "travel"],
        "mbti": "INTJ",
        "big_five_openness": 0.71,
        "big_five_conscientiousness": 0.85,
        "big_five_extraversion": 0.44,
        "big_five_agreeableness": 0.48,
        "big_five_neuroticism": 0.31,
    },
    {
        "id": "bot_5",
        "name": "Zhang Yuxuan",
        "age": 29,
        "sex": "M",
        "location": "Shenzhen",
        "communication_style": "humorous",
        "personality_summary": "Finance professional, witty and ambitious",
        "interests": ["investing", "coffee", "photography"],
        "mbti": "ENFP",
        "big_five_openness": 0.74,
        "big_five_conscientiousness": 0.67,
        "big_five_extraversion": 0.78,
        "big_five_agreeableness": 0.63,
        "big_five_neuroticism": 0.36,
    },
    {
        "id": "bot_6",
        "name": "Li Xinyi",
        "age": 52,
        "sex": "F",
        "location": "Guangzhou",
        "communication_style": "warm",
        "personality_summary": "Freelancer, creative and independent",
        "interests": ["painting", "gardening", "meditation"],
        "mbti": "INFP",
        "big_five_openness": 0.86,
        "big_five_conscientiousness": 0.58,
        "big_five_extraversion": 0.33,
        "big_five_agreeableness": 0.82,
        "big_five_neuroticism": 0.34,
    },
    {
        "id": "bot_7",
        "name": "Zhou Mingxuan",
        "age": 21,
        "sex": "M",
        "location": "Wuhan",
        "communication_style": "casual",
        "personality_summary": "Designer, expressive and spontaneous",
        "interests": ["design", "music", "anime"],
        "mbti": "ESFP",
        "big_five_openness": 0.81,
        "big_five_conscientiousness": 0.49,
        "big_five_extraversion": 0.84,
        "big_five_agreeableness": 0.66,
        "big_five_neuroticism": 0.41,
    },
    {
        "id": "bot_expert_psych",
        "name": "Dr. Maya Liu",
        "age": 42,
        "sex": "F",
        "location": "San Francisco",
        "communication_style": "formal",
        "personality_summary": "Warm, insightful psychologist focused on reflection and emotional clarity",
        "interests": ["empathy", "growth", "reflection"],
        "mbti": "INFJ",
        "big_five_openness": 0.90,
        "big_five_conscientiousness": 0.80,
        "big_five_extraversion": 0.55,
        "big_five_agreeableness": 0.85,
        "big_five_neuroticism": 0.25,
    },
    {
        "id": "bot_expert_career",
        "name": "Alex Park",
        "age": 36,
        "sex": "M",
        "location": "New York",
        "communication_style": "direct",
        "personality_summary": "Driven strategist focused on actionable career planning",
        "interests": ["strategy", "achievement", "growth"],
        "mbti": "ENTJ",
        "big_five_openness": 0.80,
        "big_five_conscientiousness": 0.85,
        "big_five_extraversion": 0.75,
        "big_five_agreeableness": 0.70,
        "big_five_neuroticism": 0.30,
    },
    {
        "id": "bot_mirror_self",
        "name": "Mirror Self",
        "age": None,
        "sex": None,
        "location": "Inner Dialogue",
        "communication_style": "casual",
        "personality_summary": "Dynamic personality mirror that reflects the user back to themselves",
        "interests": ["self_reflection", "authenticity", "growth"],
        "mbti": "Adaptive",
        "big_five_openness": 0.50,
        "big_five_conscientiousness": 0.50,
        "big_five_extraversion": 0.50,
        "big_five_agreeableness": 0.50,
        "big_five_neuroticism": 0.50,
    },
]


# ---------------------------------------------------------------------------
# Persona loading
# ---------------------------------------------------------------------------

def _safe_title(value: str | None) -> str:
    if not value:
        return ""
    return str(value).replace("_", " ").title()


def _guess_name_from_prompt(prompt: str | None) -> str:
    if not prompt:
        return ""
    match = re.search(r"You are ([^,]+)", prompt)
    return match.group(1).strip() if match else ""


def _normalize_persona(summary: dict, base: dict | None = None) -> dict:
    base = dict(base or {})
    persona_id = summary.get("profile_id") or base.get("id") or "unknown"
    normalized = {
        "id": persona_id,
        "name": summary.get("name") or base.get("name") or _safe_title(persona_id),
        "age": summary.get("age", base.get("age")),
        "sex": summary.get("sex", base.get("sex")),
        "location": summary.get("location") or base.get("location") or "",
        "job": summary.get("job") or base.get("job") or "",
        "communication_style": summary.get("communication_style") or base.get("communication_style") or "casual",
        "core_values": list(summary.get("core_values") or base.get("core_values") or []),
        "interests": list(summary.get("interests") or base.get("interests") or []),
        "relationship_goals": summary.get("relationship_goals") or base.get("relationship_goals") or "",
        "personality_summary": summary.get("personality_summary") or base.get("personality_summary") or "",
        "mbti": summary.get("mbti") or base.get("mbti") or "N/A",
    }
    for key in BIG5_KEYS:
        normalized[key] = summary.get(key, base.get(key, 0.5))
        normalized[f"{key}_confidence"] = summary.get(f"{key}_confidence", base.get(f"{key}_confidence", 0.74))
    return normalized


def _load_personas_from_backend() -> dict[str, dict]:
    if not BACKEND:
        return {}
    try:
        pool = get_bot_pool()
        if not pool:
            return {}
        return {
            pid: _normalize_persona({"profile_id": pid, **summary})
            for pid, summary in pool.get_agent_summaries().items()
        }
    except Exception:
        return {}


def _load_personas_from_file() -> dict[str, dict]:
    personas_path = Path("data/processed/bot_personas.json")
    if not personas_path.exists():
        return {}
    try:
        raw_personas = json.loads(personas_path.read_text())
    except Exception:
        return {}

    personas: dict[str, dict] = {}
    for record in raw_personas:
        persona_id = record.get("profile_id")
        original = record.get("original_profile", {})
        features = record.get("features", {})
        if not persona_id:
            continue
        summary = {
            "profile_id": persona_id,
            "name": _guess_name_from_prompt(record.get("system_prompt")),
            "age": original.get("age"),
            "sex": (original.get("sex") or "").upper() or None,
            "location": _safe_title(original.get("location")),
            "job": _safe_title(original.get("job")),
            "communication_style": features.get("communication_style", "casual"),
            "core_values": list(features.get("core_values", [])),
            "interests": list(features.get("core_values", [])),
            "relationship_goals": features.get("relationship_goals", ""),
            "personality_summary": features.get("personality_summary")
            or f"{_safe_title(original.get('job')) or 'Conversational partner'} with a {features.get('communication_style', 'casual')} style",
            "mbti": features.get("mbti") or "N/A",
        }
        for key in BIG5_KEYS:
            summary[key] = features.get(key, 0.5)
            summary[f"{key}_confidence"] = max(
                features.get("communication_confidence", 0.72),
                features.get("goals_confidence", 0.72),
                features.get("values_confidence", 0.72),
            )
        personas[persona_id] = _normalize_persona(summary)
    return personas


def _synthesize_persona(persona_id: str) -> dict:
    if persona_id.startswith("bot_") and persona_id[4:].isdigit():
        idx = int(persona_id[4:])
        styles = ["casual", "warm", "direct", "formal", "humorous", "serious"]
        locations = ["London", "New York", "Seoul", "Tokyo", "Shanghai", "Berlin"]
        return _normalize_persona(
            {
                "profile_id": persona_id,
                "name": f"Persona {idx}",
                "age": 24 + (idx % 15),
                "sex": "F" if idx % 2 == 0 else "M",
                "location": locations[idx % len(locations)],
                "communication_style": styles[idx % len(styles)],
                "interests": ["music", "travel", "coffee"],
                "personality_summary": "Fallback social persona for demo mode",
                "mbti": "ENFP",
                "big_five_openness": 0.55 + ((idx % 5) * 0.07),
                "big_five_conscientiousness": 0.48 + ((idx % 4) * 0.09),
                "big_five_extraversion": 0.44 + ((idx % 6) * 0.08),
                "big_five_agreeableness": 0.52 + ((idx % 5) * 0.06),
                "big_five_neuroticism": 0.30 + ((idx % 4) * 0.07),
            }
        )
    if persona_id == "bot_expert_psych":
        return next(p for p in FALLBACK_PERSONAS if p["id"] == persona_id)
    if persona_id == "bot_expert_career":
        return next(p for p in FALLBACK_PERSONAS if p["id"] == persona_id)
    return next(p for p in FALLBACK_PERSONAS if p["id"] == "bot_mirror_self")


def _build_persona_catalog() -> dict[str, dict]:
    raw_personas = _load_personas_from_file()
    backend_personas = _load_personas_from_backend()

    if backend_personas:
        merged = dict(raw_personas)
        for pid, summary in backend_personas.items():
            merged[pid] = _normalize_persona(summary, merged.get(pid))
    elif raw_personas:
        merged = raw_personas
    else:
        merged = {p["id"]: dict(p) for p in FALLBACK_PERSONAS}

    required_ids = []
    for persona_ids in MODE_PERSONA_IDS.values():
        required_ids.extend(persona_ids)
    for persona_id in required_ids:
        merged.setdefault(persona_id, _synthesize_persona(persona_id))

    return merged


PERSONA_CATALOG = _build_persona_catalog()


def _personas_for_mode(mode: str) -> list[dict]:
    persona_ids = MODE_PERSONA_IDS.get(mode, MODE_PERSONA_IDS["Social Chat"])
    return [PERSONA_CATALOG[pid] for pid in persona_ids if pid in PERSONA_CATALOG]


def _persona_choices(mode: str) -> list[tuple[str, str]]:
    choices = []
    for persona in _personas_for_mode(mode):
        sex_emoji = SEX_EMOJI.get(persona.get("sex") or "", "👤")
        style_emoji = STYLE_EMOJI.get(persona.get("communication_style", ""), "💬")
        label = f"{sex_emoji} {persona.get('name', persona['id'])} {style_emoji} · {persona['id']}"
        choices.append((label, persona["id"]))
    return choices


def _persona_by_id(persona_id: str | None) -> dict | None:
    if not persona_id:
        return None
    return PERSONA_CATALOG.get(persona_id)


# ---------------------------------------------------------------------------
# Visualization builders
# ---------------------------------------------------------------------------

def make_personality_radar(features: dict, confidences: dict) -> go.Figure:
    values = [features.get(key, 0.5) for key in BIG5_KEYS]
    conf = [confidences.get(key, 0.0) for key in BIG5_KEYS]
    closed_labels = BIG5_LABELS + [BIG5_LABELS[0]]
    closed_values = values + [values[0]]
    closed_conf = conf + [conf[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=closed_values,
            theta=closed_labels,
            fill="toself",
            name="🎯 Predicted",
            line=dict(color=ACCENT_BLUE, width=3),
            fillcolor="rgba(0, 123, 255, 0.18)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=closed_conf,
            theta=closed_labels,
            name="🪄 Confidence",
            line=dict(color="rgba(91, 107, 130, 0.9)", dash="dot", width=2),
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="rgba(0, 123, 255, 0.12)",
                linecolor="rgba(0, 123, 255, 0.18)",
                tickfont_size=10,
            ),
            angularaxis=dict(
                gridcolor="rgba(0, 123, 255, 0.12)",
                tickfont_size=11,
            ),
        ),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.14),
        height=340,
        **LIGHT_LAYOUT,
    )
    return fig


def make_emotion_timeline(emotion_history: list[dict]) -> go.Figure:
    fig = go.Figure()

    if not emotion_history:
        fig.add_annotation(
            text="Emotion data will appear as you chat",
            x=0.5,
            y=0.5,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(color=MUTED, size=13),
        )
        fig.update_layout(height=240, **LIGHT_LAYOUT)
        return fig

    turns = [point["turn"] for point in emotion_history]
    emotions = [point["emotion"] for point in emotion_history]
    valences = [EMOTION_VALENCE.get(emotion, 0.0) for emotion in emotions]
    colors = [EMOTION_COLORS.get(emotion, "#94A3B8") for emotion in emotions]

    fig.add_trace(
        go.Scatter(
            x=turns,
            y=valences,
            mode="lines",
            line=dict(color="rgba(0, 123, 255, 0.22)", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 123, 255, 0.08)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=turns,
            y=valences,
            mode="markers+lines",
            marker=dict(color=colors, size=10, line=dict(color="#FFFFFF", width=1)),
            line=dict(color="rgba(0, 123, 255, 0.5)", width=1.6, dash="dot"),
            text=emotions,
            hovertemplate="<b>Turn %{x}</b><br>%{text}<br>Valence: %{y:.2f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(91, 107, 130, 0.55)")
    fig.update_layout(
        xaxis=dict(title="Conversation Turn", showgrid=False, tickfont_size=10),
        yaxis=dict(
            title="Valence",
            range=[-1.15, 1.15],
            gridcolor="rgba(0, 123, 255, 0.09)",
            zeroline=False,
            tickfont_size=10,
        ),
        height=240,
        **LIGHT_LAYOUT,
    )
    return fig


def make_pipeline_chart(agent_costs: dict) -> go.Figure:
    fig = go.Figure()

    if not agent_costs:
        fig.add_annotation(
            text="Pipeline stats will appear after the first backend response",
            x=0.5,
            y=0.5,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(color=MUTED, size=13),
        )
        fig.update_layout(height=260, **LIGHT_LAYOUT)
        return fig

    agents = list(agent_costs.keys())
    in_toks = [agent_costs[name].get("input_tokens", 0) for name in agents]
    out_toks = [agent_costs[name].get("output_tokens", 0) for name in agents]
    cost_labels = [f"${agent_costs[name].get('cost', 0.0):.4f}" for name in agents]

    fig.add_trace(
        go.Bar(
            y=agents,
            x=in_toks,
            name="📥 Input tokens",
            orientation="h",
            marker_color="rgba(0, 123, 255, 0.78)",
            hovertemplate="%{y}<br>Input: %{x} tokens<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=agents,
            x=out_toks,
            name="📤 Output tokens",
            orientation="h",
            marker_color="rgba(59, 130, 246, 0.38)",
            customdata=cost_labels,
            hovertemplate="%{y}<br>Output: %{x} tokens<br>Cost: %{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="Tokens", gridcolor="rgba(0, 123, 255, 0.09)", tickfont_size=10),
        yaxis=dict(tickfont_size=10),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2),
        height=max(240, len(agents) * 34 + 80),
        **LIGHT_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def _badge(text: str, color: str = ACCENT_BLUE, alpha: float = 0.12) -> str:
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    return (
        f'<span style="display:inline-flex;align-items:center;gap:4px;'
        f'padding:4px 10px;border-radius:999px;'
        f'border:1px solid rgba({r},{g},{b},0.18);'
        f'background:rgba({r},{g},{b},{alpha});'
        f'font-size:12px;color:{color};font-weight:600;">{text}</span>'
    )


def _card(title: str, body: str) -> str:
    return (
        f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:16px;'
        f'padding:14px 16px;">'
        f'<div style="font-size:11px;font-weight:700;color:{ACCENT_BLUE};'
        f'text-transform:uppercase;letter-spacing:0.09em;margin-bottom:10px;">{title}</div>'
        f'{body}</div>'
    )


def make_personality_info_html(features: dict, ctx=None) -> str:
    mbti = features.get("mbti", "N/A")
    communication_style = _safe_title(features.get("communication_style", "casual")) or "Casual"
    attachment_style = _safe_title(features.get("attachment_style")) or "Unknown"
    summary = features.get("personality_summary", "Personality signals will sharpen as the conversation grows.")
    accuracy = getattr(ctx, "perception_accuracy", 0.0) if ctx else 0.0
    turn = getattr(ctx, "turn_count", 0) if ctx else 0
    dims = getattr(ctx, "dimensions_resolved", 0) if ctx else 0
    total = getattr(ctx, "total_dimensions", 22) if ctx else 22
    cross_mode = getattr(ctx, "cross_mode_consistency", 0.0) if ctx else 0.0

    badges = [
        _badge(f"🧠 {mbti}"),
        _badge(f"💬 {communication_style}"),
        _badge(f"🔗 {attachment_style}"),
        _badge(f"📊 {accuracy:.0%} accuracy", "#10B981"),
        _badge(f"🧩 {dims}/{total} dimensions", "#F59E0B"),
        _badge(f"🔄 Turn {turn}", "#64748B", 0.10),
        _badge(f"🪞 {cross_mode:.0%} cross-mode", "#8B5CF6"),
    ]
    body = (
        f'<div style="font-size:14px;line-height:1.6;color:{TEXT};margin-bottom:10px;">{summary}</div>'
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;">{"".join(badges)}</div>'
    )
    return _card("Personality Insights", body)


def make_relationship_html(ctx) -> str:
    current = (getattr(ctx, "rel_status", None) or "stranger").lower()
    rel_type = (getattr(ctx, "rel_type", None) or "other").lower()
    sentiment = (getattr(ctx, "sentiment_label", None) or "neutral").lower()
    can_advance = getattr(ctx, "can_advance", None)
    current_idx = REL_STAGES.index(current) if current in REL_STAGES else 0

    sentiment_icon = {"positive": "😊", "neutral": "😐", "negative": "😟"}.get(sentiment, "😐")
    advance_label = (
        "Ready to deepen" if can_advance is True else "Hold steady" if can_advance is False else "Still calibrating"
    )
    bars = []
    labels = []
    for idx, label in enumerate(REL_STAGE_LABELS):
        color = ACCENT_BLUE if idx <= current_idx else BORDER
        opacity = 1.0 if idx == current_idx else 0.75 if idx < current_idx else 0.5
        bars.append(
            f'<div style="flex:1;height:8px;border-radius:999px;background:{color};opacity:{opacity};"></div>'
        )
        labels.append(
            f'<span style="flex:1;text-align:center;font-size:11px;'
            f'color:{ACCENT_BLUE if idx == current_idx else MUTED};">{label}</span>'
        )

    body = (
        f'<div style="display:flex;gap:6px;margin-bottom:8px;">{"".join(labels)}</div>'
        f'<div style="display:flex;gap:6px;margin-bottom:14px;">{"".join(bars)}</div>'
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;">'
        f'{_badge(f"❤️ {REL_STAGE_LABELS[current_idx]}")}'
        f'{_badge(f"{sentiment_icon} {sentiment.title()}", "#EC4899")}'
        f'{_badge(f"🧭 {advance_label}", "#10B981" if can_advance else "#F59E0B", 0.10)}'
        f'{_badge(f"🤝 {_safe_title(rel_type) or "Other"}", "#64748B", 0.10)}'
        f'</div>'
    )
    return _card("Relationship Status", body)


def make_memory_html(stats: dict) -> str:
    stats = stats or {}
    working = stats.get("working_memory_size", stats.get("working_memory_count", 0))
    episodic = stats.get("episodic_memory_count", stats.get("episodic_count", 0))
    semantic = stats.get("semantic_memory_count", stats.get("semantic_count", 0))
    total = stats.get("total_turns_covered", stats.get("total_memories", 0))
    compression = stats.get("compression_ratio", 0.0)

    layers = [
        ("⚡ Working", f"{working} active items in the short-term window", ACCENT_BLUE),
        ("🗂️ Episodic", f"{episodic} compressed episodes retained", "#F59E0B"),
        ("🧠 Semantic", f"{semantic} higher-level reflections stored", "#10B981"),
    ]
    body_parts = []
    for title, detail, color in layers:
        body_parts.append(
            f'<div style="border-left:4px solid {color};background:{SURFACE};'
            f'padding:10px 12px;border-radius:0 12px 12px 0;margin-bottom:8px;">'
            f'<div style="font-size:13px;font-weight:700;color:{color};margin-bottom:2px;">{title}</div>'
            f'<div style="font-size:12px;color:{MUTED};">{detail}</div>'
            f'</div>'
        )
    body_parts.append(
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;">'
        f'{_badge(f"🪵 {total} turns covered", "#64748B", 0.10)}'
        f'{_badge(f"🌀 {compression:.1f}x compression", "#8B5CF6", 0.10)}'
        f'</div>'
    )
    return _card("Three-Layer Memory", "".join(body_parts))


def make_privacy_html(report: dict) -> str:
    report = report or {
        "epsilon": 1.0,
        "privacy_level": "moderate",
        "feature_tiers": {"public": 6, "private": 10, "sensitive": 6},
        "forgotten_memories": 0,
        "forgotten_features": [],
        "consent_summary": {"consented": [], "denied": [], "total_decisions": 0},
    }
    consent = report.get("consent_summary", {})
    tiers = report.get("feature_tiers", {})
    forgotten_features = report.get("forgotten_features", [])

    body = (
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;">'
        f'{_badge(f"🔐 {str(report.get("privacy_level", "moderate")).title()} privacy")}'
        f'{_badge(f"ε {report.get("epsilon", 1.0):.2f}", "#8B5CF6", 0.10)}'
        f'{_badge(f"🧾 {consent.get("total_decisions", 0)} consent checks", "#10B981", 0.10)}'
        f'</div>'
        f'<div style="font-size:13px;line-height:1.65;color:{TEXT};">'
        f'Public: {tiers.get("public", 0)} · Private: {tiers.get("private", 0)} · '
        f'Sensitive: {tiers.get("sensitive", 0)}'
        f'</div>'
        f'<div style="font-size:12px;line-height:1.6;color:{MUTED};margin-top:8px;">'
        f'Forgotten memories: {report.get("forgotten_memories", 0)}'
        f' · Forgotten features: {len(forgotten_features)}'
        f'</div>'
    )
    return _card("Privacy Report", body)


def make_partner_html(persona: dict | None, mode: str, backend_live: bool) -> str:
    if not persona:
        return _card(
            "Choose Partner",
            f'<div style="font-size:13px;color:{MUTED};line-height:1.7;">'
            f'Select a mode, pick a persona, and the conversation will open with a greeting.</div>',
        )

    sex_emoji = SEX_EMOJI.get(persona.get("sex") or "", "👤")
    style_emoji = STYLE_EMOJI.get(persona.get("communication_style", ""), "💬")
    chips = [
        _badge(f"{MODE_EMOJI.get(mode, '💬')} {mode}"),
        _badge(f"{style_emoji} {_safe_title(persona.get('communication_style')) or 'Casual'}", "#0EA5E9", 0.10),
    ]
    if persona.get("location"):
        chips.append(_badge(f"📍 {persona['location']}", "#64748B", 0.10))
    if persona.get("job"):
        chips.append(_badge(f"💼 {persona['job']}", "#10B981", 0.10))
    body = (
        f'<div style="font-size:15px;font-weight:700;color:{TEXT};margin-bottom:6px;">'
        f'{sex_emoji} {persona.get("name", persona["id"])}'
        f'</div>'
        f'<div style="font-size:13px;line-height:1.65;color:{MUTED};margin-bottom:10px;">'
        f'{persona.get("personality_summary", "Conversational partner")}'
        f'</div>'
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;">{"".join(chips)}</div>'
        f'<div style="font-size:12px;color:{MUTED};">'
        f'Backend: {"online" if backend_live else "fallback"}'
        f'</div>'
    )
    return _card("Active Partner", body)


# ---------------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------------

class _EmptyCtx:
    predicted_features = {}
    feature_confidences = {}
    perception_accuracy = 0.0
    turn_count = 0
    dimensions_resolved = 0
    total_dimensions = 22
    rel_status = "stranger"
    rel_type = "other"
    sentiment_label = "neutral"
    can_advance = None
    cross_mode_consistency = 0.0
    personality_consistency_score = 0.0


def _initial_radar():
    return make_personality_radar({}, {})


def _initial_pipeline():
    return make_pipeline_chart({})


def _initial_dashboard() -> tuple:
    empty_ctx = _EmptyCtx()
    return (
        0.0,
        0,
        "",
        _initial_radar(),
        make_personality_info_html({}, empty_ctx),
        make_relationship_html(empty_ctx),
        make_memory_html({}),
        make_privacy_html({}),
        _initial_pipeline(),
    )


def _confidence_map(features: dict, persona: dict | None = None) -> dict:
    persona = persona or {}
    return {
        key: float(persona.get(f"{key}_confidence", 0.76 if features.get(key) else 0.0))
        for key in BIG5_KEYS
    }


def _build_dashboard(result: dict, ctx, emotions: list[dict], persona: dict | None = None) -> tuple:
    features = getattr(ctx, "predicted_features", {}) or {}
    confidences = getattr(ctx, "feature_confidences", {}) or _confidence_map(features, persona)
    emotion_text = _safe_title(result.get("emotion", {}).get("current_emotion", {}).get("emotion")) or "Neutral"
    privacy_report = result.get("privacy_report", {})

    return (
        round(float(getattr(ctx, "personality_consistency_score", 0.0) or 0.0), 2),
        int(getattr(ctx, "turn_count", 0) or 0),
        emotion_text,
        make_personality_radar(features, confidences),
        make_personality_info_html(features, ctx),
        make_relationship_html(ctx),
        make_memory_html(result.get("memory_stats", {})),
        make_privacy_html(privacy_report),
        make_pipeline_chart(result.get("agent_costs", {})),
    )


# ---------------------------------------------------------------------------
# Fallback conversation helpers
# ---------------------------------------------------------------------------

def _detect_emotion(message: str) -> tuple[str, float, float]:
    text = message.lower()
    keyword_groups = [
        ("love", ["love", "adore", "romantic", "crush"], 0.89, 0.82),
        ("joy", ["happy", "glad", "excited", "great", "awesome"], 0.84, 0.74),
        ("trust", ["trust", "safe", "secure"], 0.78, 0.66),
        ("anxiety", ["anxious", "nervous", "stress", "worried"], 0.81, 0.76),
        ("sadness", ["sad", "down", "lonely", "hurt"], 0.83, 0.72),
        ("anger", ["angry", "mad", "annoyed", "frustrated"], 0.82, 0.79),
        ("interest", ["curious", "wonder", "think", "maybe"], 0.72, 0.54),
    ]
    for emotion, keywords, confidence, intensity in keyword_groups:
        if any(keyword in text for keyword in keywords):
            return emotion, confidence, intensity
    return "neutral", 0.64, 0.34


def _fallback_reply(persona: dict, mode: str, message: str, turn_count: int, emotion: str) -> str:
    summary = persona.get("personality_summary", "thoughtful conversational partner")
    interests = ", ".join(persona.get("interests", [])[:2])
    if persona["id"] == "bot_expert_psych":
        return (
            f"🧠 I hear a {emotion} note in that. Before we solve it, what feels most charged for you right now, "
            f"and what would feeling a little more grounded look like in this situation?"
        )
    if persona["id"] == "bot_expert_career":
        return (
            f"📈 Strategic read: your message suggests both pressure and opportunity. "
            f"What outcome do you want in the next 90 days, and what constraint is actually blocking the first move?"
        )
    if persona["id"] == "bot_mirror_self":
        return (
            f"🪞 If I mirror that back, I hear you circling something important. "
            f"What are you already convinced is true here, and what part of that belief deserves a second look?"
        )

    opener = {
        "joy": "😊 That energy comes through clearly.",
        "love": "💖 That feels warm and sincere.",
        "anxiety": "🌧️ That sounds heavier than you're letting on.",
        "sadness": "🤍 There's real weight in that.",
        "anger": "🔥 I can hear the edge in that.",
        "interest": "✨ That's an interesting angle.",
        "neutral": "💬 I'm with you.",
    }.get(emotion, "💬 I'm with you.")
    follow_up = (
        f"{persona.get('name', persona['id'])} feels like a {summary.lower()}."
        if turn_count == 1
        else f"I'd lean into {interests or 'what matters most to you'} from here."
    )
    return f"{opener} {follow_up} What part do you want to explore next?"


def _fallback_context(persona: dict, mode: str, turn_count: int, emotion: str) -> SimpleNamespace:
    valence = EMOTION_VALENCE.get(emotion, 0.0)
    stage_idx = min(turn_count // 3, len(REL_STAGES) - 1)
    rel_type = {
        "Social Chat": "romantic_interest",
        "Expert Consulting": "professional_support",
        "Self-Dialogue": "self_reflection",
    }.get(mode, "other")
    features = {
        key: float(persona.get(key, 0.5))
        for key in BIG5_KEYS
    }
    features.update(
        {
            "mbti": persona.get("mbti", "N/A"),
            "communication_style": persona.get("communication_style", "casual"),
            "personality_summary": persona.get("personality_summary", ""),
            "attachment_style": "emerging_pattern",
        }
    )
    return SimpleNamespace(
        predicted_features=features,
        feature_confidences=_confidence_map(features, persona),
        perception_accuracy=min(0.62 + (turn_count * 0.05), 0.93),
        turn_count=turn_count,
        dimensions_resolved=min(4 + turn_count * 2, 22),
        total_dimensions=22,
        rel_status=REL_STAGES[stage_idx],
        rel_type=rel_type,
        sentiment_label="positive" if valence > 0.18 else "negative" if valence < -0.18 else "neutral",
        can_advance=True if turn_count >= 3 else None,
        cross_mode_consistency=min(0.70 + (turn_count * 0.04), 0.95),
        personality_consistency_score=min(0.78 + (turn_count * 0.03), 0.96),
    )


def _fallback_result(persona: dict, mode: str, message: str, emotions: list[dict], turn_count: int) -> tuple[dict, SimpleNamespace]:
    emotion, confidence, intensity = _detect_emotion(message)
    ctx = _fallback_context(persona, mode, turn_count, emotion)
    result = {
        "turn": turn_count,
        "bot_message": _fallback_reply(persona, mode, message, turn_count, emotion),
        "emotion": {
            "current_emotion": {
                "emotion": emotion,
                "confidence": confidence,
                "intensity": intensity,
            }
        },
        "conversation_sentiment": {
            "label": ctx.sentiment_label,
            "score": round(EMOTION_VALENCE.get(emotion, 0.0), 2),
            "trend": "improving" if turn_count > 1 and EMOTION_VALENCE.get(emotion, 0.0) >= 0 else "stable",
        },
        "memory_stats": {
            "working_memory_size": min(turn_count + 1, 8),
            "episodic_memory_count": turn_count // 3,
            "semantic_memory_count": turn_count // 5,
            "total_turns_covered": turn_count,
            "compression_ratio": round(max(turn_count // 3, 1) * 10 / max(turn_count, 1), 2),
        },
        "privacy_report": {
            "epsilon": 1.0,
            "privacy_level": "moderate",
            "feature_tiers": {"public": 6, "private": 10, "sensitive": 6},
            "forgotten_memories": 0,
            "forgotten_features": [],
            "consent_summary": {"consented": [], "denied": [], "total_decisions": 0},
        },
        "agent_costs": {},
    }
    return result, ctx


# ---------------------------------------------------------------------------
# UI styling
# ---------------------------------------------------------------------------

CSS = f"""
:root {{
    --accent-blue: {ACCENT_BLUE};
    --surface: {SURFACE};
    --card-bg: {CARD_BG};
    --border: {BORDER};
    --text-main: {TEXT};
    --text-muted: {MUTED};
}}

body, .gradio-container {{
    background: linear-gradient(180deg, #f9fbff 0%, #f3f7fc 100%) !important;
    color: var(--text-main) !important;
}}

.gradio-container {{
    max-width: 1380px !important;
}}

.hero {{
    padding: 18px 20px 10px;
    border: 1px solid var(--border);
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(0,123,255,0.12), rgba(255,255,255,0.94));
    margin-bottom: 14px;
}}

.hero h1 {{
    margin: 0 0 8px !important;
    color: var(--accent-blue) !important;
}}

.hero p {{
    margin: 0 !important;
    color: var(--text-muted) !important;
    font-size: 14px;
}}

.section-title h3 {{
    color: var(--accent-blue) !important;
    margin-bottom: 0.2rem !important;
}}

.demo-panel, .analysis-panel {{
    border: 1px solid var(--border);
    border-radius: 20px;
    background: rgba(255,255,255,0.9);
    padding: 14px;
}}

.analysis-panel .gr-accordion,
.demo-panel .gr-accordion,
.metric-row > div,
.gr-chatbot,
.gr-plot,
.gr-html,
.gr-textbox,
.gr-dropdown,
.gr-radio {{
    border-radius: 16px !important;
}}

.metric-row .gr-number,
.metric-row .gr-textbox {{
    min-height: 92px;
}}

.gr-chatbot {{
    border: 1px solid var(--border) !important;
    background: var(--card-bg) !important;
}}

.gr-accordion {{
    border: 1px solid var(--border) !important;
    background: rgba(255,255,255,0.82) !important;
}}

.gr-button-primary {{
    box-shadow: 0 10px 28px rgba(0,123,255,0.16) !important;
}}

.footer-note {{
    color: var(--text-muted);
    font-size: 12px;
}}
"""

ABOUT_MD = """
### 📖 About AI-YOU

AI-YOU is a multi-agent conversational system for personality inference through natural dialogue. The demo shows how a coordinated agent stack can infer Big Five signals, track emotional dynamics, maintain structured memory, and estimate relationship progression while the chat unfolds.

### ✨ Key Innovations

- **🧠 Multi-agent inference**: specialized agents divide work across personality prediction, emotion reading, memory, sentiment, and relationship estimation.
- **📡 Real-time personality tracking**: the dashboard updates trait confidence and consistency after each user turn.
- **🗂️ Three-layer memory**: working, episodic, and semantic memory layers preserve conversational context at different granularities.
- **🔒 Privacy-aware design**: privacy reporting exposes differential privacy level, consent activity, and memory handling signals.

### 🏗️ Architecture Overview

1. **User message intake** routes the turn into the orchestrator.
2. **Parallel analysis** runs emotion, memory retrieval, and safety checks.
3. **Feature update** refines personality beliefs and confidence estimates.
4. **Response generation** selects the active persona and produces the next reply.
5. **Dashboard refresh** projects the latest state into personality, relationship, memory, privacy, and pipeline views.
"""


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

with gr.Blocks(title="AI-YOU · Personal Digital Division") as demo:
    state_orch = gr.State(None)
    state_emotions = gr.State([])
    state_last = gr.State({})
    state_active_bot = gr.State(None)

    gr.Markdown(
        """
<div class="hero">
  <h1>🧠 AI-YOU: Personal Digital Division</h1>
  <p>Multi-agent personality inference through natural conversation |
  <a href="#">Paper</a> |
  <a href="https://github.com/YanLin-Quinne/AI-YOU">GitHub</a> |
  EMNLP 2026 Demo</p>
</div>
        """
    )

    with gr.Tab("💬 Demo"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, elem_classes=["demo-panel"]):
                gr.Markdown("### 🎭 Choose Your Experience", elem_classes=["section-title"])
                mode_radio = gr.Radio(
                    ["Social Chat", "Expert Consulting", "Self-Dialogue"],
                    value="Social Chat",
                    label="💫 Conversation Mode",
                )
                partner_dropdown = gr.Dropdown(
                    choices=_persona_choices("Social Chat"),
                    value=None,
                    label="🤝 Choose Partner",
                    interactive=True,
                    info="Select a persona to open the conversation.",
                )
                active_partner = gr.HTML(value=make_partner_html(None, "Social Chat", BACKEND))
                gr.Markdown("---")
                chatbot = gr.Chatbot(label="💬 Conversation", height=420)
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type a message...",
                        scale=4,
                        show_label=False,
                    )
                    send_btn = gr.Button("🚀 Send", variant="primary", scale=1)
                change_partner_btn = gr.Button("🔄 Change Partner", variant="secondary", size="sm")

            with gr.Column(scale=1, elem_classes=["analysis-panel"]):
                gr.Markdown("### 📊 Real-time Analysis", elem_classes=["section-title"])
                with gr.Row(equal_height=True, elem_classes=["metric-row"]):
                    consistency_num = gr.Number(label="🎯 Consistency", precision=2, value=0.0, interactive=False)
                    turn_num = gr.Number(label="💬 Turn", precision=0, value=0, interactive=False)
                    emotion_box = gr.Textbox(label="😊 Emotion", value="", max_lines=1, interactive=False)
                personality_plot = gr.Plot(label="🕸️ Big Five Personality", value=_initial_radar())
                personality_info = gr.HTML(value=make_personality_info_html({}, _EmptyCtx()))

                with gr.Accordion("❤️ Relationship Prediction", open=False):
                    relationship_html = gr.HTML(value=make_relationship_html(_EmptyCtx()))

                with gr.Accordion("🧠 Memory System", open=False):
                    memory_html = gr.HTML(value=make_memory_html({}))

                with gr.Accordion("🔒 Privacy & Consistency", open=False):
                    privacy_html = gr.HTML(value=make_privacy_html({}))

                with gr.Accordion("⚙️ Pipeline Stats", open=False):
                    pipeline_plot = gr.Plot(label="📦 Token Usage", value=_initial_pipeline())
                    gr.Markdown(
                        "<div class='footer-note'>Input and output token usage per agent stage appears here when the live backend is active.</div>"
                    )

    with gr.Tab("📖 About"):
        gr.Markdown(ABOUT_MD)

    DASH_OUTS = [
        consistency_num,
        turn_num,
        emotion_box,
        personality_plot,
        personality_info,
        relationship_html,
        memory_html,
        privacy_html,
        pipeline_plot,
    ]

    SELECT_OUTS = [
        state_active_bot,
        state_orch,
        state_emotions,
        state_last,
        chatbot,
        active_partner,
    ] + DASH_OUTS

    SEND_OUTS = [
        chatbot,
        msg_input,
        state_orch,
        state_emotions,
        state_last,
    ] + DASH_OUTS

    RESET_OUTS = [
        partner_dropdown,
        msg_input,
        state_active_bot,
        state_orch,
        state_emotions,
        state_last,
        chatbot,
        active_partner,
    ] + DASH_OUTS

    def _reset_view(mode: str) -> tuple:
        return (
            gr.update(choices=_persona_choices(mode), value=None),
            "",
            None,
            None,
            [],
            {},
            [],
            make_partner_html(None, mode, BACKEND),
            *_initial_dashboard(),
        )

    async def on_mode_change(mode: str):
        return _reset_view(mode)

    async def on_partner_select(mode: str, bot_id: str | None):
        if not bot_id:
            return (
                None,
                None,
                [],
                {},
                [],
                make_partner_html(None, mode, BACKEND),
                *_initial_dashboard(),
            )

        persona = _persona_by_id(bot_id) or {"id": bot_id, "name": bot_id}
        partner_html = make_partner_html(persona, mode, BACKEND)

        if not BACKEND:
            greeting = {
                "Social Chat": f"💕 Hi, I'm {persona.get('name', bot_id)}. Tell me what's on your mind.",
                "Expert Consulting": f"🧭 Hi, I'm {persona.get('name', bot_id)}. What would you like help thinking through today?",
                "Self-Dialogue": "🪞 I’m here as your reflective mirror. What part of yourself do you want to hear back more clearly?",
            }.get(mode, f"Hello, I'm {persona.get('name', bot_id)}.")
            return (
                bot_id,
                None,
                [],
                {},
                [{"role": "assistant", "content": greeting}],
                partner_html,
                *_initial_dashboard(),
            )

        try:
            pool = get_bot_pool()
            user_id = str(uuid.uuid4())[:8]
            orch = OrchestratorAgent(user_id, pool, bot_id)
            start = orch.start_new_conversation(bot_id)
            greeting = start.get("greeting", "Hello! Nice to meet you.")
            return (
                bot_id,
                orch,
                [],
                {},
                [{"role": "assistant", "content": greeting}],
                partner_html,
                *_initial_dashboard(),
            )
        except Exception as exc:
            fallback_greeting = f"⚠️ Live backend was unavailable for {bot_id}: {exc}\n\nI switched to fallback demo mode."
            return (
                bot_id,
                None,
                [],
                {},
                [{"role": "assistant", "content": fallback_greeting}],
                make_partner_html(persona, mode, False),
                *_initial_dashboard(),
            )

    async def on_send(
        message: str,
        chat_history: list,
        orch,
        emotions: list,
        last_resp: dict,
        active_bot_id: str | None,
        mode: str,
    ):
        if not message.strip():
            no_change = tuple(gr.update() for _ in DASH_OUTS)
            return (chat_history, message, orch, emotions, last_resp) + no_change

        chat_history = list(chat_history or [])
        emotions = list(emotions or [])

        if not active_bot_id:
            chat_history = chat_history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Please choose a partner first."},
            ]
            return (chat_history, "", orch, emotions, last_resp) + _initial_dashboard()

        chat_history.append({"role": "user", "content": message})
        persona = _persona_by_id(active_bot_id) or {"id": active_bot_id, "name": active_bot_id}

        if orch is None:
            turn_count = 1 + sum(1 for item in chat_history if item.get("role") == "user") - 1
            result, ctx = _fallback_result(persona, mode, message, emotions, turn_count)
            emo_data = result.get("emotion", {}).get("current_emotion", {})
            emotions.append(
                {
                    "turn": turn_count,
                    "emotion": emo_data.get("emotion", "neutral"),
                    "confidence": emo_data.get("confidence", 0.0),
                    "intensity": emo_data.get("intensity", 0.0),
                }
            )
            chat_history.append({"role": "assistant", "content": result["bot_message"]})
            return (
                chat_history,
                "",
                None,
                emotions,
                result,
                *_build_dashboard(result, ctx, emotions, persona),
            )

        try:
            result = await orch.process_user_message(message)
        except Exception as exc:
            chat_history.append({"role": "assistant", "content": f"⚠️ Error: {exc}"})
            return (chat_history, "", orch, emotions, last_resp) + _initial_dashboard()

        chat_history.append({"role": "assistant", "content": result.get("bot_message", "…")})

        emo_data = result.get("emotion", {}).get("current_emotion", {})
        if emo_data.get("emotion"):
            emotions.append(
                {
                    "turn": result.get("turn", len(emotions) + 1),
                    "emotion": emo_data["emotion"],
                    "confidence": emo_data.get("confidence", 0.0),
                    "intensity": emo_data.get("intensity", 0.0),
                }
            )

        if getattr(orch, "_pending_relationship_turn", False):
            try:
                rel_result = await orch.run_relationship_prediction()
                if rel_result.get("relationship_prediction"):
                    result["relationship_prediction"] = rel_result["relationship_prediction"]
            except Exception:
                pass

        if "privacy_report" not in result:
            try:
                result["privacy_report"] = orch.privacy_manager.get_privacy_report()
            except Exception:
                result["privacy_report"] = {}

        return (
            chat_history,
            "",
            orch,
            emotions,
            result,
            *_build_dashboard(result, orch.ctx, emotions, persona),
        )

    async def on_change_partner(mode: str):
        return _reset_view(mode)

    mode_radio.change(
        fn=on_mode_change,
        inputs=[mode_radio],
        outputs=RESET_OUTS,
    )
    partner_dropdown.change(
        fn=on_partner_select,
        inputs=[mode_radio, partner_dropdown],
        outputs=SELECT_OUTS,
    )
    send_btn.click(
        fn=on_send,
        inputs=[msg_input, chatbot, state_orch, state_emotions, state_last, state_active_bot, mode_radio],
        outputs=SEND_OUTS,
    )
    msg_input.submit(
        fn=on_send,
        inputs=[msg_input, chatbot, state_orch, state_emotions, state_last, state_active_bot, mode_radio],
        outputs=SEND_OUTS,
    )
    change_partner_btn.click(
        fn=on_change_partner,
        inputs=[mode_radio],
        outputs=RESET_OUTS,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme="soft", css=CSS)
