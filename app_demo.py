"""
app_demo.py — AI YOU Agent · Professional Demo UI
EMNLP 2025 System Demonstration

Two-column layout:
  LEFT  (40%): Persona selector → Chat interface
  RIGHT (60%): 6-tab real-time dashboard (Personality, Emotion, Relationship,
               Memory, Pipeline, Data)
"""

import asyncio
import uuid

import gradio as gr
import plotly.graph_objects as go

# ─── Backend bootstrap ────────────────────────────────────────────────────────

try:
    from src.bootstrap import create_default_bootstrap, get_bot_pool
    from src.agents.orchestrator import OrchestratorAgent

    _bootstrap = create_default_bootstrap()
    asyncio.get_event_loop().run_until_complete(_bootstrap.run_all())
    BACKEND = True
except Exception as _boot_err:
    print(f"[app_demo] Backend unavailable: {_boot_err}")
    BACKEND = False

# ─── Constants ────────────────────────────────────────────────────────────────

EMOTION_VALENCE = {
    "joy": 0.9, "excitement": 0.8, "love": 0.85, "trust": 0.6,
    "interest": 0.4, "surprise": 0.3, "neutral": 0.0,
    "anxiety": -0.5, "sadness": -0.7, "fear": -0.6,
    "anger": -0.8, "disgust": -0.9,
}
EMOTION_COLORS = {
    "joy": "#FFD700", "excitement": "#FF6B6B", "love": "#FF69B4",
    "trust": "#7EC8E3", "interest": "#90EE90", "surprise": "#00CED1",
    "neutral": "#888888", "anxiety": "#DDA0DD", "sadness": "#4169E1",
    "fear": "#9370DB", "anger": "#FF4444", "disgust": "#8B7355",
}

BIG5_KEYS = [
    "big_five_openness", "big_five_conscientiousness", "big_five_extraversion",
    "big_five_agreeableness", "big_five_neuroticism",
]
BIG5_LABELS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

REL_STAGES = ["stranger", "acquaintance", "crush", "dating", "committed"]
REL_STAGE_LABELS = ["Stranger", "Acquaintance", "Crush", "Dating", "Committed"]

_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#aaa",
    margin=dict(l=50, r=50, t=35, b=35),
)

SEX_EMOJI = {"F": "👩", "M": "👨", "female": "👩", "male": "👨"}
STYLE_EMOJI = {
    "warm": "🌸", "direct": "⚡", "formal": "🎩",
    "casual": "😊", "serious": "🧐", "humorous": "😄", "indirect": "🌀",
}

# Fallback personas when backend is unavailable
FALLBACK_PERSONAS = [
    {"id": "bot_0", "name": "Lin Wanqing",  "age": 28, "sex": "F", "location": "Beijing",   "communication_style": "warm",    "personality_summary": "Government worker, thoughtful and organized",   "interests": ["reading", "travel", "cooking"]},
    {"id": "bot_1", "name": "Chen Siyu",    "age": 32, "sex": "F", "location": "Shanghai",  "communication_style": "direct",  "personality_summary": "Administrative clerk, detail-oriented",          "interests": ["music", "yoga", "art"]},
    {"id": "bot_2", "name": "Zhao Yawen",   "age": 45, "sex": "F", "location": "Nanjing",   "communication_style": "formal",  "personality_summary": "University teacher, intellectual and patient",    "interests": ["literature", "history", "tea"]},
    {"id": "bot_3", "name": "Liu Chengyuan","age": 24, "sex": "M", "location": "Hangzhou",  "communication_style": "casual",  "personality_summary": "Programmer, curious and energetic",              "interests": ["tech", "gaming", "hiking"]},
    {"id": "bot_4", "name": "Wang Jianye",  "age": 38, "sex": "M", "location": "Chengdu",   "communication_style": "serious", "personality_summary": "Business manager, strategic thinker",            "interests": ["finance", "sports", "travel"]},
    {"id": "bot_5", "name": "Zhang Yuxuan", "age": 29, "sex": "M", "location": "Shenzhen",  "communication_style": "humorous","personality_summary": "Finance professional, witty and ambitious",      "interests": ["investing", "coffee", "photography"]},
    {"id": "bot_6", "name": "Li Xinyi",     "age": 52, "sex": "F", "location": "Guangzhou", "communication_style": "warm",    "personality_summary": "Freelancer, creative and independent",           "interests": ["painting", "gardening", "meditation"]},
    {"id": "bot_7", "name": "Zhou Mingxuan","age": 21, "sex": "M", "location": "Wuhan",     "communication_style": "casual",  "personality_summary": "Designer, expressive and spontaneous",           "interests": ["design", "music", "anime"]},
]


EXPERT_PERSONAS = [
    {"id": "bot_expert_psych",  "name": "Dr. Maya Liu",  "age": 42, "sex": "F", "location": "San Francisco", "communication_style": "formal",  "personality_summary": "Clinical psychologist, empathetic and insightful", "interests": ["psychology", "mindfulness", "research"], "mode": "expert"},
    {"id": "bot_expert_career", "name": "Alex Park",      "age": 36, "sex": "M", "location": "New York",      "communication_style": "direct",  "personality_summary": "Career strategist, analytical and motivating",    "interests": ["strategy", "leadership", "coaching"],   "mode": "expert"},
]

SELF_DIALOGUE_PERSONAS = [
    {"id": "bot_mirror_self", "name": "Mirror Self", "age": None, "sex": None, "location": None, "communication_style": "casual", "personality_summary": "Your digital twin — reflects your own personality", "interests": ["self-reflection"], "mode": "self_dialogue"},
]


def _load_personas() -> list[dict]:
    if not BACKEND:
        for p in FALLBACK_PERSONAS:
            p["mode"] = "social"
        return FALLBACK_PERSONAS
    try:
        pool = get_bot_pool()
        summaries = pool.get_agent_summaries()
        result = []
        for pid, s in list(summaries.items())[:8]:
            s["id"] = pid
            s["mode"] = "social"
            result.append(s)
        return result if result else FALLBACK_PERSONAS
    except Exception:
        for p in FALLBACK_PERSONAS:
            p["mode"] = "social"
        return FALLBACK_PERSONAS


ALL_PERSONAS = _load_personas() + EXPERT_PERSONAS + SELF_DIALOGUE_PERSONAS
PERSONAS = [p for p in ALL_PERSONAS if p.get("mode") == "social"]  # default view


def _filter_personas_by_mode(mode: str) -> list[dict]:
    mode_map = {"🗣️ Social Chat": "social", "🎓 Expert Consulting": "expert", "🪞 Self-Dialogue": "self_dialogue"}
    target = mode_map.get(mode, "social")
    return [p for p in ALL_PERSONAS if p.get("mode") == target]

# ─── Visualization builders ───────────────────────────────────────────────────


def make_personality_radar(features: dict, confidences: dict) -> go.Figure:
    values = [features.get(k, 0.5) for k in BIG5_KEYS]
    conf = [confidences.get(k, 0.0) for k in BIG5_KEYS]
    cats_closed = BIG5_LABELS + [BIG5_LABELS[0]]
    vals_closed = values + [values[0]]
    conf_closed = conf + [conf[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed, fill="toself", name="Predicted",
        line=dict(color="#6C63FF", width=2),
        fillcolor="rgba(108,99,255,0.18)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=conf_closed, theta=cats_closed, name="Confidence",
        line=dict(color="rgba(255,255,255,0.25)", dash="dot", width=1.5),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], color="#444",
                            showticklabels=True, tickfont_size=8, gridcolor="#2a2b3d"),
            angularaxis=dict(color="#666", gridcolor="#2a2b3d"),
        ),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12,
                    font_size=10, font_color="#888"),
        height=310,
        **_DARK_LAYOUT,
    )
    return fig


def make_emotion_timeline(emotion_history: list[dict]) -> go.Figure:
    fig = go.Figure()

    if not emotion_history:
        fig.add_annotation(
            text="Emotion data will appear as you chat",
            x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper",
            font=dict(color="#444", size=12),
        )
        fig.update_layout(height=240, **_DARK_LAYOUT)
        return fig

    turns = [e["turn"] for e in emotion_history]
    valences = [EMOTION_VALENCE.get(e["emotion"], 0.0) for e in emotion_history]
    emotions = [e["emotion"] for e in emotion_history]
    colors = [EMOTION_COLORS.get(em, "#888") for em in emotions]

    # Background fill
    fig.add_trace(go.Scatter(
        x=turns, y=valences, mode="lines",
        line=dict(color="rgba(108,99,255,0.3)", width=1),
        fill="tozeroy", fillcolor="rgba(108,99,255,0.07)",
        showlegend=False, hoverinfo="skip",
    ))
    # Emotion markers
    fig.add_trace(go.Scatter(
        x=turns, y=valences, mode="markers+lines",
        marker=dict(color=colors, size=9,
                    line=dict(color="rgba(255,255,255,0.2)", width=1)),
        line=dict(color="rgba(108,99,255,0.4)", width=1.5, dash="dot"),
        text=emotions,
        hovertemplate="<b>Turn %{x}</b><br>%{text}<br>Valence: %{y:.2f}<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=0, line_dash="dash",
                  line_color="rgba(255,255,255,0.12)", line_width=1)

    fig.update_layout(
        xaxis=dict(title="Conversation Turn", color="#555",
                   showgrid=False, tickfont_size=10),
        yaxis=dict(title="Valence", range=[-1.15, 1.15], color="#555",
                   gridcolor="rgba(255,255,255,0.04)", zeroline=False,
                   tickfont_size=10),
        height=240,
        **_DARK_LAYOUT,
    )
    return fig


def make_pipeline_chart(agent_costs: dict) -> go.Figure:
    fig = go.Figure()

    if not agent_costs:
        fig.add_annotation(
            text="Agent cost data will appear after the first message",
            x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper",
            font=dict(color="#444", size=12),
        )
        fig.update_layout(height=260, **_DARK_LAYOUT)
        return fig

    agents = list(agent_costs.keys())
    in_toks = [agent_costs[a].get("input_tokens", 0) for a in agents]
    out_toks = [agent_costs[a].get("output_tokens", 0) for a in agents]
    cost_labels = [f"${agent_costs[a].get('cost', 0.0):.4f}" for a in agents]

    fig.add_trace(go.Bar(
        y=agents, x=in_toks, name="Input tokens", orientation="h",
        marker_color="rgba(108,99,255,0.75)",
        hovertemplate="%{y}<br>Input: %{x} tokens<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=agents, x=out_toks, name="Output tokens", orientation="h",
        marker_color="rgba(255,107,107,0.75)",
        customdata=cost_labels,
        hovertemplate="%{y}<br>Output: %{x} tokens · Cost: %{customdata}<extra></extra>",
    ))
    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="Tokens", color="#555",
                   gridcolor="rgba(255,255,255,0.05)", tickfont_size=10),
        yaxis=dict(color="#888", tickfont_size=10),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.25,
                    font_size=10, font_color="#888"),
        height=max(220, len(agents) * 36 + 80),
        **_DARK_LAYOUT,
    )
    return fig


# ─── HTML builders ────────────────────────────────────────────────────────────

def _badge(text: str, color: str = "#6C63FF", bg_alpha: float = 0.15) -> str:
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    return (
        f'<span style="display:inline-flex;align-items:center;gap:4px;'
        f'background:rgba({r},{g},{b},{bg_alpha});'
        f'border:1px solid rgba({r},{g},{b},0.3);'
        f'border-radius:20px;padding:3px 10px;font-size:11px;color:{color};'
        f'font-weight:500;white-space:nowrap;">{text}</span>'
    )


def make_personality_info_html(features: dict, ctx=None) -> str:
    mbti = features.get("mbti", "–")
    comm = features.get("communication_style", "–")
    att = features.get("attachment_style", "–")
    accuracy = getattr(ctx, "perception_accuracy", 0.0) if ctx else 0.0
    turn = getattr(ctx, "turn_count", 0) if ctx else 0
    dims = getattr(ctx, "dimensions_resolved", 0) if ctx else 0
    total = getattr(ctx, "total_dimensions", 22) if ctx else 22

    badges = [
        _badge(f"🧠 {mbti}"),
        _badge(f"💬 {comm.title() if comm != '–' else '–'}"),
        _badge(f"🔗 {att.replace('_', ' ').title() if att != '–' else '–'}"),
        _badge(f"📊 {accuracy:.0%} accuracy", "#4ade80"),
        _badge(f"📐 {dims}/{total} dims", "#fbbf24"),
        _badge(f"🔄 Turn {turn}", "#888888", 0.1),
    ]
    return (
        '<div style="display:flex;gap:6px;flex-wrap:wrap;padding:6px 0 4px;">'
        + "".join(badges)
        + "</div>"
    )


def make_emotion_info_html(result: dict, emotion_history: list) -> str:
    emo_data = result.get("emotion", {}).get("current_emotion", {})
    emotion = emo_data.get("emotion", "neutral")
    conf = emo_data.get("confidence", 0.0)
    intensity = emo_data.get("intensity", 0.0)

    conv_sent = result.get("conversation_sentiment", {})
    if isinstance(conv_sent, dict):
        sent_label = conv_sent.get("label", "neutral")
        sent_score = conv_sent.get("score", 0.0)
        trend = conv_sent.get("trend", "stable")
    else:
        sent_label, sent_score, trend = "neutral", 0.0, "stable"

    trend_icon  = {"improving": "↑", "stable": "→", "declining": "↓"}.get(trend, "→")
    trend_color = {"improving": "#4ade80", "stable": "#fbbf24", "declining": "#f87171"}.get(trend, "#888")
    emo_color = EMOTION_COLORS.get(emotion, "#888")
    conf_w, int_w = int(conf * 100), int(intensity * 100)

    return f"""
<div style="background:#1a1b26;border:1px solid #2a2b3d;border-radius:10px;padding:12px 14px;margin-top:8px;">
  <div style="font-size:11px;font-weight:600;color:#666;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:10px;">Current State</div>
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
    {_badge(emotion.title(), emo_color)}
    <div style="flex:1;">
      <div style="font-size:9px;color:#444;margin-bottom:3px;letter-spacing:0.5px;">CONFIDENCE</div>
      <div style="background:#0f1117;border-radius:4px;height:5px;overflow:hidden;">
        <div style="background:{emo_color};width:{conf_w}%;height:100%;border-radius:4px;transition:width 0.4s;"></div>
      </div>
    </div>
    <div style="flex:1;">
      <div style="font-size:9px;color:#444;margin-bottom:3px;letter-spacing:0.5px;">INTENSITY</div>
      <div style="background:#0f1117;border-radius:4px;height:5px;overflow:hidden;">
        <div style="background:#6C63FF;width:{int_w}%;height:100%;border-radius:4px;transition:width 0.4s;"></div>
      </div>
    </div>
  </div>
  <div style="display:flex;gap:6px;flex-wrap:wrap;">
    {_badge(f'<span style="color:{trend_color}">{trend_icon}</span> {sent_label.title()}')}
    {_badge(f'Score: {sent_score:+.2f}', "#888888", 0.1)}
    {_badge(f'{len(emotion_history)} turns tracked', "#888888", 0.1)}
  </div>
</div>"""


def make_relationship_html(ctx) -> str:
    current = (getattr(ctx, "rel_status", None) or "stranger").lower()
    rel_type = (getattr(ctx, "rel_type", None) or "other").lower()
    sentiment = (getattr(ctx, "sentiment_label", None) or "neutral").lower()
    can_advance = getattr(ctx, "can_advance", None)

    current_idx = REL_STAGES.index(current) if current in REL_STAGES else 0

    sentiment_icon = {"positive": "😊", "neutral": "😐", "negative": "😟"}.get(sentiment, "😐")
    advance_icon  = "✅" if can_advance is True else ("❌" if can_advance is False else "⏳")
    advance_label = "Can Advance" if can_advance is True else ("Cannot Advance" if can_advance is False else "Advance: TBD")

    stages_html = ""
    label_html = ""
    for i, (_, label) in enumerate(zip(REL_STAGES, REL_STAGE_LABELS)):
        if i < current_idx:
            bar_style = "background:#6C63FF;"
        elif i == current_idx:
            bar_style = "background:#6C63FF;box-shadow:0 0 10px rgba(108,99,255,0.6);"
        else:
            bar_style = "background:#2a2b3d;"
        stages_html += f'<div style="flex:1;height:6px;border-radius:3px;transition:background 0.4s;{bar_style}" title="{label}"></div>'
        lc = "#6C63FF" if i == current_idx else ("#555" if i < current_idx else "#333")
        label_html += f'<span style="flex:1;text-align:center;font-size:9px;color:{lc};font-weight:{"600" if i == current_idx else "400"};">{label}</span>'

    return f"""
<div style="background:#1a1b26;border:1px solid #2a2b3d;border-radius:10px;padding:14px 16px;">
  <div style="font-size:11px;font-weight:600;color:#666;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:12px;">Relationship Progression</div>
  <div style="display:flex;margin-bottom:5px;">{label_html}</div>
  <div style="display:flex;gap:4px;margin-bottom:14px;">{stages_html}</div>
  <div style="display:flex;gap:7px;flex-wrap:wrap;">
    {_badge(f"📍 {REL_STAGE_LABELS[current_idx]}")}
    {_badge(f"{sentiment_icon} {sentiment.title()}")}
    {_badge(f"{advance_icon} {advance_label}")}
    {_badge(f"💝 {rel_type.replace('_',' ').title()}")}
  </div>
</div>"""


def make_memory_html(stats: dict) -> str:
    s = stats or {}
    total    = s.get("total_memories", 0)
    working  = s.get("working_memory_count", 0)
    episodic = s.get("episodic_count", 0)
    semantic = s.get("semantic_count", 0)

    def layer(title: str, detail: str, color: str) -> str:
        return (
            f'<div style="background:rgba(255,255,255,0.02);border-left:3px solid {color};'
            f'border-radius:0 8px 8px 0;padding:8px 12px;margin-bottom:7px;">'
            f'<div style="font-size:12px;font-weight:600;color:{color};margin-bottom:2px;">{title}</div>'
            f'<div style="font-size:11px;color:#444;">{detail}</div></div>'
        )

    return f"""
<div style="background:#1a1b26;border:1px solid #2a2b3d;border-radius:10px;padding:14px 16px;">
  <div style="font-size:11px;font-weight:600;color:#666;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:12px;">Three-Layer Cognitive Memory</div>
  {layer("Layer 1 · Working Memory", f"{working} items · Sliding window (last 20 turns)", "#6C63FF")}
  {layer("Layer 2 · Episodic Memory", f"{episodic} episodes · Compressed every 10 turns via LLM", "#fbbf24")}
  {layer("Layer 3 · Semantic Memory", f"{semantic} reflections · Feature updates every 50 turns", "#4ade80")}
  <div style="margin-top:6px;font-size:11px;color:#333;">
    Total stored: <span style="color:#555;">{total}</span> memory objects
  </div>
</div>"""


def _persona_btn_label(p: dict) -> str:
    emoji = SEX_EMOJI.get(p.get("sex", ""), "👤")
    style_emoji = STYLE_EMOJI.get(p.get("communication_style", ""), "💬")
    summary = p.get("personality_summary", "")
    if len(summary) > 55:
        summary = summary[:52] + "…"
    interests = ", ".join(p.get("interests", [])[:2])
    return (
        f"{emoji}  {p.get('name', p['id'])}\n"
        f"{p.get('age', '')} · {p.get('location', '')}\n"
        f"{style_emoji} {p.get('communication_style', '').title()}\n"
        f"{summary}"
    )


# ─── Empty/initial state helpers ─────────────────────────────────────────────

class _EmptyCtx:
    rel_status = "stranger"; rel_type = "other"
    sentiment_label = "neutral"; can_advance = None


def _initial_radar():      return make_personality_radar({}, {})
def _initial_emo_fig():    return make_emotion_timeline([])
def _initial_pipe_fig():   return make_pipeline_chart({})
def _initial_rel():        return make_relationship_html(_EmptyCtx())
def _initial_mem():        return make_memory_html({})
def _initial_pers_info():  return make_personality_info_html({})
def _initial_emo_info():   return make_emotion_info_html({}, [])


def _initial_dashboard() -> tuple:
    return (
        _initial_radar(), _initial_pers_info(),
        _initial_emo_fig(), _initial_emo_info(),
        _initial_rel(), _initial_mem(),
        _initial_pipe_fig(), {},
    )


# ─── CSS ─────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --accent:      #6C63FF;
    --accent-dim:  rgba(108,99,255,0.14);
    --accent-glow: rgba(108,99,255,0.45);
    --bg:          #0f1117;
    --bg-card:     #1a1b26;
    --bg-hover:    #22243a;
    --border:      #2a2b3d;
    --text:        #e1e1e6;
    --text-muted:  #777;
    --text-dim:    #444;
}

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}
.gradio-container { max-width: 1420px !important; margin: 0 auto !important; }

/* ── Header ── */
.demo-header {
    text-align: center;
    padding: 18px 0 12px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 14px;
    background: linear-gradient(180deg, rgba(108,99,255,0.07) 0%, transparent 100%);
}
.demo-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 22px; font-weight: 700;
    color: var(--text); margin: 0 0 5px;
    letter-spacing: -0.3px;
}
.demo-header h1 span { color: var(--accent); }
.demo-header .sub1 { font-size: 11px; color: var(--text-muted); margin: 0; letter-spacing: 0.6px; }
.demo-header .sub2 { font-size: 12px; color: #444; margin: 5px 0 0; font-style: italic; }

/* ── Persona panel ── */
.persona-section-title {
    font-size: 10px; font-weight: 700;
    color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px;
    text-align: center; padding: 6px 0 10px;
    border-bottom: 1px solid var(--border); margin-bottom: 10px;
}

/* Persona buttons */
button.persona-btn {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    padding: 10px 8px !important;
    cursor: pointer !important;
    transition: all 0.22s ease !important;
    text-align: center !important;
    min-height: 120px !important;
    line-height: 1.5 !important;
    white-space: pre-wrap !important;
    font-size: 12px !important;
}
button.persona-btn:hover {
    border-color: var(--accent) !important;
    background: var(--bg-hover) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 22px var(--accent-dim) !important;
}

/* ── Chat panel ── */
.chatbot { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
input[type="text"], textarea {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 8px !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-dim) !important;
    outline: none !important;
}

/* Send button */
.send-btn button {
    background: var(--accent) !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 13px !important;
    transition: all 0.2s !important;
}
.send-btn button:hover {
    background: #7b74ff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 14px var(--accent-glow) !important;
}

/* Back button */
.back-btn button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-size: 11px !important;
    transition: all 0.2s !important;
}
.back-btn button:hover { border-color: #555 !important; color: var(--text) !important; }

/* Bot info bar */
.bot-info-bar {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 14px;
    margin-bottom: 8px; font-size: 12px;
    color: var(--text-muted); display: flex; align-items: center; gap: 8px;
}

/* ── Tabs ── */
.tabs .tab-nav button {
    font-size: 11px !important; font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    background: transparent !important; border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--text-muted) !important; padding: 7px 11px !important;
    border-radius: 0 !important; transition: color 0.18s !important;
}
.tabs .tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}
.tabs .tab-nav button:hover:not(.selected) { color: var(--text) !important; }

/* ── Misc ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
"""

# ─── App ─────────────────────────────────────────────────────────────────────

with gr.Blocks(title="AI YOU · EMNLP Demo") as demo:

    # ── Persistent state ──────────────────────────────────────────────────────
    state_orch     = gr.State(None)   # OrchestratorAgent instance
    state_emotions = gr.State([])     # list of emotion dicts
    state_last     = gr.State({})     # last full response dict

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="demo-header">
      <h1>🧠 <span>AI YOU</span> · Multi-Agent Relationship Prediction System</h1>
      <p class="sub1">EMNLP 2025 System Demonstration &nbsp;·&nbsp; 12-Agent Pipeline &nbsp;·&nbsp;
        Bayesian Inference &nbsp;·&nbsp; Conformal Prediction &nbsp;·&nbsp; Three-Layer Memory</p>
      <p class="sub2">Infer Big-Five personality, emotion trajectory, and relationship stage from free-form conversation</p>
    </div>
    """)

    # ── Two-column layout ─────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── LEFT: Persona selector + Chat ────────────────────────────────────
        with gr.Column(scale=2, min_width=320):

            # Persona selection panel
            with gr.Column(visible=True) as persona_panel:
                mode_radio = gr.Radio(
                    choices=["🗣️ Social Chat", "🎓 Expert Consulting", "🪞 Self-Dialogue"],
                    value="🗣️ Social Chat",
                    label="Conversation Mode",
                    elem_classes=["mode-radio"],
                )
                gr.HTML('<div class="persona-section-title">Choose a Conversation Partner</div>')

                # Social personas
                persona_buttons: list[tuple[gr.Button, str]] = []
                with gr.Column(visible=True) as social_group:
                    for row_start in range(0, min(len(PERSONAS), 8), 4):
                        with gr.Row():
                            for p in PERSONAS[row_start : row_start + 4]:
                                btn = gr.Button(_persona_btn_label(p), elem_classes=["persona-btn"])
                                persona_buttons.append((btn, p["id"]))

                # Expert personas
                with gr.Column(visible=False) as expert_group:
                    with gr.Row():
                        for p in EXPERT_PERSONAS:
                            btn = gr.Button(_persona_btn_label(p), elem_classes=["persona-btn"])
                            persona_buttons.append((btn, p["id"]))

                # Self-dialogue personas
                with gr.Column(visible=False) as self_group:
                    with gr.Row():
                        for p in SELF_DIALOGUE_PERSONAS:
                            btn = gr.Button(_persona_btn_label(p), elem_classes=["persona-btn"])
                            persona_buttons.append((btn, p["id"]))

            # Chat panel (hidden until persona selected)
            with gr.Column(visible=False) as chat_panel:
                bot_info_bar = gr.HTML(
                    '<div class="bot-info-bar">No active conversation.</div>'
                )
                chatbot = gr.Chatbot(
                    height=440,
                    show_label=False,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type a message and press Enter…",
                        show_label=False, scale=5, container=False,
                    )
                    send_btn = gr.Button(
                        "Send", variant="primary", scale=1,
                        elem_classes=["send-btn"],
                    )
                back_btn = gr.Button(
                    "← Change Partner", size="sm",
                    elem_classes=["back-btn"],
                )

        # ── RIGHT: Dashboard ──────────────────────────────────────────────────
        with gr.Column(scale=3, min_width=480):
            with gr.Tabs():

                with gr.Tab("🧬 Personality"):
                    personality_plot = gr.Plot(
                        value=_initial_radar(), show_label=False
                    )
                    personality_info = gr.HTML(value=_initial_pers_info())

                with gr.Tab("💭 Emotion"):
                    emotion_plot = gr.Plot(
                        value=_initial_emo_fig(), show_label=False
                    )
                    emotion_info = gr.HTML(value=_initial_emo_info())

                with gr.Tab("❤️ Relationship"):
                    relationship_html_comp = gr.HTML(value=_initial_rel())

                with gr.Tab("🧠 Memory"):
                    memory_html_comp = gr.HTML(value=_initial_mem())

                with gr.Tab("⚙️ Pipeline"):
                    pipeline_plot = gr.Plot(
                        value=_initial_pipe_fig(), show_label=False
                    )
                    gr.HTML(
                        '<div style="font-size:10px;color:#333;padding:3px 6px;">'
                        "Token usage per agent phase (cumulative). "
                        "Purple = input · Red = output."
                        "</div>"
                    )

                with gr.Tab("📋 Raw Data"):
                    data_json = gr.JSON(label="Last Response", value={})

    # ─── Output groups ────────────────────────────────────────────────────────

    # 8 dashboard outputs (order must match _initial_dashboard)
    DASH_OUTS = [
        personality_plot, personality_info,
        emotion_plot, emotion_info,
        relationship_html_comp, memory_html_comp,
        pipeline_plot, data_json,
    ]

    # Full output list for persona-select / back (15 total)
    FULL_OUTS = [
        state_orch, state_emotions, state_last,
        persona_panel, chat_panel,
        chatbot, bot_info_bar,
    ] + DASH_OUTS

    # Output list for send (13 total)
    SEND_OUTS = [
        chatbot, msg_input,
        state_orch, state_emotions, state_last,
    ] + DASH_OUTS

    # ─── Handlers ─────────────────────────────────────────────────────────────

    async def on_persona_select(bot_id: str):
        """Create a new OrchestratorAgent and start conversation."""
        if not BACKEND:
            p = next((x for x in PERSONAS if x["id"] == bot_id), {"name": bot_id})
            greeting = f"[Demo] Hello! I'm {p.get('name', bot_id)}. Backend is offline."
            info = f'<div class="bot-info-bar"><strong>{p.get("name", bot_id)}</strong> · Demo mode</div>'
            return (
                None, [], {},
                gr.update(visible=False), gr.update(visible=True),
                [{"role": "assistant", "content": greeting}], info,
                *_initial_dashboard(),
            )

        try:
            pool = get_bot_pool()
            if pool is None:
                raise RuntimeError("Bot pool not initialized — check API keys in Space settings")
            uid  = str(uuid.uuid4())[:8]
            orch = OrchestratorAgent(uid, pool, bot_id)
            start = orch.start_new_conversation(bot_id)

            greeting    = start.get("greeting", "Hello! Nice to meet you.")
            bot_profile = start.get("bot_profile", {})
            compat      = start.get("compatibility_score", 0.0)

            name  = bot_profile.get("profile_id", bot_id)
            age   = bot_profile.get("age", "")
            loc   = bot_profile.get("location", "")
            style = bot_profile.get("communication_style", "")

            parts = [f"<strong>{name}</strong>"]
            if age:   parts.append(str(age))
            if loc:   parts.append(loc)
            if style: parts.append(style.title())
            compat_span = f'<span style="margin-left:auto;color:#6C63FF;font-weight:600;">Compat {compat:.0%}</span>'
            info = (
                f'<div class="bot-info-bar">'
                + " · ".join(parts)
                + compat_span
                + "</div>"
            )

            return (
                orch, [], {},
                gr.update(visible=False), gr.update(visible=True),
                [{"role": "assistant", "content": greeting}], info,
                *_initial_dashboard(),
            )

        except Exception as exc:
            return (
                None, [], {},
                gr.update(visible=False), gr.update(visible=True),
                [{"role": "assistant", "content": f"⚠️ Could not start conversation: {exc}"}],
                '<div class="bot-info-bar">Error — see console.</div>',
                *_initial_dashboard(),
            )

    async def on_send(
        message: str,
        chat_history: list,
        orch,
        emotions: list,
        last_resp: dict,
    ):
        """Process message and refresh all dashboard components."""
        # Empty input — return unchanged
        if not message.strip():
            no_change = tuple(gr.update() for _ in DASH_OUTS)
            return (chat_history, message, orch, emotions, last_resp) + no_change

        # No active orchestrator
        if orch is None:
            chat_history = list(chat_history) + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": "Please select a conversation partner first."},
            ]
            return (chat_history, "", orch, emotions, last_resp) + tuple(_initial_dashboard())

        chat_history = list(chat_history) + [{"role": "user", "content": message}]

        try:
            result = await orch.process_user_message(message)
        except Exception as exc:
            chat_history.append({"role": "assistant", "content": f"⚠️ Error: {exc}"})
            return (chat_history, "", orch, emotions, last_resp) + tuple(_initial_dashboard())

        # Bot reply
        bot_msg = result.get("bot_message", "…")
        chat_history.append({"role": "assistant", "content": bot_msg})

        # Update emotion history
        emo_data = result.get("emotion", {}).get("current_emotion", {})
        if emo_data.get("emotion"):
            emotions = emotions + [{
                "turn":       result.get("turn", len(emotions) + 1),
                "emotion":    emo_data["emotion"],
                "confidence": emo_data.get("confidence", 0.0),
                "intensity":  emo_data.get("intensity", 0.0),
            }]

        # Background: relationship prediction
        if getattr(orch, "_pending_relationship_turn", False):
            try:
                rel_result = await orch.run_relationship_prediction()
                if rel_result.get("relationship_prediction"):
                    result["relationship_prediction"] = rel_result["relationship_prediction"]
            except Exception:
                pass

        ctx = orch.ctx

        # Build all visualizations
        radar     = make_personality_radar(ctx.predicted_features, ctx.feature_confidences)
        pers_info = make_personality_info_html(ctx.predicted_features, ctx)
        emo_fig   = make_emotion_timeline(emotions)
        emo_info  = make_emotion_info_html(result, emotions)
        rel_html  = make_relationship_html(ctx)
        mem_html  = make_memory_html(result.get("memory_stats", {}))
        pipe_fig  = make_pipeline_chart(result.get("agent_costs", {}))

        return (
            chat_history, "",
            orch, emotions, result,
            radar, pers_info,
            emo_fig, emo_info,
            rel_html, mem_html,
            pipe_fig, result,
        )

    async def on_back():
        """Return to persona selection screen."""
        return (
            None, [], {},
            gr.update(visible=True), gr.update(visible=False),
            [], '<div class="bot-info-bar">No active conversation.</div>',
            *_initial_dashboard(),
        )

    # ─── Wire events ──────────────────────────────────────────────────────────

    # Persona buttons — each captures its bot_id in a closure
    for _btn, _bid in persona_buttons:
        async def _selector(_b=_bid):
            return await on_persona_select(_b)

        _btn.click(fn=_selector, inputs=[], outputs=FULL_OUTS)

    send_btn.click(
        fn=on_send,
        inputs=[msg_input, chatbot, state_orch, state_emotions, state_last],
        outputs=SEND_OUTS,
    )
    msg_input.submit(
        fn=on_send,
        inputs=[msg_input, chatbot, state_orch, state_emotions, state_last],
        outputs=SEND_OUTS,
    )
    back_btn.click(fn=on_back, inputs=[], outputs=FULL_OUTS)

    # Mode switching — toggle persona groups
    def switch_mode(mode):
        return (
            gr.update(visible=(mode == "🗣️ Social Chat")),
            gr.update(visible=(mode == "🎓 Expert Consulting")),
            gr.update(visible=(mode == "🪞 Self-Dialogue")),
        )

    mode_radio.change(
        fn=switch_mode,
        inputs=[mode_radio],
        outputs=[social_group, expert_group, self_group],
    )


# ─── Launch ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CSS)
