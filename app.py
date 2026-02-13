"""
SoulMatch Agent - Gradio Web UI
For local use and HuggingFace Spaces deployment.
"""

import json
import random
import os
import sys
from pathlib import Path

import gradio as gr

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.schema import PersonaProfile
from src.agents.persona_agent import PersonaAgent, PersonaAgentPool
from src.agents.emotion_agent import EmotionAgent
from src.agents.scam_detection_agent import ScamDetectionAgent
from src.agents.feature_prediction_agent import FeaturePredictionAgent
from src.memory.memory_manager import MemoryManager
from src.config import settings


# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────

bot_pool: PersonaAgentPool = None
personas_data: list[dict] = []


# ─────────────────────────────────────────────
# Persona Display Mapping (Chinese UI)
# ─────────────────────────────────────────────

PERSONA_DISPLAY = [
    {
        "chinese_name": "林婉清",
        "emoji": "👩‍💼",
        "job_display": "政府工作者",
        "location_short": "旧金山",
        "status": "来聊天",
        "status_color": "#FFA500",  # orange
    },
    {
        "chinese_name": "陈思雨",
        "emoji": "📝",
        "job_display": "行政职员",
        "location_short": "旧金山",
        "status": "在线中",
        "status_color": "#FFD700",  # gold
    },
    {
        "chinese_name": "赵雅文",
        "emoji": "📚",
        "job_display": "大学教师",
        "location_short": "奥克兰",
        "status": "闲逛中",
        "status_color": "#FF8C00",  # dark orange
    },
    {
        "chinese_name": "刘程远",
        "emoji": "💻",
        "job_display": "程序员",
        "location_short": "旧金山",
        "status": "随缘聊",
        "status_color": "#FFA500",  # orange
    },
    {
        "chinese_name": "王建业",
        "emoji": "💼",
        "job_display": "企业管理",
        "location_short": "卡斯特罗谷",
        "status": "来聊天",
        "status_color": "#FFD700",  # gold
    },
    {
        "chinese_name": "张宇轩",
        "emoji": "💰",
        "job_display": "金融从业者",
        "location_short": "奥克兰",
        "status": "在线中",
        "status_color": "#FF8C00",  # dark orange
    },
    {
        "chinese_name": "李欣怡",
        "emoji": "🌸",
        "job_display": "自由职业",
        "location_short": "海沃德",
        "status": "随缘聊",
        "status_color": "#FFA500",  # orange
    },
    {
        "chinese_name": "周逸风",
        "emoji": "😎",
        "job_display": "市场营销",
        "location_short": "旧金山",
        "status": "闲逛中",
        "status_color": "#FFD700",  # gold
    },
]


def load_bot_pool():
    """Load bot personas at startup"""
    global bot_pool, personas_data

    personas_path = Path("data/processed/bot_personas.json")
    if not personas_path.exists():
        raise FileNotFoundError(
            f"Bot personas not found at {personas_path}. "
            "Run: python scripts/preprocess_data.py first."
        )

    with open(personas_path, "r") as f:
        personas_data = json.load(f)

    # Determine which API to use
    use_claude = bool(settings.anthropic_api_key)
    bot_pool = PersonaAgentPool(use_claude=use_claude, temperature=0.8)

    personas = [PersonaProfile(**p) for p in personas_data]
    bot_pool.load_personas(personas)

    return bot_pool


# ─────────────────────────────────────────────
# Session state per user (dict keyed by session hash)
# ─────────────────────────────────────────────

class ChatSession:
    """Per-user chat session state"""

    def __init__(self, bot_agent: PersonaAgent, use_claude: bool):
        self.bot_agent = bot_agent
        self.use_claude = use_claude
        self.turn = 0
        self.emotion_agent = EmotionAgent(use_claude=use_claude)
        self.scam_agent = ScamDetectionAgent(use_semantic=False)
        self.feature_agent = FeaturePredictionAgent("gradio_user", use_claude=use_claude)
        self.conversation_history: list[dict] = []

    def chat(self, user_message: str) -> tuple[str, str, str]:
        """
        Process one turn of conversation.
        Returns: (bot_reply, emotion_info, warning_info)
        """
        self.turn += 1
        self.conversation_history.append({"speaker": "user", "message": user_message})

        # 1. Bot response
        bot_reply = self.bot_agent.generate_response(user_message)
        self.conversation_history.append({"speaker": "bot", "message": bot_reply})

        # 2. Emotion detection
        emotion_info = ""
        try:
            emo = self.emotion_agent.analyze_message(user_message)
            if emo and "current_emotion" in emo:
                ce = emo["current_emotion"]
                emotion_info = f"{ce.get('emotion', 'neutral')} (confidence: {ce.get('confidence', 0):.0%})"
        except Exception:
            emotion_info = "neutral"

        # 3. Scam detection
        warning_info = ""
        try:
            scam = self.scam_agent.analyze_message(user_message)
            if scam["warning_level"] not in ("safe",):
                patterns = ", ".join(p.name for p in scam.get("detected_patterns", []))
                warning_info = f"⚠️ {scam['warning_level'].upper()}: {patterns} (risk: {scam['risk_score']:.0%})"
        except Exception:
            pass

        # 4. Feature prediction (every 3 turns, first 30 turns)
        if self.turn <= 30 and self.turn % 3 == 0:
            try:
                self.feature_agent.predict_from_conversation(self.conversation_history)
            except Exception:
                pass

        return bot_reply, emotion_info, warning_info


sessions: dict[str, ChatSession] = {}


# ─────────────────────────────────────────────
# Gradio interface functions
# ─────────────────────────────────────────────

def get_bot_choices() -> list[str]:
    """Get list of bot display names (legacy, kept for compatibility)"""
    choices = []
    for i, p in enumerate(personas_data):
        prof = p["original_profile"]
        feat = p["features"]
        sex = "F" if prof.get("sex") == "f" else "M"
        age = prof.get("age", "?")
        style = feat.get("communication_style", "casual")
        summary = feat.get("personality_summary", "")[:60]
        choices.append(f"Bot {i} | {sex}/{age} | {style} | {summary}")
    return choices


def build_card_grid_html(filter_age: str = "all") -> str:
    """
    Build HTML for card-based persona selection interface with dark theme
    
    Args:
        filter_age: Age filter - "all", "20s", "30s", "40s", "50+"
    
    Returns:
        Complete HTML string with inline CSS and JavaScript
    """
    
    # Filter personas by age
    filtered_personas = []
    for i, p in enumerate(personas_data):
        age = p["original_profile"].get("age", 0)
        
        if filter_age == "all":
            filtered_personas.append(i)
        elif filter_age == "20s" and 20 <= age < 30:
            filtered_personas.append(i)
        elif filter_age == "30s" and 30 <= age < 40:
            filtered_personas.append(i)
        elif filter_age == "40s" and 40 <= age < 50:
            filtered_personas.append(i)
        elif filter_age == "50+" and age >= 50:
            filtered_personas.append(i)
    
    # Build card HTML for each persona
    cards_html = ""
    for bot_idx in filtered_personas:
        p = personas_data[bot_idx]
        prof = p["original_profile"]
        display = PERSONA_DISPLAY[bot_idx]
        
        age = prof.get("age", "?")
        
        # Determine age group for tag
        if age < 30:
            age_group = "20s"
        elif age < 40:
            age_group = "30s"
        elif age < 50:
            age_group = "40s"
        else:
            age_group = "50+"
        
        cards_html += f"""
        <div class="persona-card" onclick="selectBot({bot_idx})">
            <div class="card-emoji">{display['emoji']}</div>
            <div class="card-name">{display['chinese_name']}</div>
            <div class="card-job">{display['job_display']} · {display['location_short']}</div>
            <div class="card-tags">
                <span class="tag tag-age">{age_group}</span>
                <span class="tag tag-location">{display['location_short']}</span>
                <span class="tag tag-status" style="background-color: {display['status_color']};">{display['status']}</span>
            </div>
        </div>
        """
    
    # Complete HTML with CSS and JavaScript
    html = f"""
    <style>
        .card-selection-container {{
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
            border-radius: 16px;
            padding: 40px 30px;
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        
        .title {{
            font-size: 42px;
            font-weight: 800;
            background: linear-gradient(90deg, #00d4aa 0%, #00ff88 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 12px;
            letter-spacing: -0.5px;
        }}
        
        .subtitle {{
            text-align: center;
            color: #a0a0b0;
            font-size: 15px;
            margin-bottom: 32px;
            line-height: 1.6;
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
        }}
        
        .age-filters {{
            display: flex;
            justify-content: center;
            gap: 12px;
            margin-bottom: 32px;
            flex-wrap: wrap;
        }}
        
        .filter-btn {{
            background: #2a2a3e;
            color: #a0a0b0;
            border: 2px solid transparent;
            padding: 10px 24px;
            border-radius: 24px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        
        .filter-btn:hover {{
            background: #3a3a4e;
            color: #fff;
        }}
        
        .filter-btn.active {{
            background: linear-gradient(90deg, #00d4aa 0%, #00ff88 100%);
            color: #0f0f23;
            border-color: #00ff88;
        }}
        
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
            gap: 20px;
            margin-bottom: 32px;
        }}
        
        @media (max-width: 768px) {{
            .cards-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        
        .persona-card {{
            background: linear-gradient(135deg, #1a1a3e 0%, #252547 100%);
            border-radius: 16px;
            padding: 24px 20px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: 2px solid transparent;
            position: relative;
            overflow: hidden;
        }}
        
        .persona-card::before {{
            content: '';
            position: absolute;
            inset: 0;
            border-radius: 16px;
            padding: 2px;
            background: linear-gradient(135deg, #00d4aa, #00ff88);
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        
        .persona-card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 12px 40px rgba(0, 212, 170, 0.3);
        }}
        
        .persona-card:hover::before {{
            opacity: 1;
        }}
        
        .card-emoji {{
            font-size: 48px;
            text-align: center;
            margin-bottom: 12px;
        }}
        
        .card-name {{
            font-size: 20px;
            font-weight: 700;
            color: #fff;
            text-align: center;
            margin-bottom: 8px;
        }}
        
        .card-job {{
            font-size: 13px;
            color: #8888a8;
            text-align: center;
            margin-bottom: 16px;
        }}
        
        .card-tags {{
            display: flex;
            gap: 6px;
            justify-content: center;
            flex-wrap: wrap;
        }}
        
        .tag {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}
        
        .tag-age {{
            background: #3b82f6;
            color: #fff;
        }}
        
        .tag-location {{
            background: #10b981;
            color: #fff;
        }}
        
        .tag-status {{
            color: #fff;
        }}
        
        .footer {{
            text-align: center;
            color: #8888a8;
            font-size: 14px;
            margin-top: 24px;
        }}
    </style>
    
    <div class="card-selection-container">
        <div class="title">谁是真人?</div>
        <div class="subtitle">
            选择一个人开始聊天。30句对话后系统将推断对方的性格、心理、社会特征。注意——部分角色是AI伪装的。
        </div>
        
        <div class="age-filters">
            <button class="filter-btn {'active' if filter_age == 'all' else ''}" onclick="filterAge('all')">全部</button>
            <button class="filter-btn {'active' if filter_age == '20s' else ''}" onclick="filterAge('20s')">20s</button>
            <button class="filter-btn {'active' if filter_age == '30s' else ''}" onclick="filterAge('30s')">30s</button>
            <button class="filter-btn {'active' if filter_age == '40s' else ''}" onclick="filterAge('40s')">40s</button>
            <button class="filter-btn {'active' if filter_age == '50+' else ''}" onclick="filterAge('50+')">50+</button>
        </div>
        
        <div class="cards-grid">
            {cards_html}
        </div>
        
        <div class="footer">
            🎭 8人中有部分AI角色，你能分辨吗？
        </div>
    </div>
    
    <script>
        function selectBot(botIdx) {{
            // Set selected bot ID in hidden textbox
            const textbox = document.querySelector('#selected-bot-id textarea');
            if (textbox) {{
                textbox.value = botIdx;
                textbox.dispatchEvent(new Event('input', {{ bubbles: true }}));
            }}
            
            // Trigger hidden select button
            const selectBtn = document.querySelector('#select-btn');
            if (selectBtn) {{
                selectBtn.click();
            }}
        }}
        
        function filterAge(ageGroup) {{
            // Set age filter in hidden textbox
            const filterBox = document.querySelector('#age-filter textarea');
            if (filterBox) {{
                filterBox.value = ageGroup;
                filterBox.dispatchEvent(new Event('input', {{ bubbles: true }}));
            }}
            
            // Trigger hidden filter button
            const filterBtn = document.querySelector('#filter-btn');
            if (filterBtn) {{
                filterBtn.click();
            }}
        }}
    </script>
    """
    
    return html


def start_chat(bot_idx: int):
    """
    Start a new conversation with selected bot
    
    Args:
        bot_idx: Bot index (0-7)
    
    Returns:
        Tuple of (chat_history, profile_info, emotion_box, warning_box, turn_counter,
                  view1_update, view2_update)
    """
    bot_id = f"bot_{bot_idx}"

    agent = bot_pool.get_agent(bot_id)
    if not agent:
        return [], "Bot not found", "", "", "Turn 0/30", gr.update(visible=True), gr.update(visible=False)

    use_claude = bool(settings.anthropic_api_key)
    session = ChatSession(agent, use_claude)
    sessions["current"] = session

    # Generate greeting
    greeting = agent.generate_greeting()
    session.conversation_history.append({"speaker": "bot", "message": greeting})

    # Bot profile info with Chinese display data
    p = personas_data[bot_idx]
    prof = p["original_profile"]
    feat = p["features"]
    display = PERSONA_DISPLAY[bot_idx]
    
    profile_info = f"""
<div style="background: linear-gradient(135deg, #1a1a3e 0%, #252547 100%); padding: 24px; border-radius: 16px; color: #fff;">
    <div style="text-align: center; font-size: 48px; margin-bottom: 16px;">{display['emoji']}</div>
    <div style="text-align: center; font-size: 24px; font-weight: 700; margin-bottom: 8px;">{display['chinese_name']}</div>
    <div style="text-align: center; color: #8888a8; margin-bottom: 16px;">{display['job_display']} · {display['location_short']}</div>
    <div style="border-top: 1px solid #3a3a5e; padding-top: 16px; margin-top: 16px;">
        <div style="margin-bottom: 8px;"><strong>年龄:</strong> {prof.get('age', '?')}</div>
        <div style="margin-bottom: 8px;"><strong>风格:</strong> {feat.get('communication_style', '?')}</div>
        <div style="margin-bottom: 8px;"><strong>价值观:</strong> {', '.join(feat.get('core_values', []))}</div>
        <div><strong>目标:</strong> {feat.get('relationship_goals', '?')}</div>
    </div>
</div>
"""

    chat_history = [{"role": "assistant", "content": greeting}]
    
    # Hide view 1 (selection), show view 2 (chat)
    return (
        chat_history,
        profile_info,
        "",  # emotion_box
        "",  # warning_box
        "Turn 0/30",  # turn_counter
        gr.update(visible=False),  # view1
        gr.update(visible=True),   # view2
    )


def respond(user_message: str, chat_history: list):
    """Handle user message and return updated state"""
    if not user_message.strip():
        session = sessions.get("current")
        turn_text = f"Turn {session.turn}/30" if session else "Turn 0/30"
        return chat_history, "", "", turn_text

    session = sessions.get("current")
    if not session:
        chat_history.append({"role": "assistant", "content": "请先选择一个角色开始聊天。"})
        return chat_history, "", "", "Turn 0/30"

    bot_reply, emotion_info, warning_info = session.chat(user_message)

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": bot_reply})

    turn_text = f"Turn {session.turn}/30"
    return chat_history, emotion_info, warning_info, turn_text


def get_feature_summary():
    """Get current predicted features"""
    session = sessions.get("current")
    if not session:
        return "No active session"

    summary = session.feature_agent.get_feature_summary()
    if not summary:
        return "Not enough data yet (chat more!)"

    lines = []
    for k, v in summary.items():
        if v and v != "unknown":
            lines.append(f"**{k}**: {v}")
    return "\n".join(lines) if lines else "Still learning about you... keep chatting!"


# ─────────────────────────────────────────────
# Build Gradio UI
# ─────────────────────────────────────────────

def build_app():
    """Build the two-view Gradio app: selection page + chat page"""
    load_bot_pool()

    with gr.Blocks(title="SoulMatch - 谁是真人?") as demo:
        # Hidden components for JavaScript bridge
        selected_bot_id = gr.Textbox(visible=False, elem_id="selected-bot-id")
        age_filter_state = gr.State("all")
        age_filter_box = gr.Textbox(visible=False, elem_id="age-filter")
        
        # ═══════════════════════════════════════════════════════════
        # VIEW 1: Card Selection Page
        # ═══════════════════════════════════════════════════════════
        with gr.Column(visible=True, elem_id="view-selection") as view1:
            card_grid = gr.HTML(build_card_grid_html("all"))
            
            # Hidden button to trigger selection
            select_btn = gr.Button("Select", visible=False, elem_id="select-btn")
            
            # Hidden button to trigger filter
            filter_btn = gr.Button("Filter", visible=False, elem_id="filter-btn")
        
        # ═══════════════════════════════════════════════════════════
        # VIEW 2: Chat Interface
        # ═══════════════════════════════════════════════════════════
        with gr.Column(visible=False, elem_id="view-chat") as view2:
            # Back button
            with gr.Row():
                back_btn = gr.Button("← 返回选择", variant="secondary", size="sm")
                turn_counter = gr.Markdown("Turn 0/30", elem_classes=["turn-counter"])
            
            with gr.Row():
                # Left sidebar: persona info
                with gr.Column(scale=1):
                    profile_html = gr.HTML("<div style='color: #fff;'>选择一个角色开始聊天</div>")
                    
                    gr.Markdown("---")
                    
                    # Feature prediction section
                    gr.Markdown("### 🔮 预测特征")
                    feature_btn = gr.Button("显示预测结果", variant="secondary", size="sm")
                    feature_md = gr.Markdown("*聊天30轮后可查看预测*")
                
                # Center: chat area
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="对话",
                        height=500,
                        show_label=True,
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="",
                            placeholder="输入消息...",
                            scale=4,
                            show_label=False,
                        )
                        send_btn = gr.Button("发送", variant="primary", scale=1)
                
                # Right sidebar: emotion & scam detection
                with gr.Column(scale=1):
                    gr.Markdown("### 😊 情感检测")
                    emotion_box = gr.Textbox(
                        label="",
                        value="neutral",
                        interactive=False,
                        show_label=False,
                    )
                    
                    gr.Markdown("### ⚠️ 诈骗警告")
                    warning_box = gr.Textbox(
                        label="",
                        value="安全",
                        interactive=False,
                        show_label=False,
                    )
        
        # ═══════════════════════════════════════════════════════════
        # Event Handlers
        # ═══════════════════════════════════════════════════════════
        
        # Age filter callback
        def apply_age_filter(age_group):
            return build_card_grid_html(age_group), age_group
        
        filter_btn.click(
            fn=apply_age_filter,
            inputs=[age_filter_box],
            outputs=[card_grid, age_filter_state],
        )
        
        # Bot selection callback
        def handle_bot_selection(bot_id_str):
            try:
                bot_idx = int(bot_id_str)
                return start_chat(bot_idx)
            except (ValueError, IndexError):
                return [], "错误：无效的角色ID", "", "", "Turn 0/30", gr.update(visible=True), gr.update(visible=False)
        
        select_btn.click(
            fn=handle_bot_selection,
            inputs=[selected_bot_id],
            outputs=[chatbot, profile_html, emotion_box, warning_box, turn_counter, view1, view2],
        )
        
        # Back button: return to selection
        def go_back():
            # Clear current session
            if "current" in sessions:
                del sessions["current"]
            return (
                [],  # clear chatbot
                gr.update(visible=True),   # show view1
                gr.update(visible=False),  # hide view2
                build_card_grid_html("all"),  # reset card grid
            )
        
        back_btn.click(
            fn=go_back,
            outputs=[chatbot, view1, view2, card_grid],
        )
        
        # Chat send button
        send_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, emotion_box, warning_box, turn_counter],
        ).then(lambda: "", outputs=msg_input)
        
        # Chat input submit
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, emotion_box, warning_box, turn_counter],
        ).then(lambda: "", outputs=msg_input)
        
        # Feature prediction button
        feature_btn.click(
            fn=get_feature_summary,
            outputs=feature_md,
        )

    return demo


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Custom dark theme CSS for Gradio components (Gradio 6.x)
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%) !important;
    }
    .dark {
        background: #0f0f23;
    }
    """
    
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Base(),
        css=custom_css,
    )
