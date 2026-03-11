"""
SoulMatch Agent - Professional Demo UI
EMNLP 2025 System Demonstration
"""

import gradio as gr
import json
import asyncio
import uuid

# Backend
try:
    from src.api.session_manager import SessionManager
    from src.agents.persona_agent import PersonaAgentPool
    BACKEND = True
except:
    BACKEND = False

if BACKEND:
    sm = SessionManager()
    pool = PersonaAgentPool()
    pool.load_from_file("./data/processed/bot_personas.json")
    sm.set_bot_personas_pool(pool)

PERSONAS = [
    {"id": "bot_0", "name": "林婉清", "emoji": "👩‍💼", "age": 28, "job": "政府工作者"},
    {"id": "bot_1", "name": "陈思雨", "emoji": "📝", "age": 32, "job": "行政职员"},
    {"id": "bot_2", "name": "赵雅文", "emoji": "📚", "age": 45, "job": "大学教师"},
    {"id": "bot_3", "name": "刘程远", "emoji": "💻", "age": 24, "job": "程序员"},
    {"id": "bot_4", "name": "王建业", "emoji": "💼", "age": 38, "job": "企业管理"},
    {"id": "bot_5", "name": "张宇轩", "emoji": "💰", "age": 29, "job": "金融从业者"},
    {"id": "bot_6", "name": "李欣怡", "emoji": "🌸", "age": 52, "job": "自由职业"},
    {"id": "bot_7", "name": "周明轩", "emoji": "🎨", "age": 21, "job": "设计师"},
]

async def process(sid, msg, bot_id):
    if not BACKEND:
        return {"reply": "Backend unavailable", "turn": 0}

    orch = sm.get_session(sid)
    if not orch:
        sm.create_session(user_id=sid, bot_id=bot_id)
        orch = sm.get_session(sid)
        orch.start_new_conversation(bot_id=bot_id)

    result = await orch.process_user_message(msg)
    ctx = orch.ctx

    return {
        "reply": result.get("bot_message", ""),
        "turn": ctx.turn_count,
        "big5": {
            "O": round(ctx.predicted_features.get("big_five_openness", 0) * 100),
            "C": round(ctx.predicted_features.get("big_five_conscientiousness", 0) * 100),
            "E": round(ctx.predicted_features.get("big_five_extraversion", 0) * 100),
            "A": round(ctx.predicted_features.get("big_five_agreeableness", 0) * 100),
            "N": round(ctx.predicted_features.get("big_five_neuroticism", 0) * 100),
        },
        "mbti": ctx.predicted_features.get("mbti", "?"),
    }

def chat(msg, hist, sid, bid):
    if not msg.strip():
        return hist, ""

    result = asyncio.get_event_loop().run_until_complete(process(sid, msg, bid))

    hist.append({"role": "user", "content": msg})
    hist.append({"role": "assistant", "content": result["reply"]})

    return hist, ""

def start(idx):
    p = PERSONAS[idx]
    sid = str(uuid.uuid4())
    greeting = f"你好！我是{p['name']},{p['age']}岁,{p['job']}。很高兴认识你！"

    return (
        [{"role": "assistant", "content": greeting}],
        sid,
        p["id"],
        gr.update(visible=False),
        gr.update(visible=True),
    )

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Noto+Sans+SC:wght@300;400;500;700&display=swap');

:root {
    --bg-deep: #0a0b0f;
    --bg-card: #12131a;
    --bg-hover: #1a1b25;
    --accent: #00e5a0;
    --accent-dim: #00e5a033;
    --accent-glow: #00e5a066;
    --text-primary: #e8e8ec;
    --text-secondary: #7a7b8e;
    --text-dim: #44455a;
    --border: #1e1f2e;
    --gradient-1: linear-gradient(135deg, #00e5a0 0%, #00b4d8 100%);
    --gradient-2: linear-gradient(135deg, #7b2ff7 0%, #00e5a0 100%);
}

body, .gradio-container {
    background: var(--bg-deep) !important;
    color: var(--text-primary) !important;
    font-family: 'Noto Sans SC', sans-serif !important;
}

button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text-primary) !important;
    transition: all 0.3s !important;
}

button:hover {
    border-color: var(--accent) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px var(--accent-dim) !important;
}

.chatbot {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
}

input, textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
}

input:focus, textarea:focus {
    border-color: var(--accent) !important;
}
"""

with gr.Blocks(title="SoulMatch Agent") as demo:

    gr.HTML('''
    <div style="text-align: center; margin: 40px 0 24px;">
        <h1 style="font-family: 'Space Mono', monospace; font-size: 3em; font-weight: 700; background: linear-gradient(135deg, #7b2ff7 0%, #00e5a0 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px;">SoulMatch</h1>
        <p style="color: #7a7b8e; font-size: 1.1em;">通过30句对话推断你的完整性格画像</p>
    </div>
    ''')

    sid_state = gr.State(str(uuid.uuid4()))
    bid_state = gr.State("bot_0")

    with gr.Row():
        with gr.Column(scale=1, visible=True) as select_panel:
            gr.HTML('<p style="text-align: center; color: #7a7b8e; font-size: 14px; margin-bottom: 16px;">选择一个人开始聊天</p>')

            for i, p in enumerate(PERSONAS):
                btn = gr.Button(
                    f"{p['emoji']} {p['name']} · {p['age']}岁 · {p['job']}",
                    size="lg"
                )
                btn.click(
                    fn=lambda idx=i: start(idx),
                    inputs=[],
                    outputs=[gr.Chatbot(elem_id="chatbot"), sid_state, bid_state, select_panel, gr.Column(elem_id="chat_panel")]
                )

        with gr.Column(scale=3, visible=False, elem_id="chat_panel") as chat_panel:
            chatbot = gr.Chatbot(
                height=600,
                label="对话"
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="输入消息...",
                    show_label=False,
                    scale=5,
                    container=False
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)

    send_btn.click(
        fn=chat,
        inputs=[msg_box, chatbot, sid_state, bid_state],
        outputs=[chatbot, msg_box]
    )

    msg_box.submit(
        fn=chat,
        inputs=[msg_box, chatbot, sid_state, bid_state],
        outputs=[chatbot, msg_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CSS)
