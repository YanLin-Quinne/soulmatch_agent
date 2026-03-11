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
    {"id": "bot_0", "name": "Alex", "age": 28, "job": "Software Engineer"},
    {"id": "bot_1", "name": "Maya", "age": 32, "job": "Journalist"},
    {"id": "bot_2", "name": "James", "age": 45, "job": "Chef"},
    {"id": "bot_3", "name": "Sofia", "age": 24, "job": "PhD Student"},
    {"id": "bot_4", "name": "Leo", "age": 38, "job": "Entrepreneur"},
    {"id": "bot_5", "name": "Priya", "age": 29, "job": "UX Designer"},
    {"id": "bot_6", "name": "Omar", "age": 52, "job": "History Teacher"},
    {"id": "bot_7", "name": "Zoe", "age": 21, "job": "Musician"},
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
    greeting = f"Hi! I'm {p['name']}, {p['age']}, {p['job']}. Nice to meet you!"

    return (
        [{"role": "assistant", "content": greeting}],
        sid,
        p["id"],
        gr.update(visible=False),
        gr.update(visible=True),
    )

CSS = """
.container { max-width: 1400px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 32px; text-align: center; border-radius: 12px; margin-bottom: 24px; }
.title { font-size: 2.5em; font-weight: 700; margin-bottom: 8px; }
.subtitle { font-size: 1.1em; opacity: 0.95; }
.persona-card { border: 2px solid #e5e7eb; border-radius: 12px; padding: 20px; cursor: pointer; transition: all 0.3s; background: white; }
.persona-card:hover { border-color: #667eea; box-shadow: 0 4px 12px rgba(102,126,234,0.15); transform: translateY(-2px); }
"""

with gr.Blocks(title="SoulMatch Agent") as demo:

    gr.HTML('<div class="header"><div class="title">SoulMatch Agent</div><div class="subtitle">Progressive Personality Profiling via Conversational Inference</div></div>')

    sid_state = gr.State(str(uuid.uuid4()))
    bid_state = gr.State("bot_0")

    with gr.Row():
        with gr.Column(scale=1, visible=True) as select_panel:
            gr.Markdown("### Select a Persona to Chat With")
            persona_radio = gr.Radio(
                choices=[f"{p['name']}, {p['age']} — {p['job']}" for p in PERSONAS],
                label="",
                value=f"{PERSONAS[0]['name']}, {PERSONAS[0]['age']} — {PERSONAS[0]['job']}"
            )
            start_btn = gr.Button("Start Conversation", variant="primary", size="lg")
            gr.Markdown("---")
            gr.Markdown("**About**: This system infers your personality traits through natural conversation. Chat for 30 turns to see your complete profile.")

        with gr.Column(scale=3, visible=False) as chat_panel:
            chatbot = gr.Chatbot(
                height=600,
                label="Conversation"
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    scale=5,
                    container=False
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

    start_btn.click(
        fn=lambda x: start([i for i, p in enumerate(PERSONAS) if p['name'] in x][0]),
        inputs=[persona_radio],
        outputs=[chatbot, sid_state, bid_state, select_panel, chat_panel]
    )

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
