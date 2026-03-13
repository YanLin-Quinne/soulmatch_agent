"""
SoulMatch Agent - Flask + Custom UI
前后端同源,完全自定义 HTML/CSS
"""
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import asyncio
import uuid
import os
import json

from src.api.session_manager import SessionManager
from src.agents.persona_agent import PersonaAgentPool

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# 初始化后端
sm = SessionManager()
pool = PersonaAgentPool()
pool.load_from_file("./data/processed/bot_personas.json")
sm.set_bot_personas_pool(pool)

# Load personas from bot_personas.json
with open("./data/processed/bot_personas.json") as f:
    bot_data = json.load(f)
    PERSONAS = []
    for i, bot in enumerate(bot_data[:8]):  # Use first 8 bots
        profile = bot["original_profile"]
        # Extract name from system prompt
        name = bot["system_prompt"].split(',')[0].replace('You are ', '').strip()
        # Use randomuser.me API for real photos
        gender = 'women' if profile["sex"] == "f" else 'men'
        avatar_url = f"https://randomuser.me/api/portraits/{gender}/{i}.jpg"
        PERSONAS.append({
            "id": bot["profile_id"],
            "name": name,
            "avatar": avatar_url,
            "age": profile["age"],
            "job": profile["job"],
            "location": profile.get("location", ""),
        })

@app.route('/')
def index():
    return render_template('index.html', personas=PERSONAS)

@app.route('/api/start', methods=['POST'])
def start_chat():
    data = request.json
    bot_id = data.get('bot_id')

    sid = str(uuid.uuid4())
    session['sid'] = sid
    session['bot_id'] = bot_id

    sm.create_session(user_id=sid, bot_id=bot_id)
    orch = sm.get_session(sid)
    orch.start_new_conversation(bot_id=bot_id)

    # Get bot's system prompt for greeting
    bot_info = next((b for b in bot_data if b['profile_id'] == bot_id), bot_data[0])
    # Extract name from system prompt (e.g., "You are Lin Xiaoyu...")
    prompt = bot_info['system_prompt']
    name = prompt.split(',')[0].replace('You are ', '').strip()
    profile = bot_info['original_profile']
    greeting = f"Hi! I'm {name}, {profile['age']}, {profile['job']} from {profile['location']}. Nice to meet you!"

    return jsonify({"greeting": greeting, "sid": sid})

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        msg = data.get('message')
        sid = data.get('sid') or session.get('sid')

        if not sid:
            return jsonify({"error": "No session ID"}), 400

        orch = sm.get_session(sid)
        if not orch:
            return jsonify({"error": "Session not found"}), 404

        # Use nest_asyncio to handle nested event loops
        import nest_asyncio
        nest_asyncio.apply()

        result = asyncio.run(orch.process_user_message(msg))
        ctx = orch.ctx

        return jsonify({
            "reply": result.get("bot_message", ""),
            "turn": ctx.turn_count,
            "features": {
                "big5": {
                    "O": round(ctx.predicted_features.get("big_five_openness", 0) * 100),
                    "C": round(ctx.predicted_features.get("big_five_conscientiousness", 0) * 100),
                    "E": round(ctx.predicted_features.get("big_five_extraversion", 0) * 100),
                    "A": round(ctx.predicted_features.get("big_five_agreeableness", 0) * 100),
                    "N": round(ctx.predicted_features.get("big_five_neuroticism", 0) * 100),
                },
                "mbti": ctx.predicted_features.get("mbti", "?"),
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
