---
title: SoulMatch Agent
emoji: 💕
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# SoulMatch Agent v2.0

Multi-agent relationship prediction system with conformal uncertainty quantification.

🚀 **Live Demo**: https://huggingface.co/spaces/Quinnnnnne/SoulMatch-Agent

## Features

- 🤖 6 协同Agent（Orchestrator, Feature, Emotion, Scam, Persona, Question）
- 📊 42维用户特征推断（Big Five + MBTI + 依恋风格 + 爱语 + 信任轨迹）
- 🎯 保形预测（APS）不确定性量化
- 🧠 三层记忆管理（Working → Episodic → Semantic）
- 💬 实时WebSocket通信
- 🎨 精美UI设计（social-forecast风格）

## Quick Start

Visit the Space and start chatting with AI personas!

## Tech Stack

- Backend: FastAPI + Python 3.11
- Frontend: React + TypeScript + Vite
- LLM: GPT-5.2, Gemini Flash, DeepSeek
- Memory: ChromaDB
- Calibration: Conformal Prediction (APS)

## Local Development

```bash
# Backend
pip install -r requirements.txt
uvicorn src.api.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

## Configuration

Environment variables (set in HuggingFace Space settings):
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `DEEPSEEK_API_KEY`
- `ANTHROPIC_API_KEY`
- `QWEN_API_KEY`

## Paper

Based on research combining:
- Social Agents (ICLR 2026): Demographic diversity for wisdom of crowds
- Conformal Prediction: Uncertainty quantification with coverage guarantees

## License

MIT
