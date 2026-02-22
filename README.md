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

**Live Demo**: https://huggingface.co/spaces/Quinnnnnne/SoulMatch-Agent

## Features

- 15 collaborative agents (Orchestrator, Feature, Emotion, Scam, Persona, Relationship, Memory, etc.)
- 42-dimensional user feature inference (Big Five, MBTI, attachment style, love language, trust trajectory)
- Conformal prediction (APS) with coverage guarantees for uncertainty quantification
- Three-layer memory management (Working, Episodic, Semantic) with anti-hallucination mechanisms
- Relationship state prediction with conformal advancement assessment
- Real-time WebSocket communication
- Multi-LLM routing with automatic fallback (GPT-5.2, Gemini 3.1 Pro, Claude Opus 4.6, Qwen 3.5 Plus, DeepSeek V3.2)

## Research Contributions

Based on research combining:
- **Social Agents** (ICLR 2026): Multi-agent "wisdom of crowds" for behavioral prediction
- **Conformal Prediction for LLM-as-Judge** (EMNLP 2025): Uncertainty quantification with coverage guarantees
- **Memory-augmented dialogue**: Structured memory compression to combat hallucination in long-context conversations

Core research questions:
1. How does structured memory reduce feature prediction hallucination in long dialogues?
2. Can conformal prediction provide valid uncertainty quantification for relationship state advancement?
3. Does multi-agent collaboration outperform single-model prediction for relationship states?

## Tech Stack

- **Backend**: FastAPI + Python 3.11 + WebSocket
- **Frontend**: React + TypeScript + Vite
- **LLM**: GPT-5.2, Gemini 3.1 Pro, Claude Opus 4.6, Qwen 3.5 Plus, DeepSeek V3.2
- **Memory**: ChromaDB (vector store) + Three-layer memory architecture
- **Calibration**: Conformal Prediction (Adaptive Prediction Sets)
- **Data**: OkCupid Profiles (59,946 profiles, 31 dimensions)

## Quick Start

```bash
# Clone and install
git clone https://github.com/YanLin-Quinne/soulmatch_agent.git
cd soulmatch_agent
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys

# Start backend
uvicorn src.api.main:app --reload

# Start frontend (in another terminal)
cd frontend
npm install
npm run dev
```

## Configuration

Environment variables (set in `.env` or HuggingFace Space settings):
- `OPENAI_API_KEY` - OpenAI GPT-5.2
- `GEMINI_API_KEY` - Google Gemini 3.1 Pro / 2.5 Flash
- `ANTHROPIC_API_KEY` - Anthropic Claude Opus 4.6
- `QWEN_API_KEY` - Alibaba Qwen 3.5 Plus
- `DEEPSEEK_API_KEY` - DeepSeek V3.2 Reasoner

## Architecture

```
User Message
    |
    v
OrchestratorAgent (8-step pipeline)
    |
    +--[parallel]--> EmotionAgent + ScamDetectionAgent + MemoryManager.retrieve()
    |
    +--[sequential]--> FeaturePredictionAgent (42-dim, every 3 turns)
    |
    +--[sequential]--> FeatureTransitionPredictor (t -> t+1)
    |
    +--[sequential]--> RelationshipPredictionAgent (every 5 turns, conformal prediction)
    |
    +--[sequential]--> MilestoneEvaluator (turn 10/30)
    |
    +--[sequential]--> QuestionStrategyAgent + DiscussionEngine
    |
    +--[sequential]--> PersonaAgent (bot response generation)
    |
    +--[sequential]--> MemoryManager.execute() (every 5 turns)
```

## License

MIT
