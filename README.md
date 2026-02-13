# SoulMatch

AI-powered dating prediction agent. Chat with 8 bot personas — after 30 exchanges the system predicts personality traits, emotional patterns, and social tendencies.

## Architecture

```
Frontend (React + Vite)
    │  WebSocket
    ▼
FastAPI Backend ─── SessionManager
    │
    ▼
OrchestratorAgent (pipeline coordinator)
    │
    ├─ Phase 1 [parallel]
    │   ├── EmotionAgent        → detect emotion, intensity, reply strategy
    │   ├── ScamDetectionAgent  → rule + semantic scam detection
    │   └── MemoryManager       → retrieve relevant memories from ChromaDB
    │
    ├─ Phase 2 [sequential]
    │   └── FeaturePredictionAgent → CoT reasoning → 22-dim features (confidence convergence)
    │
    ├─ Phase 3 [sequential]
    │   └── QuestionStrategyAgent → probe low-confidence features naturally
    │
    ├─ Phase 4 [sequential]
    │   └── PersonaAgent → tool calling → generate bot response (full context injection)
    │
    └─ Phase 5 [sequential]
        └── MemoryManager → store new memories

MCP Server (stdio)
    └── Exposes: analyze_emotion, predict_features, check_scam, suggest_topics, get_usage_stats
```

### Multi-Provider LLM Router

Unified interface across 5 providers with automatic fallback:

| Agent Role | Primary | Fallback Chain |
|---|---|---|
| Persona | Claude Sonnet | GPT-4o → DeepSeek Chat → Qwen Plus |
| Emotion | Gemini Flash | Claude Haiku → GPT-4o-mini → Qwen Turbo |
| Feature | Claude Sonnet | GPT-4o → DeepSeek Chat → Qwen Plus |
| Scam | Claude Haiku | GPT-4o-mini → Gemini Flash → DeepSeek |
| Memory | Claude Haiku | GPT-4o-mini → Gemini Flash → Qwen Turbo |
| Question | Gemini Flash | Claude Haiku → DeepSeek → Qwen Turbo |

Cost tracking per provider. Lazy client initialization.

### Shared AgentContext

All agents share a mutable `AgentContext` per turn — emotion, memories, feature predictions, probing goals, and tool results flow through the pipeline so each agent sees the full picture.

## What's Implemented

- **Multi-LLM Router** — 5 providers (Anthropic, OpenAI, Gemini, DeepSeek, Qwen), 9 models, automatic fallback, cost tracking
- **Memory** — ChromaDB vector store, LLM-scored importance, memories injected into persona system prompts
- **Confidence Convergence** — replaced hard 30-turn cutoff; feature prediction stops early when avg confidence > 0.80
- **Question Strategy** — agent suggests 1-3 natural topics to probe low-confidence features
- **Emotion Detection** — 8 emotion categories, intensity tracking, reply strategy
- **Scam Detection** — rule-based pattern matching + LLM semantic analysis
- **Bayesian Feature Updater** — 22-dimension trait prediction with prior updating
- **Pipeline Orchestration** — parallel Phase 1, sequential Phases 2-5, shared context throughout
- **Tool Calling** — ToolRegistry + ToolExecutor with 6 built-in tools (time, web search, weather, dating advice, compatibility, conversation stats). PersonaAgent uses Claude native `tool_use` with prompt-based fallback for other providers
- **MCP Server** — Model Context Protocol server exposes 5 SoulMatch capabilities (emotion analysis, feature prediction, scam check, topic suggestion, usage stats) via stdio transport. Compatible with Claude Desktop and any MCP client
- **Structured Reasoning** — Chain-of-Thought (step-by-step decomposition with `<reasoning>`/`<conclusion>` tags) and ReAct (Thought → Action → Observation loops). FeaturePredictionAgent uses CoT from turn 3+ for more accurate trait inference
- **SFT Training Pipeline** — FeaturePredictionDataset + MemorySummarizationDataset from synthetic dialogues, SFTTrainer with Qwen3-0.6B default, ChatML formatting, train/eval split
- **RL (GRPO) Training Pipeline** — Group Relative Policy Optimization with multi-signal reward (JSON validity +0.2, feature accuracy +0.5, confidence calibration +0.15, completeness +0.15). Trains on top of SFT model

## What's NOT Implemented

- **Dynamic Multi-Agent Discussion** — fixed pipeline, no free-form agent debate
- **Local / HuggingFace LLM Inference** — router only calls cloud APIs; training uses local models but inference doesn't yet
- **Session Persistence** — sessions are in-memory only, lost on restart
- **Streaming Responses** — WebSocket sends complete messages, no token-by-token streaming
- **Web Search / Weather (live)** — tool stubs return placeholders; plug in SerpAPI/Brave + OpenWeatherMap keys to enable

## Tech Stack

- **Backend**: Python 3.12, FastAPI, WebSocket, Pydantic
- **Frontend**: React 18, TypeScript, Vite
- **Vector DB**: ChromaDB
- **LLM Providers**: Anthropic (Claude), OpenAI (GPT-4o), Google (Gemini), DeepSeek, Qwen
- **MCP**: mcp 1.26+ (stdio transport)
- **Training**: PyTorch, Transformers, TRL (SFT + GRPO)

## Project Structure

```
SoulMatch/
├── src/
│   ├── agents/
│   │   ├── orchestrator.py              # Pipeline coordinator
│   │   ├── persona_agent.py             # Bot role-play + tool calling + context injection
│   │   ├── emotion_agent.py             # 8-category emotion detection
│   │   ├── feature_prediction_agent.py  # CoT reasoning → 22-dim trait prediction
│   │   ├── scam_detection_agent.py      # Rule + semantic scam detection
│   │   ├── question_strategy_agent.py   # Low-confidence feature probing
│   │   ├── reasoning.py                 # ChainOfThought + ReAct reasoning engines
│   │   ├── llm_router.py               # Multi-provider LLM client
│   │   ├── agent_context.py             # Shared per-turn context
│   │   ├── bayesian_updater.py          # Bayesian trait updater
│   │   ├── state_machine.py             # Conversation state management
│   │   ├── prompt_generator.py          # System prompt construction
│   │   └── tools/
│   │       ├── registry.py              # Tool schema + dispatch
│   │       ├── builtin.py               # 6 built-in tools
│   │       └── executor.py              # Claude tool_use + prompt-based fallback
│   ├── mcp/
│   │   └── server.py                    # MCP server (stdio transport)
│   ├── memory/
│   │   └── memory_manager.py            # ChromaDB + LLM-scored memories
│   ├── matching/
│   │   └── matching_engine.py           # Compatibility scoring
│   ├── data/                            # OkCupid data processing
│   ├── training/
│   │   ├── conversation_simulator.py    # Bot-to-bot dialogue generation
│   │   ├── synthetic_dialogue_generator.py  # Training data generator
│   │   ├── sft_trainer.py               # SFT cold start (Qwen3-0.6B)
│   │   └── rl_trainer.py                # GRPO reinforcement learning
│   ├── api/
│   │   ├── main.py                      # FastAPI + WebSocket + tool registration
│   │   └── session_manager.py           # Session lifecycle
│   └── config.py                        # Pydantic settings
├── frontend/
│   └── src/
│       ├── App.tsx                       # React UI (select + chat pages)
│       └── index.css                     # Dark theme styles
├── mcp_config.json                      # Claude Desktop MCP integration
├── data/
│   └── processed/
│       └── bot_personas.json            # 8 bot persona definitions
├── scripts/                             # Data download, preprocessing
└── tests/                               # Unit + integration tests
```

## Setup

### 1. Install Dependencies

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn websockets anthropic openai google-genai chromadb loguru pydantic-settings scikit-learn pandas numpy torch transformers trl mcp
```

### 2. Configure API Keys

Create `.env`:

```env
ANTHROPIC_API_KEY=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
OPENAI_API_KEY=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
GEMINI_API_KEY=AIzaSyC5r9e1ltnY_8TOKrdSbWrfbyl1KkGuXOQ
DEEPSEEK_API_KEY=sk-18b0c5dddcbd486f9fe9d6cc25e2dc86
QWEN_API_KEY=sk-8a87e6ea2c1349cabe97cc6a4f6a946a
CHROMA_DB_PATH=./chroma_db
```

All 5 keys are optional — the router falls back to available providers.

### 3. Generate Bot Personas (first run)

```bash
python scripts/download_okcupid_data.py
python scripts/preprocess_data.py
```

### 4. Run

```bash
# Terminal 1 — backend
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — frontend
cd frontend && npm install && npm run dev

# Optional: MCP server (for Claude Desktop integration)
python -m src.mcp.server
```

- Backend API docs: http://localhost:8000/docs
- Frontend: http://localhost:5173
- Admin endpoints: `/api/v1/admin/usage`, `/api/v1/admin/tools`

### 5. Training (optional)

```bash
# Step 1: Generate synthetic dialogue data
python -c "from src.training import create_synthetic_dataset; create_synthetic_dataset()"

# Step 2: SFT cold start
python -m src.training.sft_trainer --data data/training/synthetic_dialogues.jsonl --model Qwen/Qwen3-0.6B --epochs 3

# Step 3: RL (GRPO) to improve pass@1
python -m src.training.rl_trainer --sft-model models/sft/final --data data/training/synthetic_dialogues.jsonl
```

## Roadmap

- [x] Multi-provider LLM router with fallback + cost tracking
- [x] Memory injection into persona prompts (ChromaDB)
- [x] Confidence-based convergence for feature prediction
- [x] Question strategy agent for low-confidence features
- [x] Shared AgentContext pipeline
- [x] Parallel Phase 1 execution (emotion + scam + memory)
- [x] English-only frontend with avatar cards
- [x] Tool calling — ToolRegistry + ToolExecutor, Claude native tool_use + prompt-based fallback
- [x] MCP server — expose agent capabilities via Model Context Protocol
- [x] Structured reasoning — CoT + ReAct chains for feature prediction
- [x] SFT cold start training — FeaturePredictionDataset + MemorySummarizationDataset + SFTTrainer
- [x] RL (GRPO) training — multi-signal reward function + GRPOTrainer
- [ ] Dynamic multi-agent discussion — agents debate before responding
- [ ] Local / HF LLM inference — serve Qwen3 or LLaMA locally via vLLM
- [ ] Session persistence — Redis or SQLite for session state
- [ ] Streaming responses — token-by-token WebSocket streaming
- [ ] Live web search / weather — plug in real API keys

## License

MIT
