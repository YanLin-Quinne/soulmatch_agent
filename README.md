# SoulMatch

AI-powered dating prediction agent. Chat with 8 bot personas — after 30 exchanges the system predicts personality traits, emotional patterns, and social tendencies using Bayesian updating with conformal prediction guarantees.

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
    │   └── FeaturePredictionAgent
    │       ├── CoT reasoning → LLM signal extraction
    │       ├── Bayesian updater → 24-dim feature posterior
    │       └── Conformal calibrator → prediction sets with 90% coverage guarantee
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

### Multi-Provider LLM Router (2026 Latest Models)

Unified interface across 5 providers with automatic fallback:

| Agent Role | Primary | Fallback Chain |
|---|---|---|
| Persona | Claude Opus 4.6 | GPT-5.2 → DeepSeek Reasoner → Qwen 3.5 Plus |
| Emotion | Gemini 2.5 Flash | Claude Haiku → GPT-4o-mini → Qwen Turbo |
| Feature | Claude Opus 4.6 | GPT-5.2 → DeepSeek Reasoner → Qwen 3.5 Plus |
| Scam | Claude Haiku | GPT-4o-mini → Gemini Flash → DeepSeek Chat |
| Memory | Claude Haiku | GPT-4o-mini → Gemini Flash → Qwen Turbo |
| Question | Gemini 3.1 Pro | Claude Haiku → DeepSeek Chat → Qwen Turbo |

Cost tracking per provider. Lazy client initialization.

### Shared AgentContext

All agents share a mutable `AgentContext` per turn — emotion, memories, feature predictions, probing goals, and tool results flow through the pipeline so each agent sees the full picture.

### Feature Prediction Pipeline

The core prediction system combines LLM signal extraction, Bayesian updating, and conformal prediction for calibrated uncertainty quantification across 24 feature dimensions.

**Problem**: LLM-reported confidence scores are uncalibrated — when the model says "80% confident", the true accuracy is often much lower. This makes confidence-based convergence unreliable.

**Solution**: Adaptive Prediction Sets (APS) from conformal prediction theory provide distribution-free coverage guarantees. Instead of a point estimate + uncalibrated confidence, the system outputs prediction sets guaranteed to contain the true value with probability ≥ 1-α.

```
Pipeline per turn:
  Conversation → LLM (CoT reasoning) → raw features + _confidence (uncalibrated)
                                              ↓
                              Bayesian updater (precision-weighted posterior)
                                              ↓
                              Conformal calibrator (APS with per-turn thresholds)
                                              ↓
                              Output: prediction sets + calibrated confidence

Example (diet prediction over 30 turns):
  Turn  5:  C(diet) = {omnivore, vegetarian, vegan}     set_size=3  "uncertain"
  Turn 15:  C(diet) = {omnivore, vegetarian}             set_size=2  "narrowing"
  Turn 25:  C(diet) = {vegetarian}                       set_size=1  "determined"
  Coverage guarantee: P(true_diet ∈ C) ≥ 90%
```

**24 feature dimensions** across 3 types:

| Type | Dimensions | CP Method |
|---|---|---|
| Categorical (10) | sex, orientation, communication_style, relationship_goals, diet, drinks, smokes, drugs, education, religion | APS over discrete options |
| Continuous → Binned (13) | Big Five (×5), interests (×8) | Discretized to {low, medium, high} then APS |
| Ordinal (1) | age | Discretized to 6 age bins then APS |

**Calibration pipeline**:
1. Generate synthetic conversations from OKCupid profiles (ground truth labels known)
2. Run feature predictor at turn checkpoints (5, 10, 15, 20, 25, 30)
3. Collect nonconformity scores: `s = 1 - P_LLM(true_label)`
4. Compute per-turn quantile thresholds `q̂_t` with finite-sample correction
5. At inference: include option `y` in prediction set if `P(y) ≥ 1 - q̂_t`

**Key properties**:
- Coverage guarantee holds regardless of LLM quality (distribution-free)
- Set size provides interpretable uncertainty metric (|C|=1 → certain, |C|=full → unknown)
- Per-turn calibration handles sequential non-exchangeability
- Graceful degradation: without calibrator, system falls back to raw LLM confidence

## What's Implemented

* **Multi-LLM Router** — 5 providers (Anthropic, OpenAI, Gemini, DeepSeek, Qwen), 15 models including GPT-5.2, Claude Opus 4.6, Gemini 3.1 Pro, DeepSeek Reasoner, Qwen 3.5 Plus, automatic fallback, cost tracking
* **Memory** — ChromaDB vector store, LLM-scored importance, memories injected into persona system prompts
* **Confidence Convergence** — replaced hard 30-turn cutoff; feature prediction stops early when avg confidence > 0.80
* **Question Strategy** — agent suggests 1-3 natural topics to probe low-confidence features
* **Emotion Detection** — 8 emotion categories, intensity tracking, reply strategy
* **Scam Detection** — rule-based pattern matching + LLM semantic analysis
* **Bayesian Feature Updater** — 24-dimension trait prediction with precision-weighted posterior updates
* **Conformal Prediction** — Adaptive Prediction Sets (APS) calibrated on synthetic dialogues; per-turn quantile thresholds; 90% coverage guarantee; save/load fitted calibrator for inference
* **Calibration Pipeline** — offline pipeline to generate calibration data from synthetic dialogues, fit conformal calibrator, evaluate coverage and ECE on held-out test set
* **Pipeline Orchestration** — parallel Phase 1, sequential Phases 2-5, shared context throughout
* **Tool Calling** — ToolRegistry + ToolExecutor with 6 built-in tools (time, web search, weather, dating advice, compatibility, conversation stats). PersonaAgent uses Claude native `tool_use` with prompt-based fallback for other providers
* **MCP Server** — Model Context Protocol server exposes 5 SoulMatch capabilities (emotion analysis, feature prediction, scam check, topic suggestion, usage stats) via stdio transport. Compatible with Claude Desktop and any MCP client
* **Structured Reasoning** — Chain-of-Thought (step-by-step decomposition with `<reasoning>`/`<conclusion>` tags) and ReAct (Thought → Action → Observation loops). FeaturePredictionAgent uses CoT from turn 3+ for more accurate trait inference
* **SFT Training Pipeline** — FeaturePredictionDataset + MemorySummarizationDataset from synthetic dialogues, SFTTrainer with Qwen3-0.6B default, ChatML formatting, train/eval split
* **RL (GRPO) Training Pipeline** — Group Relative Policy Optimization with multi-signal reward (JSON validity +0.2, feature accuracy +0.5, confidence calibration +0.15, completeness +0.15). Trains on top of SFT model

## What's NOT Implemented

* **Dynamic Multi-Agent Discussion** — fixed pipeline, no free-form agent debate
* **Local / HuggingFace LLM Inference** — router only calls cloud APIs; training uses local models but inference doesn't yet
* **Session Persistence** — sessions are in-memory only, lost on restart
* **Streaming Responses** — WebSocket sends complete messages, no token-by-token streaming
* **Web Search / Weather (live)** — tool stubs return placeholders; plug in SerpAPI/Brave + OpenWeatherMap keys to enable

## Tech Stack

* **Backend**: Python 3.10+, FastAPI, WebSocket, Pydantic
* **Frontend**: React 18, TypeScript, Vite
* **Vector DB**: ChromaDB
* **LLM Providers**: Anthropic (Claude Opus 4.6), OpenAI (GPT-5.2), Google (Gemini 3.1 Pro), DeepSeek (Reasoner V3.2), Qwen (3.5 Plus)
* **MCP**: mcp 1.26+ (stdio transport)
* **Training**: PyTorch, Transformers, TRL (SFT + GRPO)
* **Uncertainty Quantification**: Conformal Prediction (APS), NumPy

## Project Structure

```
SoulMatch/
├── src/
│   ├── agents/
│   │   ├── orchestrator.py              # Pipeline coordinator
│   │   ├── persona_agent.py             # Bot role-play + tool calling + context injection
│   │   ├── emotion_agent.py             # 8-category emotion detection
│   │   ├── feature_prediction_agent.py  # CoT reasoning → Bayesian update → conformal calibration
│   │   ├── conformal_calibrator.py      # APS conformal prediction with per-turn calibration
│   │   ├── bayesian_updater.py          # Precision-weighted Bayesian trait updater
│   │   ├── scam_detection_agent.py      # Rule + semantic scam detection
│   │   ├── question_strategy_agent.py   # Low-confidence feature probing
│   │   ├── reasoning.py                 # ChainOfThought + ReAct reasoning engines
│   │   ├── llm_router.py               # Multi-provider LLM client
│   │   ├── agent_context.py             # Shared per-turn context
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
│   │   ├── calibration_pipeline.py      # Conformal prediction calibration pipeline
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
├── data/
│   ├── raw/
│   │   └── okcupid_profiles.csv         # 59,946 OkCupid profiles
│   ├── processed/
│   │   ├── okcupid_processed.parquet    # Cleaned profiles (56,789)
│   │   └── bot_personas.json            # 8 bot persona definitions
│   ├── training/
│   │   └── synthetic_dialogues*.jsonl   # Generated bot-to-bot conversations
│   └── calibration/
│       ├── conformal_calibrator.json    # Fitted calibrator (per-turn quantiles)
│       └── evaluation_results.json      # Coverage + efficiency metrics
├── mcp_config.json                      # Claude Desktop MCP integration
├── scripts/                             # Data download, preprocessing
└── tests/                               # Unit + integration tests
```

## Setup

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn websockets anthropic openai google-genai chromadb loguru pydantic-settings scikit-learn pandas numpy torch transformers trl mcp
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

All 5 LLM keys are optional — the router falls back to available providers.

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

* Backend API docs: http://localhost:8000/docs
* Frontend: http://localhost:5173
* Admin endpoints: `/api/v1/admin/usage`, `/api/v1/admin/tools`

### 5. Calibrate Conformal Predictor

```bash
# Step 1: Generate synthetic dialogues (requires LLM API key, ~$3-5)
python -c "from src.training import create_synthetic_dataset; create_synthetic_dataset()"

# Step 2: Fit conformal calibrator (fast, no API calls)
python -m src.training.calibration_pipeline --mode simulate --dialogues data/training/synthetic_dialogues_balanced.jsonl

# Step 3 (optional): Evaluate calibration quality
python -m src.training.calibration_pipeline --mode evaluate --calibrator data/calibration/conformal_calibrator.json --dialogues data/training/synthetic_dialogues_balanced.jsonl
```

The fitted calibrator is automatically loaded by `FeaturePredictionAgent` at startup. Without it, the system gracefully falls back to raw LLM confidence.

### 6. Training (optional)

```bash
# SFT cold start
python -m src.training.sft_trainer --data data/training/synthetic_dialogues.jsonl --model Qwen/Qwen3-0.6B --epochs 3

# RL (GRPO) to improve pass@1
python -m src.training.rl_trainer --sft-model models/sft/final --data data/training/synthetic_dialogues.jsonl
```

## References

* Angelopoulos & Bates (2021). [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511)
* Kumar et al. (2023). [Conformal Prediction with Large Language Models for Multi-Choice Question Answering](https://arxiv.org/abs/2305.18404)
* Sheng et al. (2025). [Conformal Prediction for Analyzing Uncertainty of LLM-as-a-Judge](https://arxiv.org/abs/2509.18658)
* Gibbs & Candès (2021). [Adaptive Conformal Inference Under Distribution Shift](https://arxiv.org/abs/2106.00170)

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
- [x] Conformal prediction — APS calibration pipeline, per-turn coverage guarantees, ECE evaluation
- [ ] Dynamic multi-agent discussion — agents debate before responding
- [ ] Local / HF LLM inference — serve Qwen3 or LLaMA locally via vLLM
- [ ] Session persistence — Redis or SQLite for session state
- [ ] Streaming responses — token-by-token WebSocket streaming
- [ ] Live web search / weather — plug in real API keys

## License

MIT
