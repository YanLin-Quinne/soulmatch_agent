# SoulMatch v2.0 - Agent I/O Specification & Communication Protocol

## Overview

SoulMatch v2.0 uses a **shared context architecture** where all agents read from and write to a central `AgentContext` object. This document specifies the exact input/output contract for each agent and the communication flow.

---

## Core Communication Pattern

```
User Message → Orchestrator → Agent Pipeline → PersonaAgent → Response
                    ↓
              AgentContext (shared state)
                    ↑
         All agents read/write here
```

**Key principle**: Agents do NOT directly communicate with each other. They communicate through the shared `AgentContext` object.

---

## Agent I/O Specifications

### 1. EmotionAgent

**File**: `src/agents/emotion_agent.py`

**Input**:
```python
{
    "user_message": str,           # Current user message
    "conversation_history": list   # Last 5 turns for context
}
```

**Output** (writes to `ctx`):
```python
ctx.current_emotion = {
    "emotion": str,                # e.g., "joy", "sadness", "anger"
    "confidence": float,           # 0.0-1.0
    "intensity": float,            # 0.0-1.0
    "valence": float,              # -1.0 to 1.0 (negative to positive)
    "arousal": float               # 0.0-1.0 (calm to excited)
}
ctx.reply_strategy = str           # e.g., "empathetic", "encouraging"
```

**Trigger**: Every turn

**Execution time**: ~0.5-1s (fast model: Gemini Flash)

---

### 2. ScamDetectionAgent

**File**: `src/agents/scam_detection_agent.py`

**Input**:
```python
{
    "user_message": str,
    "conversation_history": list   # Last 10 turns
}
```

**Output** (writes to `ctx`):
```python
ctx.scam_risk_score = float        # 0.0-1.0
ctx.scam_warning_level = str       # "none", "low", "medium", "high"
ctx.scam_patterns = list[str]      # Detected patterns
```

**Trigger**: Every 2 turns

**Execution time**: ~0.3-0.5s

---

### 3. MemoryManager.retrieve()

**File**: `src/agents/memory_manager.py`

**Input**:
```python
{
    "query": str,                  # Current user message
    "user_id": str,
    "top_k": int                   # Default: 5
}
```

**Output** (writes to `ctx`):
```python
ctx.retrieved_memories = [
    {
        "content": str,
        "timestamp": str,
        "relevance_score": float,
        "memory_type": str         # "conversation", "feature", "milestone"
    },
    ...
]
```

**Trigger**: Every turn

**Execution time**: ~0.1-0.2s (ChromaDB vector search)

---

### 4. FeaturePredictionAgent

**File**: `src/agents/feature_prediction_agent.py`

**Input**:
```python
{
    "conversation_history": list,  # Last 10 turns
    "current_emotion": dict,       # From EmotionAgent
    "retrieved_memories": list     # From MemoryManager
}
```

**Output** (writes to `ctx`):
```python
ctx.predicted_features = {
    # Big Five (5 dimensions)
    "big_five_openness": float,
    "big_five_conscientiousness": float,
    "big_five_extraversion": float,
    "big_five_agreeableness": float,
    "big_five_neuroticism": float,

    # MBTI (6 dimensions)
    "mbti_type": str,              # e.g., "INFP"
    "mbti_confidence": float,
    "mbti_ei": float,              # Extraversion-Introversion axis
    "mbti_sn": float,              # Sensing-Intuition axis
    "mbti_tf": float,              # Thinking-Feeling axis
    "mbti_jp": float,              # Judging-Perceiving axis

    # Attachment (3 dimensions)
    "attachment_style": str,       # "secure", "anxious", "avoidant", "disorganized"
    "attachment_anxiety": float,
    "attachment_avoidance": float,

    # Love languages (2 dimensions)
    "primary_love_language": str,
    "secondary_love_language": str,

    # Trust (3 dimensions)
    "trust_score": float,
    "trust_velocity": float,
    "trust_history": list[float]
}

ctx.feature_confidences = {
    "big_five_openness": float,    # 0.0-1.0 for each feature
    ...
}

ctx.conformal_result = {
    "prediction_sets": dict,       # Conformal prediction sets
    "coverage_guarantee": float    # 1-alpha (e.g., 0.90)
}
```

**Trigger**: Every 3 turns

**Execution time**: ~2-3s (Claude Opus 4.6)

**MBTI Inference Logic**:
```python
# Based on psychometric correlations
mbti_ei = big_five_extraversion           # r=0.74
mbti_sn = big_five_openness              # r=0.72
mbti_tf = 1.0 - big_five_agreeableness   # r=-0.44
mbti_jp = big_five_conscientiousness     # r=0.49
```

---

### 5. FeatureTransitionPredictor

**File**: `src/agents/feature_transition_predictor.py`

**Input**:
```python
{
    "current_features": dict,      # From ctx.predicted_features
    "feature_history": list,       # From ctx.feature_history
    "emotion_trend": str,          # "improving", "declining", "stable"
    "relationship_status": str,    # From ctx.rel_status
    "memory_trigger": bool         # Whether memory was just updated
}
```

**Output** (writes to `ctx`):
```python
ctx.predicted_feature_changes = {
    "likely_to_change": list[str],         # Feature names
    "predicted_direction": dict[str, str], # "+", "-", or "0"
    "change_probability": dict[str, float],# 0.0-1.0
    "stable_features": list[str]           # Features unlikely to change
}
```

**Trigger**: Every 3 turns (after FeaturePredictionAgent)

**Execution time**: ~0.1s (rule-based, no LLM)

**Change Probability Rules**:
```python
CHANGE_PROBS = {
    # Personality traits: very stable
    "big_five_*": 0.02,
    "mbti_type": 0.01,

    # Trust: highly dynamic
    "trust_score": 0.70,
    "trust_velocity": 0.60,

    # Attachment: slowly evolving
    "attachment_anxiety": 0.15,
    "attachment_avoidance": 0.10,

    # Love languages: moderately stable
    "primary_love_language": 0.05,
    "secondary_love_language": 0.08
}
```

---

### 6. RelationshipPredictionAgent

**File**: `src/agents/relationship_prediction_agent.py`

**Input**:
```python
{
    "conversation_history": list,  # Last 50 turns
    "retrieved_memories": list,
    "current_emotion": dict,
    "predicted_features": dict,
    "turn_count": int
}
```

**Output** (writes to `ctx`):
```python
ctx.relationship_result = {
    # Core predictions
    "sentiment": str,                      # "positive", "neutral", "negative"
    "sentiment_confidence": float,
    "rel_type": str,                       # "love", "friendship", "family", "other"
    "rel_type_probs": dict[str, float],
    "rel_status": str,                     # "stranger", "acquaintance", "crush", "dating", "committed"
    "rel_status_probs": dict[str, float],

    # Conformal prediction for advancement
    "can_advance": bool,                   # Point prediction
    "advance_prediction_set": list[str],   # e.g., ["yes", "uncertain"]
    "advance_coverage_guarantee": float,   # 0.90

    # Time-series prediction (t+1)
    "next_status_prediction": str,
    "next_status_probs": dict[str, float],

    # Milestone report (only at turns 10/30)
    "milestone_report": dict | None,

    # Internal snapshot
    "snapshot": {
        "turn": int,
        "rel_status": str,
        "trust_score": float,
        "sentiment": str,
        "timestamp": str
    },

    "reasoning_trace": str                 # LLM reasoning
}

# Also updates
ctx.rel_status = str
ctx.rel_type = str
ctx.sentiment_label = str
ctx.can_advance = bool
ctx.relationship_snapshots.append(snapshot)
```

**Trigger**: Every 5 turns + turns 10/30

**Execution time**: ~3-5s (multi-role LLM assessment)

**Internal 5-Step Workflow**:
```
1. Context Compression (50 turns → 15 lines)
   Input: conversation_history
   Output: compressed_context (str)

2. Sentiment Analysis (sliding window)
   Input: emotion_history (last 5 turns)
   Output: sentiment ("positive"/"neutral"/"negative")

3. Multi-Role Assessment (3 LLM calls in parallel)
   Role 1: Emotion Expert (weight=0.4)
   Role 2: Values Expert (weight=0.3)
   Role 3: Behavior Expert (weight=0.3)
   Output: weighted_assessment

4. Conformal Prediction
   Input: predicted_features, rel_status
   Output: advance_prediction_set, coverage_guarantee

5. t+1 Prediction (Markov transition)
   Input: current_rel_status
   Output: next_status_probs
```

---

### 7. MilestoneEvaluator

**File**: `src/agents/milestone_evaluator.py`

**Input**:
```python
{
    "turn": int,                   # 10 or 30
    "feature_history": list,
    "relationship_snapshots": list,
    "current_features": dict
}
```

**Output** (writes to `ctx`):
```python
ctx.milestone_reports[turn] = {
    # Turn 10: Initial assessment
    "current_status": str,
    "predicted_status_at_30": str,
    "predicted_status_probs": dict,
    "sentiment_trend": str,
    "trust_level": float,
    "confidence": float,

    # Turn 30: Precise assessment
    "final_status": str,
    "prediction_accuracy": float,      # Compare turn 10 prediction vs actual
    "conformal_efficiency": float,     # Avg prediction set size
    "memory_contribution": float,      # How much memory helped
    "feature_convergence": dict,       # Which features stabilized
    "trust_trajectory_analysis": dict
}
```

**Trigger**: Turn 10 and turn 30 only

**Execution time**: ~1-2s (statistical analysis, no LLM)

---

### 8. QuestionStrategyAgent

**File**: `src/agents/question_strategy_agent.py`

**Input**:
```python
{
    "predicted_features": dict,
    "feature_confidences": dict,
    "conversation_history": list
}
```

**Output** (writes to `ctx`):
```python
ctx.suggested_probes = [
    {
        "feature": str,            # e.g., "big_five_openness"
        "current_confidence": float,
        "probe_question": str,     # Suggested question to ask
        "priority": int            # 1-5
    },
    ...
]
```

**Trigger**: Every turn (if any feature confidence < 0.6)

**Execution time**: ~0.5-1s

---

### 9. DiscussionEngine

**File**: `src/agents/discussion_engine.py`

**Input**:
```python
{
    "ctx": AgentContext  # Full context
}
```

**Output** (writes to `ctx`):
```python
ctx.discussion_synthesis = str  # Summary of all agent findings
```

**Trigger**: Every 3 turns

**Execution time**: ~1s

---

### 10. PersonaAgent

**File**: `src/agents/persona_agent.py`

**Input**:
```python
{
    "user_message": str,
    "ctx": AgentContext,           # Full context including:
                                   # - current_emotion
                                   # - predicted_features
                                   # - relationship_result
                                   # - suggested_probes
                                   # - discussion_synthesis
    "anti_ai_delay": float         # Gaussian sampled delay
}
```

**Output**:
```python
bot_response = str                 # Final response to user
```

**Trigger**: Every turn

**Execution time**: ~2-3s + anti_ai_delay (μ=2s, σ=1s)

**Context Injection**:
```python
system_prompt = f"""
You are {bot_name}, {bot_description}.

Current relationship context:
- Status: {ctx.rel_status}
- Type: {ctx.rel_type}
- Sentiment: {ctx.sentiment_label}
- Trust: {ctx.predicted_features['trust_score']:.2f}
- Can advance: {ctx.can_advance}

User's inferred traits:
- MBTI: {ctx.predicted_features['mbti_type']}
- Attachment: {ctx.predicted_features['attachment_style']}
- Primary love language: {ctx.predicted_features['primary_love_language']}

Current emotion: {ctx.current_emotion['emotion']} (intensity: {ctx.current_emotion['intensity']:.2f})
Reply strategy: {ctx.reply_strategy}

{ctx.discussion_synthesis}
"""
```

---

### 11. MemoryManager.execute()

**File**: `src/agents/memory_manager.py`

**Input**:
```python
{
    "recent_history": list,        # Last 5 turns
    "predicted_features": dict,
    "relationship_result": dict
}
```

**Output**:
```python
{
    "memory_action": str,          # "store", "update", "delete"
    "memories_affected": list[str] # Memory IDs
}
```

**Trigger**: Every 5 turns

**Execution time**: ~0.5-1s (ChromaDB operations)

---

### 12. MatchingEngine

**File**: `src/agents/matching_engine.py`

**Input**:
```python
{
    "user_feature_vector": np.ndarray,  # 42 dimensions
    "candidates": list[dict]            # Bot profiles
}
```

**Output**:
```python
{
    "ranked_candidates": list[dict],
    "compatibility_scores": dict[str, float]  # bot_id -> score
}
```

**Trigger**: Session start + every 10 turns

**Execution time**: ~0.1s (cosine similarity)

---

## Orchestrator Pipeline (8 Steps)

**File**: `src/agents/orchestrator.py`

```python
async def process_turn(user_message: str) -> str:
    # Step 1: Parallel retrieval
    await asyncio.gather(
        emotion_agent.execute(ctx),
        scam_agent.execute(ctx),
        memory_manager.retrieve(ctx)
    )

    # Step 2: Feature prediction (every 3 turns)
    if ctx.turn_count % 3 == 0:
        await feature_prediction_agent.execute(ctx)

    # Step 3: Feature transition prediction (every 3 turns)
    if ctx.turn_count % 3 == 0:
        feature_transition_predictor.predict_next(ctx)

    # Step 4: Relationship prediction (every 5 turns + milestones)
    if ctx.turn_count % 5 == 0 or ctx.turn_count in [10, 30]:
        await relationship_prediction_agent.execute(ctx)

    # Step 5: Milestone evaluation (turns 10/30)
    if ctx.turn_count in [10, 30]:
        milestone_evaluator.evaluate(ctx)

    # Step 6: Question strategy (if needed)
    if any(conf < 0.6 for conf in ctx.feature_confidences.values()):
        await question_strategy_agent.execute(ctx)

    # Step 7: Discussion synthesis (every 3 turns)
    if ctx.turn_count % 3 == 0:
        await discussion_engine.execute(ctx)

    # Step 8: Persona response
    response = await persona_agent.execute(ctx)

    # Step 9: Memory storage (every 5 turns)
    if ctx.turn_count % 5 == 0:
        await memory_manager.execute(ctx)

    return response
```

---

## WebSocket Communication Protocol

**File**: `src/api/main.py`

### Client → Server

```json
{
    "type": "user_message",
    "content": "Hello, how are you?",
    "user_id": "user_123",
    "session_id": "session_456"
}
```

### Server → Client Events

#### 1. Bot Response
```json
{
    "type": "bot_response",
    "content": "I'm doing great! How about you?",
    "bot_id": "mina",
    "turn": 5
}
```

#### 2. Emotion Update (every turn)
```json
{
    "type": "emotion_update",
    "data": {
        "emotion": "joy",
        "confidence": 0.85,
        "intensity": 0.72,
        "valence": 0.8,
        "arousal": 0.6
    }
}
```

#### 3. Feature Update (every 3 turns)
```json
{
    "type": "feature_update",
    "data": {
        "predicted_features": {...},
        "confidences": {...},
        "conformal_result": {...}
    }
}
```

#### 4. Relationship Prediction (every 5 turns)
```json
{
    "type": "relationship_prediction",
    "data": {
        "sentiment": "positive",
        "rel_status": "acquaintance",
        "rel_type": "friendship",
        "can_advance": true,
        "advance_prediction_set": ["yes"],
        "trust_score": 0.65,
        "turn": 10
    }
}
```

#### 5. Milestone Report (turns 10/30)
```json
{
    "type": "milestone_report",
    "data": {
        "turn": 10,
        "current_status": "acquaintance",
        "predicted_status_at_30": "crush",
        "sentiment_trend": "improving",
        "trust_level": 0.65,
        "confidence": 0.78
    }
}
```

#### 6. Scam Warning (when detected)
```json
{
    "type": "scam_warning",
    "data": {
        "risk_score": 0.75,
        "warning_level": "high",
        "patterns": ["financial_request", "urgency"]
    }
}
```

---

## Multi-Agent Communication: Why NOT MCP/Skills?

### Current Architecture: Shared Context (Blackboard Pattern)

```
┌─────────────────────────────────────────┐
│         AgentContext (Shared)           │
│  ┌───────────────────────────────────┐  │
│  │ • current_emotion                 │  │
│  │ • predicted_features              │  │
│  │ • relationship_result             │  │
│  │ • retrieved_memories              │  │
│  │ • scam_risk_score                 │  │
│  │ • suggested_probes                │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
         ↑                    ↑
         │ write              │ read
         │                    │
    ┌────┴────┐          ┌────┴────┐
    │ Agent A │          │ Agent B │
    └─────────┘          └─────────┘
```

**Advantages**:
- Simple: All agents read/write to one place
- Fast: No message passing overhead
- Transparent: Easy to debug (inspect ctx at any point)
- Flexible: Agents can access any data they need

### Why NOT MCP (Model Context Protocol)?

MCP is designed for **external tool integration** (e.g., connecting to databases, APIs, file systems). Our agents are **internal components** that share the same process space.

**MCP would add unnecessary complexity**:
```
Agent A → MCP Server → MCP Client → Agent B
  (overhead: serialization, network, deserialization)

vs.

Agent A → ctx → Agent B
  (direct memory access)
```

### Why NOT Skills?

Skills are for **user-invocable commands** (like `/commit`, `/review-pr`). Our agents are **automatic pipeline components** that run on every turn.

**Skills would break the pipeline flow**:
```
User → Orchestrator → Skill Invocation → Wait for user → Continue
  (breaks automatic flow)

vs.

User → Orchestrator → Agent Pipeline → Response
  (seamless automatic flow)
```

### When WOULD We Use MCP/Skills?

**MCP**: If we need to integrate external services:
- Fetch real-time weather data
- Query external knowledge bases
- Access user's calendar/email

**Skills**: If we add user-invocable commands:
- `/analyze-relationship` - Generate detailed relationship report
- `/predict-future` - Predict relationship at turn 50
- `/export-data` - Export conversation data

---

## Future: Multi-Agent Discussion (Not Yet Implemented)

If we want agents to **debate** before making decisions (like the multi-agent paper), we would add:

```python
class AgentDiscussionRoom:
    """
    Agents discuss and reach consensus before writing to ctx.
    """

    async def discuss(self, topic: str, agents: list[Agent]) -> dict:
        """
        Round-robin discussion:
        1. Each agent proposes their view
        2. Agents critique each other
        3. Weighted voting to reach consensus
        """
        proposals = []
        for agent in agents:
            proposal = await agent.propose(topic, ctx)
            proposals.append(proposal)

        critiques = []
        for agent in agents:
            critique = await agent.critique(proposals)
            critiques.append(critique)

        consensus = self._weighted_vote(proposals, critiques)
        return consensus
```

**Example usage**:
```python
# In RelationshipPredictionAgent
discussion_room = AgentDiscussionRoom()
consensus = await discussion_room.discuss(
    topic="Can this relationship advance?",
    agents=[emotion_expert, values_expert, behavior_expert]
)
ctx.can_advance = consensus["decision"]
```

This is **not yet implemented** because:
1. Adds 3x latency (3 LLM calls → 9 LLM calls)
2. Current weighted aggregation works well
3. Can add later if needed for paper experiments

---

## Summary

| Component | Input | Output | Trigger | Latency |
|---|---|---|---|---|
| EmotionAgent | user_message | current_emotion | Every turn | 0.5-1s |
| ScamDetectionAgent | user_message | scam_risk_score | Every 2 turns | 0.3-0.5s |
| MemoryManager.retrieve | query | retrieved_memories | Every turn | 0.1-0.2s |
| FeaturePredictionAgent | history + emotion | predicted_features (42D) | Every 3 turns | 2-3s |
| FeatureTransitionPredictor | features + trend | predicted_changes | Every 3 turns | 0.1s |
| RelationshipPredictionAgent | full context | relationship_result | Every 5 turns | 3-5s |
| MilestoneEvaluator | history + snapshots | milestone_report | Turns 10/30 | 1-2s |
| QuestionStrategyAgent | features + confidences | suggested_probes | When needed | 0.5-1s |
| DiscussionEngine | full context | discussion_synthesis | Every 3 turns | 1s |
| PersonaAgent | full context | bot_response | Every turn | 2-3s |
| MemoryManager.execute | recent history | memory_action | Every 5 turns | 0.5-1s |

**Total latency per turn**: ~3-5s (most agents run conditionally)

**Communication**: Shared context (blackboard pattern), not MCP/Skills

**Future**: Can add agent discussion room for multi-agent debate if needed for research
