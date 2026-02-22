# SoulMatch v2.0 Quick Start Guide

## Requirements

- Python 3.9+
- Node.js 16+
- Dependencies installed (requirements.txt + package.json)

## Startup Steps

### 1. Start Backend

```bash
cd /Users/quinne/Desktop/soulmatch_agent_test

# Ensure environment variables are configured correctly
# Check API keys in .env file

# Start FastAPI server
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will start at `http://localhost:8000`

### 2. Start Frontend (New Terminal)

```bash
cd /Users/quinne/Desktop/soulmatch_agent_test/frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

Frontend will start at `http://localhost:5173`

### 3. Access Application

Open browser and visit: `http://localhost:5173`

## Testing v2.0 New Features

### Test Scenario 1: Basic Relationship Prediction Flow

1. On LobbyConfig page, select a Bot (e.g., Mina)
2. Enable "You (real person)" toggle (real person mode)
3. Click "Start with Mina →"
4. Conduct at least 10 turns of conversation
5. Observe the "Relation" tab in the right sidebar:
   - Relationship status progress bar (stranger→acquaintance→...)
   - Conformal prediction badge (Can Advance?)
   - Trust trajectory line chart

### Test Scenario 2: Milestone Evaluation

1. Continue conversation to turn 10
2. Observe "Relation" tab auto-popup milestone report
3. Report content:
   - Current relationship status
   - Predicted status at turn 30
   - Sentiment trend and trust level
4. Continue conversation to turn 30
5. Observe turn 30 precise evaluation report

### Test Scenario 3: Feature Inference (Real Person Mode)

1. Mention your interests, personality, values in conversation
2. Observe "Predict" tab feature confidence real-time updates
3. After turn 3, system will infer your MBTI type and attachment style
4. Observe "Relation" tab trust score changes

### Test Scenario 4: Conformal Prediction Visualization

1. After turn 5, observe "Can Advance?" badge
2. Badge color meanings:
   - Green ["yes"]: Can advance relationship
   - Yellow ["yes","uncertain"]: Maybe advance
   - Red ["uncertain","no"]: Not ready to advance
3. Badge displays "@ 90%" indicating 90% coverage guarantee

## Verifying Core Functionality

### Backend Verification

```bash
# Run integration tests
python test_v2_integration.py

# Expected output:
# ✓ AgentContext extended fields test passed
# ✓ FeatureTransitionPredictor test passed
# ✓ MilestoneEvaluator test passed
# ✓ relationship_context_block test passed
# All tests passed! ✓
```

### Frontend Verification

1. Check browser console for no errors
2. Check WebSocket connection success (Network tab)
3. Check relationship prediction data correctly received:
   - `relationship_prediction` event (every 5 turns)
   - `milestone_report` event (turns 10/30)

### API Verification

```bash
# Check backend health status
curl http://localhost:8000/health

# Check WebSocket endpoint
# (requires WebSocket client tool, e.g., wscat)
wscat -c ws://localhost:8000/ws/test_user
```

## Debugging Tips

### Backend Logs

Backend uses loguru for detailed logging, key log tags:

- `[RelationshipPredictionAgent]`: Relationship prediction workflow
- `[FeatureTransitionPredictor]`: Feature change prediction
- `[MilestoneEvaluator]`: Milestone evaluation
- `[FeaturePredictionAgent]`: MBTI/attachment inference

### Frontend Debugging

Open browser developer tools:

1. **Console**: View WebSocket messages and errors
2. **Network → WS**: View WebSocket communication
3. **React DevTools**: View component state

### Common Issues

**Q: Relationship prediction not showing?**
A: Relationship prediction starts from turn 5, ensure conversation has at least 5 turns

**Q: Milestone report not popping up?**
A: Milestone report only triggers at turns 10 and 30, check turnCount

**Q: Trust trajectory chart empty?**
A: Trust trajectory needs at least 2 data points, ensure conversation has at least 10 turns

**Q: MBTI inference is null?**
A: MBTI inference requires complete Big Five features, ensure conversation is deep enough

## Performance Optimization Recommendations

1. **Reduce LLM Calls**:
   - RelationshipAgent triggers every 5 turns (adjustable)
   - FeaturePredictionAgent triggers every 3 turns (adjustable)

2. **Frontend Optimization**:
   - Trust trajectory chart uses SVG instead of Canvas (lighter weight)
   - Milestone report auto-closes after 8 seconds

3. **Backend Optimization**:
   - Context compression reduces token consumption (50 turns→15 lines)
   - Conformal prediction uses pre-trained quantiles (no real-time calculation needed)

## Next Development Steps

1. **Anti-AI Delay Mechanism**:
   - Modify `persona_agent.py`
   - Add Gaussian delay (μ=2s, σ=1s)

2. **Paper Experiments**:
   - Generate synthetic dialogue dataset
   - Run ablation experiments
   - Calculate evaluation metrics

3. **Frontend Enhancements**:
   - Add LobbyConfig page (currently using original selection page)
   - Add ParticipantBar component (multi-participant display)

## Technical Support

- Project directory: `/Users/quinne/Desktop/soulmatch_agent_test`
- Implementation summary: `IMPLEMENTATION_SUMMARY_V2.md`
- Test script: `test_v2_integration.py`

## Paper-Related

### Core Metrics Calculation

```python
# ECE (Expected Calibration Error)
from evaluation.metrics import compute_ece

ece = compute_ece(predictions, ground_truth, confidences)

# Conformal Coverage
from evaluation.metrics import compute_coverage

coverage = compute_coverage(prediction_sets, ground_truth)

# F1-score on relationship status
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred, average='weighted')
```

### Ablation Experiment Configuration

1. **w/o memory**: Disable MemoryManager
2. **w/o conformal**: Use point prediction
3. **w/o multi-role**: Use single LLM evaluation
4. **w/o extended features**: Use only original 24 dimensions
5. **Complete system**: SoulMatch v2.0

## LLM Configuration (2026-02-22)

### Supported Models
The system now supports 5 state-of-the-art LLM providers:

1. **OpenAI GPT-5.2** - Latest flagship model with advanced reasoning
2. **Google Gemini 3.1 Pro Preview** - Advanced reasoning (fallback: Gemini 2.5 Flash)
3. **Anthropic Claude Opus 4.6** - Most capable Claude model
4. **Alibaba Qwen 3.5 Plus** - Latest Qwen series
5. **DeepSeek Reasoner V3.2** - Advanced reasoning model

### API Keys
All API keys are pre-configured in `.env.example`. The system automatically handles fallback if a provider is unavailable.

### Model Selection Strategy
- **High-quality tasks**: Claude Opus 4.6 → GPT-5.2 → DeepSeek Reasoner
- **Fast tasks**: Gemini 3.1 Pro / 2.5 Flash → Claude Haiku
- **Cost-effective tasks**: Claude Haiku → GPT-4o-mini → Gemini Flash

---

**Happy experimenting!** 🚀
