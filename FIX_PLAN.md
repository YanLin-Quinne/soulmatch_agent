# SoulMatch Agent 前后端修复方案

## 问题诊断总结

### ✅ 已修复
1. **WebSocket 端口不匹配**
   - 前端原本连接 `ws://localhost:8000`
   - 后端运行在 `7860`
   - **已修复**：`frontend/src/App.tsx:13` 改为 `ws://localhost:7860`

### ❌ 核心缺失功能

#### 1. Social Turing Challenge（图灵测试）- 完全缺失

**论文要求的三阶段：**
- ✅ Progressive Profiling (30轮推断) - 已实现
- ❌ **Social Turing Challenge** - 完全缺失
- ⚠️ Digital Twin Reflection - 部分实现

**缺失内容：**
- 后端没有图灵测试 API 端点
- 前端没有猜测界面
- 没有真人/AI 标签管理
- 没有结果评分系统

#### 2. Digital Twin 对比功能不完整

- 前端有 `ComparisonView.tsx` 但后端对比算法不完整
- 缺少维度匹配度计算
- 缺少不匹配分析

## 当前实现状态

### 后端（已有）
- ✅ `src/agents/orchestrator.py` - 12个协作 agent
- ✅ `src/agents/conversation_sentiment_agent.py` - 对话情感分析
- ✅ `src/agents/digital_twin_agent.py` - 数字分身（基础版）
- ✅ `src/agents/relationship_prediction_agent.py` - 关系预测
- ✅ `src/api/websocket.py` - WebSocket 协议
- ✅ 三层记忆架构（Working/Episodic/Semantic）
- ✅ Bayesian + Conformal Prediction

### 前端（已有）
- ✅ `frontend/src/App.tsx` - 完整聊天界面
- ✅ `frontend/src/components/DigitalTwinSetup.tsx` - 数字分身设置
- ✅ `frontend/src/components/ComparisonView.tsx` - 对比视图
- ✅ `frontend/src/components/RelationshipTab.tsx` - 关系预测展示
- ✅ 实时特征推断可视化

## 修复优先级

### P0 - 已完成 ✅
- [x] WebSocket 端口修复

### P1 - 高优先级（核心功能）
- [ ] 实现 Social Turing Challenge 后端 API
- [ ] 实现 Social Turing Challenge 前端界面
- [ ] 完善 Digital Twin 对比算法

### P2 - 中优先级（流程优化）
- [ ] 对齐论文三阶段流程转换
- [ ] 添加阶段状态管理

## 具体实施方案

### 方案 1：实现 Social Turing Challenge

#### 后端修改

**新增文件：`src/agents/turing_challenge_agent.py`**
```python
"""Social Turing Challenge Agent"""

class TuringChallengeAgent:
    def __init__(self):
        self.bot_labels = {}  # bot_id -> is_ai (True/False)

    def start_challenge(self, user_id: str, bot_id: str) -> dict:
        """开始图灵测试，返回提示信息"""
        return {
            "message": "Based on your 30-turn conversation, guess: Is this a real person or AI?",
            "options": ["Real Person", "AI Bot"]
        }

    def submit_guess(self, user_id: str, bot_id: str, guess: str) -> dict:
        """提交猜测，返回正确答案和得分"""
        is_ai = self.bot_labels.get(bot_id, True)
        correct_answer = "AI Bot" if is_ai else "Real Person"
        is_correct = (guess == correct_answer)

        return {
            "correct": is_correct,
            "your_guess": guess,
            "actual": correct_answer,
            "score": 100 if is_correct else 0,
            "explanation": f"This was {'an AI bot' if is_ai else 'a real person'}."
        }
```

**修改文件：`src/api/main.py`**
```python
# 添加新的端点
@app.post("/api/turing/start", tags=["Turing"])
async def start_turing_challenge(user_id: str):
    """开始图灵测试"""
    orchestrator = session_manager.get_session(user_id)
    if not orchestrator:
        raise HTTPException(status_code=404, detail="Session not found")

    result = orchestrator.turing_agent.start_challenge(
        user_id,
        orchestrator.preferred_bot_id
    )
    return {"success": True, "data": result}

@app.post("/api/turing/guess", tags=["Turing"])
async def submit_turing_guess(user_id: str, guess: str):
    """提交猜测"""
    orchestrator = session_manager.get_session(user_id)
    if not orchestrator:
        raise HTTPException(status_code=404, detail="Session not found")

    result = orchestrator.turing_agent.submit_guess(
        user_id,
        orchestrator.preferred_bot_id,
        guess
    )
    return {"success": True, "data": result}
```

**修改文件：`src/agents/orchestrator.py`**
```python
# 在 __init__ 中添加
from src.agents.turing_challenge_agent import TuringChallengeAgent

self.turing_agent = TuringChallengeAgent()
```

#### 前端修改

**新增文件：`frontend/src/components/TuringChallenge.tsx`**
```typescript
import { useState } from 'react';

interface TuringChallengeProps {
  onGuess: (guess: string) => void;
  result?: {
    correct: boolean;
    your_guess: string;
    actual: string;
    explanation: string;
  };
}

export default function TuringChallenge({ onGuess, result }: TuringChallengeProps) {
  const [selected, setSelected] = useState<string | null>(null);

  if (result) {
    return (
      <div className="turing-result">
        <h2>{result.correct ? '✅ Correct!' : '❌ Incorrect'}</h2>
        <p>Your guess: {result.your_guess}</p>
        <p>Actual: {result.actual}</p>
        <p>{result.explanation}</p>
      </div>
    );
  }

  return (
    <div className="turing-challenge">
      <h2>Social Turing Challenge</h2>
      <p>Based on your 30-turn conversation, make your guess:</p>

      <div className="options">
        <button
          className={selected === 'Real Person' ? 'selected' : ''}
          onClick={() => setSelected('Real Person')}
        >
          👤 Real Person
        </button>
        <button
          className={selected === 'AI Bot' ? 'selected' : ''}
          onClick={() => setSelected('AI Bot')}
        >
          🤖 AI Bot
        </button>
      </div>

      <button
        disabled={!selected}
        onClick={() => selected && onGuess(selected)}
      >
        Submit Guess
      </button>
    </div>
  );
}
```

**修改文件：`frontend/src/App.tsx`**
```typescript
// 添加新的页面状态
const [page, setPage] = useState<'home' | 'chat' | 'turing' | 'twin-setup' | 'twin-chat' | 'comparison'>('home');
const [turingResult, setTuringResult] = useState<any>(null);

// 在 WebSocket 消息处理中添加
case 'threshold_reached':
  if (data.data && data.data.turn >= 30) {
    setPage('turing');
  }
  break;

case 'turing_result':
  setTuringResult(data.data);
  break;

// 添加图灵测试页面
if (page === 'turing') {
  return (
    <TuringChallenge
      onGuess={(guess) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ action: 'turing_guess', guess }));
        }
      }}
      result={turingResult}
    />
  );
}
```

**修改文件：`src/api/websocket.py`**
```python
# 在 websocket_endpoint 中添加
elif action == "turing_guess":
    guess = message.get("guess")
    orchestrator = session_manager.get_session(user_id)
    if orchestrator:
        result = orchestrator.turing_agent.submit_guess(
            user_id,
            orchestrator.preferred_bot_id,
            guess
        )
        await manager.send_message(user_id, {
            "type": "turing_result",
            "data": result
        })
```

### 方案 2：完善 Digital Twin 对比功能

**修改文件：`src/agents/digital_twin_agent.py`**
```python
def compare_perceptions(
    self,
    friend_guess: dict,
    system_inference: dict
) -> dict:
    """
    对比朋友猜测和系统推断

    Args:
        friend_guess: 朋友的猜测 {gender, age_range, mbti, ...}
        system_inference: 系统推断 {sex, age, mbti, ...}

    Returns:
        {
            overall_match_rate: float,
            matched: int,
            total: int,
            dimension_comparison: [{dim, friend, system, match}],
            mismatch_analysis: str
        }
    """
    comparisons = []
    matched = 0
    total = 0

    # 对比性别
    if 'gender' in friend_guess and 'sex' in system_inference:
        total += 1
        match = (friend_guess['gender'].lower() == system_inference['sex'].lower())
        if match:
            matched += 1
        comparisons.append({
            'dimension': 'Gender',
            'friend': friend_guess['gender'],
            'system': system_inference['sex'],
            'match': match
        })

    # 对比年龄
    if 'age_range' in friend_guess and 'age' in system_inference:
        total += 1
        # 简单匹配逻辑
        match = str(friend_guess['age_range']) in str(system_inference['age'])
        if match:
            matched += 1
        comparisons.append({
            'dimension': 'Age',
            'friend': friend_guess['age_range'],
            'system': system_inference['age'],
            'match': match
        })

    # 对比 MBTI
    if 'mbti' in friend_guess and 'mbti' in system_inference:
        total += 1
        match = (friend_guess['mbti'] == system_inference['mbti'])
        if match:
            matched += 1
        comparisons.append({
            'dimension': 'MBTI',
            'friend': friend_guess['mbti'],
            'system': system_inference['mbti'],
            'match': match
        })

    overall_match_rate = matched / total if total > 0 else 0

    return {
        'overall_match_rate': overall_match_rate,
        'matched': matched,
        'total': total,
        'dimension_comparison': comparisons,
        'mismatch_analysis': f"{matched}/{total} dimensions matched"
    }
```

### 方案 3：对齐三阶段流程

**修改文件：`src/agents/orchestrator.py`**
```python
class OrchestratorAgent:
    def __init__(self, ...):
        # 添加阶段状态
        self.phase = "profiling"  # profiling -> turing -> reflection
        self.turing_agent = TuringChallengeAgent()

    async def process_user_message(self, message: str):
        # 检查阶段转换
        if self.phase == "profiling" and self.ctx.turn_count >= 30:
            self.phase = "turing"
            return {
                "success": True,
                "phase_transition": "turing",
                "message": "30 turns completed! Time for the Turing Challenge."
            }

        # 正常处理消息
        # ...
```

## 测试计划

### 本地测试
```bash
# 1. 启动后端
cd /Users/quinne/soulmatch_agent
uvicorn src.api.main:app --port 7860 --reload

# 2. 启动前端
cd frontend
npm run dev

# 3. 测试流程
# - 访问 http://localhost:5173
# - 选择一个角色开始对话
# - 进行 30 轮对话
# - 验证是否跳转到图灵测试
# - 提交猜测并查看结果
# - 进入数字分身对比阶段
```

### HuggingFace Spaces 部署
```bash
# 1. 构建前端
cd frontend
npm run build

# 2. 测试 Docker 构建
docker build -t soulmatch .
docker run -p 7860:7860 soulmatch

# 3. 推送到 HuggingFace
git add .
git commit -m "Fix frontend-backend mismatch and add Turing Challenge"
git push
```

## 参考资源

- **后端仓库**: https://github.com/YanLin-Quinne/soulmatch_agent
- **前端部署**: https://huggingface.co/spaces/Quinnnnnne/SoulMatch-Agent
- **OpenFactVerification**: https://github.com/Libr-AI/OpenFactVerification
- **EMNLP 论文**: `/Users/quinne/Downloads/EMNLP_march (3).pdf`

## 下一步行动

1. ✅ WebSocket 端口已修复
2. ⏳ 实现 Social Turing Challenge（P1）
3. ⏳ 完善 Digital Twin 对比（P1）
4. ⏳ 对齐三阶段流程（P2）
