# SoulMatch Agent 前后端不匹配诊断报告

## 问题总结

### 1. WebSocket 端口不匹配 ✅ 已修复
- **问题**：前端连接 `ws://localhost:8000`，后端运行在 `7860`
- **修复**：App.tsx:13 已改为 `ws://localhost:7860`

### 2. 与 EMNLP 论文的架构差异

#### 论文要求的三阶段流程：
1. **Progressive Profiling (30轮推断)** - ✅ 已实现
2. **Social Turing Challenge (猜测真人/AI)** - ❌ **完全缺失**
3. **Digital Twin Reflection (朋友猜测 vs 系统推断)** - ⚠️ 部分实现

#### 当前实现状态分析：

**后端实现（已有）：**
- ✅ `src/agents/orchestrator.py` - 12个协作 agent
- ✅ `src/agents/conversation_sentiment_agent.py` - 对话情感分析
- ✅ `src/agents/digital_twin_agent.py` - 数字分身创建
- ✅ `src/agents/relationship_prediction_agent.py` - 关系预测
- ✅ `src/api/websocket.py` - WebSocket 协议支持
- ✅ 三层记忆架构（Working/Episodic/Semantic）
- ✅ Bayesian 特征更新 + Conformal Prediction

**前端实现（已有）：**
- ✅ `frontend/src/App.tsx` - 完整的聊天界面
- ✅ `frontend/src/components/DigitalTwinSetup.tsx` - 数字分身设置
- ✅ `frontend/src/components/ComparisonView.tsx` - 对比视图
- ✅ `frontend/src/components/RelationshipTab.tsx` - 关系预测展示
- ✅ 实时特征推断可视化
- ✅ Conformal Prediction 展示

**缺失的功能：**
- ❌ **Social Turing Challenge 后端 API**
  - 缺少 `/api/turing/start` 端点
  - 缺少 `/api/turing/guess` 端点
  - 缺少真人/AI 标签管理

- ❌ **Social Turing Challenge 前端界面**
  - 缺少猜测界面组件
  - 缺少结果展示逻辑

- ⚠️ **Digital Twin 对比功能不完整**
  - 前端有 `ComparisonView.tsx` 但后端缺少完整的对比计算
  - 缺少维度匹配度算法
  - 缺少可视化数据格式

### 3. 与 OpenFactVerification 的对比

**OpenFactVerification 的优势：**
- 清晰的模块化设计（factcheck/ 核心逻辑分离）
- 统一的 API 接口模式（check_response 作为统一入口）
- 完整的前后端集成（webapp.py + templates/）
- YAML 配置管理（api_config.yaml）

**SoulMatch 可以借鉴的地方：**
- 统一的 orchestrator 入口（已有，但可以优化）
- 配置文件管理（当前硬编码较多）
- 更清晰的 API 文档

## 修复优先级

### P0 - 紧急（已完成）
- [x] WebSocket 端口修复

### P1 - 高优先级（核心功能缺失）
- [ ] 实现 Social Turing Challenge 后端 API
- [ ] 实现 Social Turing Challenge 前端界面
- [ ] 完善 Digital Twin 对比算法

### P2 - 中优先级（体验优化）
- [ ] 对齐论文三阶段流程
- [ ] 添加阶段转换逻辑
- [ ] 优化前端状态管理

### P3 - 低优先级（代码质量）
- [ ] 重构配置管理
- [ ] 添加 API 文档
- [ ] 优化错误处理

## 具体修复方案

### 1. 实现 Social Turing Challenge

#### 后端修改：

**新增文件：`src/agents/turing_challenge_agent.py`**
```python
class TuringChallengeAgent:
    """管理 Social Turing Test 逻辑"""

    def start_challenge(self, user_id: str) -> dict:
        """开始图灵测试，随机分配真人/AI角色"""
        pass

    def submit_guess(self, user_id: str, guess: str) -> dict:
        """提交猜测，返回正确答案和得分"""
        pass
```

**修改文件：`src/api/main.py`**
```python
@app.post("/api/turing/start")
async def start_turing_challenge(user_id: str):
    """开始图灵测试"""
    pass

@app.post("/api/turing/guess")
async def submit_turing_guess(user_id: str, guess: str):
    """提交猜测"""
    pass
```

#### 前端修改：

**新增文件：`frontend/src/components/TuringChallenge.tsx`**
```typescript
export default function TuringChallenge({ onGuess }) {
  // 猜测界面：真人 vs AI 选择
  // 显示对话历史回顾
  // 提交猜测并显示结果
}
```

**修改文件：`frontend/src/App.tsx`**
- 添加 `turing-challenge` 页面状态
- 在 30 轮后自动跳转到图灵测试
- 处理 WebSocket 的 `turing_result` 消息

### 2. 完善 Digital Twin 对比功能

#### 后端修改：

**修改文件：`src/agents/digital_twin_agent.py`**
```python
def compare_perceptions(
    self,
    friend_guess: dict,
    system_inference: dict
) -> dict:
    """
    对比朋友猜测和系统推断

    返回：
    - overall_match_rate: 总体匹配度
    - dimension_comparison: 各维度对比
    - mismatch_analysis: 不匹配分析
    """
    pass
```

#### 前端修改：

**修改文件：`frontend/src/components/ComparisonView.tsx`**
- 添加维度匹配度可视化
- 添加雷达图对比
- 添加不匹配维度高亮

### 3. 对齐论文三阶段流程

**修改文件：`src/agents/orchestrator.py`**
```python
class OrchestratorAgent:
    def __init__(self, ...):
        self.phase = "profiling"  # profiling -> turing -> reflection

    async def process_user_message(self, message: str):
        if self.phase == "profiling" and self.ctx.turn_count >= 30:
            self.phase = "turing"
            return {"phase_transition": "turing"}
        # ...
```

## 测试计划

### 1. 端到端测试
- [ ] 本地启动后端：`uvicorn src.api.main:app --port 7860`
- [ ] 本地启动前端：`cd frontend && npm run dev`
- [ ] 测试 WebSocket 连接
- [ ] 测试 30 轮对话流程
- [ ] 测试图灵测试流程
- [ ] 测试数字分身对比

### 2. HuggingFace Spaces 部署测试
- [ ] 构建 Docker 镜像
- [ ] 测试 wss:// 连接
- [ ] 测试完整三阶段流程

## 参考资源

- **OpenFactVerification**: https://github.com/Libr-AI/OpenFactVerification
- **EMNLP 论文**: `/Users/quinne/Downloads/EMNLP_march (3).pdf`
- **后端仓库**: https://github.com/YanLin-Quinne/soulmatch_agent
- **前端部署**: https://huggingface.co/spaces/Quinnnnnne/SoulMatch-Agent

## 下一步行动

1. **立即执行**：实现 Social Turing Challenge（P1）
2. **短期目标**：完善 Digital Twin 对比（P1）
3. **中期目标**：对齐三阶段流程（P2）
4. **长期优化**：代码重构和文档完善（P3）
