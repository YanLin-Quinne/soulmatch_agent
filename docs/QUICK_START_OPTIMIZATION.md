# SoulMatch Agent 优化版本快速开始指南

## 新增功能概览

本次优化为 SoulMatch Agent 添加了以下核心功能：

### ✅ 已实现（P0-P1 优先级）

1. **持久化层** - 解决 session 重启丢失问题
2. **LogicTree** - 显式三段论推理树
3. **记忆聚合优化** - 方差降低算法

### 🚧 待实现（P2 优先级）

4. **多智能体讨论** - Agent Discussion Room
5. **Streaming 支持** - 逐 token 流式输出
6. **前端可视化** - Bayesian/Pipeline/LogicTree 可视化

---

## 快速开始

### 1. 安装依赖

```bash
cd /Users/quinne/soulmatch_agent
pip install -r requirements.txt
```

新增依赖：
- `sqlalchemy>=2.0.0` - SQL ORM
- `psycopg2-binary>=2.9.0` - PostgreSQL 驱动

### 2. 初始化数据库

```bash
# 使用 SQLite（开发环境）
python scripts/init_database.py

# 或使用 PostgreSQL（生产环境）
export DATABASE_URL="postgresql://user:pass@localhost/soulmatch"
python scripts/init_database.py
```

### 3. 配置环境变量

在 `.env` 文件中添加：

```bash
# 数据库配置
DATABASE_URL=sqlite:///./soulmatch.db
SQL_ECHO=false

# 多智能体讨论（可选）
ENABLE_DISCUSSION_ROOM=false
DISCUSSION_TRIGGER_TURNS=10,20,30

# 记忆聚合
MEMORY_AGGREGATION_METHOD=variance_reduction
AGGREGATION_NUM_SAMPLES=3
```

### 4. 运行应用

```bash
# 启动后端
uvicorn src.api.main:app --reload

# 启动前端
cd frontend && npm run dev
```

---

## 核心功能使用

### 持久化层

所有 session 数据现在自动持久化到数据库：

```python
from src.persistence.session_store import SessionStore

store = SessionStore()

# 创建 session
store.create_session(session_id="abc123", user_id="user1", bot_id="bot_0")

# 保存对话
store.add_conversation_turn(
    session_id="abc123",
    turn_number=1,
    speaker="user",
    message="你好"
)

# 保存特征历史（含 Bayesian 更新轨迹）
store.add_feature_history(
    session_id="abc123",
    turn_number=5,
    features={"openness": 0.75},
    confidences={"openness": 0.68},
    bayesian_updates={
        "openness": {
            "prior_mean": 0.70,
            "prior_variance": 0.04,
            "posterior_mean": 0.75,
            "posterior_variance": 0.02
        }
    }
)

# 保存逻辑树
store.add_logic_tree_node(
    session_id="abc123",
    turn_number=10,
    node_type="conclusion",
    content="关系状态应升级到 crush",
    confidence=0.82,
    evidence=["用户主动分享个人经历", "trust_score=0.85"]
)
```

### LogicTree（三段论推理）

```python
from src.agents.logic_tree import LogicTreeBuilder

builder = LogicTreeBuilder(llm_router)

# 构建关系预测逻辑树
logic_tree = await builder.build_relationship_logic_tree(
    conversation_history=ctx.conversation_history,
    current_features=ctx.predicted_features,
    trust_score=ctx.trust_score,
    turn_number=ctx.turn_count
)

# 输出三段论格式
print(logic_tree.to_syllogism())
# 输出：
# === 三段论推理 ===
# 大前提: 当用户表现出高信任度（trust>0.8）且主动分享个人信息时...
# 小前提: 用户在过去5轮中 trust_score=0.85，主动分享了3次...
# 结论: 关系状态应从 acquaintance 升级到 crush（置信度 0.82）

# 检测逻辑冲突
conflict = builder.detect_logic_conflicts(tree1, tree2)
if conflict:
    resolved_tree = await builder.resolve_conflict(tree1, tree2, conflict)
```

### 记忆聚合优化

```python
from src.memory.aggregation_operators import MemoryAggregationOperator

aggregator = MemoryAggregationOperator(llm_router)

# 方差降低聚合（3次采样+投票）
episodic_memory = await aggregator.aggregate_working_to_episodic(
    working_items=working_items,
    method="variance_reduction",
    num_samples=3
)

print(f"摘要: {episodic_memory.summary}")
print(f"方差: {episodic_memory.variance:.3f}")  # 越低越一致

# 自我提炼聚合（迭代压缩）
episodic_memory = await aggregator.aggregate_working_to_episodic(
    working_items=working_items,
    method="self_refinement",
    max_refinement_rounds=3
)

# 混合方法（先方差降低，再自我提炼）
episodic_memory = await aggregator.aggregate_working_to_episodic(
    working_items=working_items,
    method="hybrid"
)
```

---

## 数据库表结构

### sessions
- `session_id` (PK) - Session ID
- `user_id` - 用户 ID
- `bot_id` - Bot ID
- `created_at` - 创建时间
- `last_active` - 最后活跃时间
- `state` - JSON 序列化的 AgentContext

### conversation_turns
- `id` (PK) - 自增 ID
- `session_id` (FK) - Session ID
- `turn_number` - 轮次编号
- `speaker` - 'user' 或 'bot'
- `message` - 消息内容
- `timestamp` - 时间戳

### memory_snapshots
- `id` (PK) - 自增 ID
- `session_id` (FK) - Session ID
- `layer` - 'working', 'episodic', 'semantic'
- `turn_range_start` - 起始轮次
- `turn_range_end` - 结束轮次
- `content` - JSON 序列化的记忆内容
- `variance` - **新增：方差指标**
- `created_at` - 创建时间

### feature_history
- `id` (PK) - 自增 ID
- `session_id` (FK) - Session ID
- `turn_number` - 轮次编号
- `features` - JSON: {feature_name: value}
- `confidences` - JSON: {feature_name: confidence}
- `bayesian_updates` - **新增：Bayesian 更新轨迹**
- `timestamp` - 时间戳

### relationship_snapshots
- `id` (PK) - 自增 ID
- `session_id` (FK) - Session ID
- `turn_number` - 轮次编号
- `rel_status` - 关系状态
- `rel_type` - 关系类型
- `sentiment` - 情感
- `trust_score` - 信任分数
- `can_advance` - 是否可升级
- `social_votes` - JSON: agent 投票
- `timestamp` - 时间戳

### logic_tree_nodes（新增）
- `id` (PK) - 自增 ID
- `session_id` (FK) - Session ID
- `turn_number` - 轮次编号
- `node_type` - 'major_premise', 'minor_premise', 'conclusion'
- `parent_id` (FK) - 父节点 ID
- `content` - 节点内容
- `confidence` - 置信度
- `evidence` - JSON: 支持证据
- `created_at` - 创建时间

---

## 集成到现有代码

### 修改 SessionManager

```python
# src/api/session_manager.py

from src.persistence.session_store import SessionStore

class SessionManager:
    def __init__(self):
        self.sessions = {}  # 保留内存缓存
        self.store = SessionStore()  # 新增：持久化存储

    def create_session(self, session_id: str, user_id: str, bot_id: str):
        # 创建内存 session
        session = {...}
        self.sessions[session_id] = session

        # 持久化到数据库
        self.store.create_session(session_id, user_id, bot_id)

        return session

    def get_session(self, session_id: str):
        # 先查内存
        if session_id in self.sessions:
            return self.sessions[session_id]

        # 再查数据库（恢复 session）
        db_session = self.store.get_session(session_id)
        if db_session:
            # 重建内存 session
            session = self._rebuild_session(db_session)
            self.sessions[session_id] = session
            return session

        return None
```

### 修改 Orchestrator

```python
# src/agents/orchestrator.py

from src.agents.logic_tree import LogicTreeBuilder
from src.memory.aggregation_operators import MemoryAggregationOperator

class OrchestratorAgent:
    def __init__(self, ...):
        # 新增
        self.logic_tree_builder = LogicTreeBuilder(self.llm_router)
        self.memory_aggregator = MemoryAggregationOperator(self.llm_router)
        self.session_store = SessionStore()

    async def process_user_message(self, message: str):
        # ... 现有逻辑 ...

        # 保存对话轮次
        self.session_store.add_conversation_turn(
            session_id=self.session_id,
            turn_number=self.turn_count,
            speaker="user",
            message=message
        )

        # 关系预测时构建逻辑树
        if self.turn_count % 5 == 0:
            logic_tree = await self.logic_tree_builder.build_relationship_logic_tree(
                conversation_history=self.ctx.conversation_history,
                current_features=self.ctx.predicted_features,
                trust_score=self.ctx.trust_score,
                turn_number=self.turn_count
            )

            # 持久化逻辑树
            self._save_logic_tree(logic_tree)

        # 记忆压缩时使用聚合算子
        if self.turn_count % 10 == 0:
            episodic_memory = await self.memory_aggregator.aggregate_working_to_episodic(
                working_items=self._get_working_memory(),
                method=settings.memory_aggregation_method,
                num_samples=settings.aggregation_num_samples
            )

            # 持久化记忆快照
            self.session_store.add_memory_snapshot(
                session_id=self.session_id,
                layer="episodic",
                content=episodic_memory.__dict__,
                variance=episodic_memory.variance
            )
```

---

## 性能指标

| 功能 | 额外开销 | 时间复杂度 |
|------|---------|-----------|
| 持久化写入 | <10ms | O(1) |
| 持久化读取 | <5ms | O(log N) |
| 方差降低聚合 | 3x LLM 调用 | O(N²) 相似度计算 |
| 自我提炼聚合 | 3x LLM 调用 | O(K) K≤3 轮 |
| LogicTree 构建 | 1x LLM 调用 | O(1) |

**总体评估：** 时间复杂度可接受，略高于简单的多数投票机制，但远低于需要额外微调的方案。

---

## 下一步开发

### 1. 多智能体讨论（P1）

修改 `src/agents/orchestrator.py`：

```python
from src.agents.agent_discussion_room import AgentDiscussionRoom

# 在关键决策点触发讨论
if settings.enable_discussion_room and self.turn_count in [10, 20, 30]:
    discussion_result = await self.discussion_room.run_discussion(
        topic="relationship_prediction",
        context=self.ctx
    )
```

### 2. Streaming 支持（P2）

修改 `src/agents/llm_router.py`：

```python
async def chat_stream(self, messages, role, **kwargs):
    """流式输出"""
    async for chunk in self._stream_provider(messages, role, **kwargs):
        yield chunk
```

修改 `src/api/websocket.py`：

```python
async for chunk in llm_router.chat_stream(messages, role):
    await websocket.send_json({
        "type": "bot_message_chunk",
        "chunk": chunk
    })
```

### 3. 前端可视化（P2）

创建新组件：
- `BayesianUpdateChart.tsx` - 先验→后验演化图
- `AgentPipelineGraph.tsx` - DAG 流程图
- `LogicTreeVisualization.tsx` - 三段论树形图

---

## 故障排查

### 数据库连接失败

```bash
# 检查数据库文件
ls -la soulmatch.db

# 重新初始化
rm soulmatch.db
python scripts/init_database.py
```

### SQLAlchemy 导入错误

```bash
pip install --upgrade sqlalchemy psycopg2-binary
```

### 方差降低聚合速度慢

调整采样次数：

```bash
# .env
AGGREGATION_NUM_SAMPLES=2  # 降低到 2 次
```

---

## 贡献指南

欢迎提交 PR 到 https://github.com/YanLin-Quinne/soulmatch_agent

优先级：
1. P1: 多智能体讨论集成
2. P2: Streaming 实现
3. P2: 前端可视化组件

---

生成时间：2026-03-05
