# AI YOU Agent 核心层优化报告

> 基于 Claude Code 源码分析（claude-code-analysis / claw-code / claw-code-parity）的五个核心架构模式，对 AI-YOU-TEST 进行系统性优化。

## 优化总览

| # | 方向 | 来源模式 | 复杂度 | 新增文件 | 修改文件 |
|---|------|---------|--------|---------|---------|
| 1 | 工具注册表 + 权限门控 | claw-code `GlobalToolRegistry` | S | 0 | 3 |
| 2 | Agent 生命周期 Hooks | claw-code plugins `PreToolUse/PostToolUse` | M | 2 | 1 |
| 3 | 会话压缩 (Compaction) | claude-code `tokenBudget` + `compact_after_turns` | M | 0 | 3 |
| 4 | 引导图 (Bootstrap Graph) | claw-code-parity 7-stage `bootstrap_graph` | M | 1 | 2 |
| 5 | Provider 抽象 + 流式输出 | claw-code API crate `ProviderClient` + SSE | L | 5 | 2 |

**总计**：新增 ~650 行代码（8 个新文件），修改 ~250 行（7 个现有文件），净增 ~850 行。

---

## Direction 1: 工具注册表 + 权限门控

### 问题
原有工具系统 (`src/agents/tools/`) 是简单的 name→handler 字典，无权限控制。任何工具可被任意调用，模块级 `_tool_context` dict 是隐式共享状态。

### 解决方案

**`src/agents/tools/registry.py`**
- 新增 `PermissionLevel(IntEnum)`，四级权限：
  - `READ_ONLY (0)` — 只读，无副作用
  - `CONTEXT_WRITE (1)` — 写入 AgentContext
  - `EXTERNAL_CALL (2)` — 调用外部 API/网络
  - `DANGEROUS (3)` — 高风险操作
- 新增 `ToolContext` dataclass，替代模块级 dict（过渡期两者共存）
- `Tool` dataclass 新增 `permission` 字段
- `ToolRegistry.list_tools()` / `get_schemas()` 支持 `max_permission` 过滤

**`src/agents/tools/executor.py`**
- `ToolExecutor` 新增 `allowed_permission` 参数
- `_execute_tool_sync()` 执行前检查 `tool.permission <= self.allowed_permission`

**`src/agents/tools/builtin.py`**
- 8 个内置工具标注权限级别：
  - READ_ONLY: `get_current_time`, `get_conversation_stats`, `recall_memory`, `lookup_user_profile`
  - CONTEXT_WRITE: `get_dating_advice`, `analyze_compatibility`
  - EXTERNAL_CALL: `search_web`, `get_weather`

### 使用示例
```python
# 限制 executor 只允许 READ_ONLY 和 CONTEXT_WRITE 工具
executor = ToolExecutor(allowed_permission=PermissionLevel.CONTEXT_WRITE)
# search_web (EXTERNAL_CALL) 会被拒绝
```

---

## Direction 2: Agent 生命周期 Hooks

### 问题
Orchestrator 的 DAG pipeline 缺少注入点，无法在 agent 执行前后插入监控、日志、安全检查等横切关注点。

### 解决方案

**新增 `src/agents/hooks.py`** (89 行)
- `HookAction` 枚举: `ALLOW`, `DENY`, `MODIFY`
- `HookResult` dataclass: action + reason
- `Hook` Protocol: `on_pre_agent()` / `on_post_agent()`
- `HookRegistry`: 注册/注销 hook，run_pre_hooks 短路 DENY
- `HookDeniedError`: hook 拒绝时抛出，被 orchestrator 优雅捕获

**新增 `src/agents/hooks_builtin.py`** (93 行)
- `LoggingHook` — 记录每个 agent phase 的名称、轮次、耗时
- `CostTrackingHook` — 追踪每个 phase 的 token 消耗和成本 delta
- `ScamRiskGateHook` — 当 `scam_risk_score >= 0.8` 时阻止 bot_response

**修改 `src/agents/orchestrator.py`**
- 新增 `_run_with_hooks(agent_name, fn)` 包装方法
- 4 个 pipeline phase 被 hook 包装: `feature_update`, `question_strategy`, `bot_response`, `memory_store`
- bot_response 被 DENY 时返回安全降级消息
- response 中附带 `agent_costs` 报告

### Hook 执行流程
```
Pre-hooks (顺序执行)
  ├─ LoggingHook: 记录开始时间
  ├─ CostTrackingHook: 快照当前 token 计数
  └─ ScamRiskGateHook: 检查风险分数 → DENY?
       ↓
Agent Phase 执行
       ↓
Post-hooks (顺序执行)
  ├─ LoggingHook: 记录耗时
  └─ CostTrackingHook: 计算 delta
```

---

## Direction 3: 会话压缩 (Session Compaction)

### 问题
`AgentContext.conversation_history` 保留最近 50 条原始消息，长对话会超出 LLM 上下文窗口。三层记忆已有 episodic 摘要，但从未回注到对话上下文中。

### 解决方案

**`src/config.py`**
- 新增 `compact_after_turns: int = 20`
- 新增 `context_token_budget: int = 8000`

**`src/agents/agent_context.py`**
- 新增字段: `compacted_summary`, `compacted_up_to_turn`
- `history_as_messages()` 修改: 若有压缩摘要，在消息列表开头注入 `[Previous conversation summary]` 前缀

**`src/agents/orchestrator.py`**
- 新增 `_compact_context_if_needed()` 方法
- 在每轮 `process_user_message()` 开头调用
- 逻辑:
  1. turn < 20 → 跳过
  2. 距上次压缩 < 10 轮 → 跳过
  3. 从 `three_layer_memory.episodic_memory` 取已有摘要（**零额外 LLM 调用**）
  4. 拼接为 `compacted_summary`
  5. 裁剪 `conversation_history` 到最近 10 条

### 关键设计
- **不做额外 LLM 调用** — 直接复用三层记忆已有的 episodic 摘要
- 压缩摘要通过 `history_as_messages()` 自动注入 LLM 上下文
- 若三层记忆未启用，退回原有截断行为（优雅降级）

---

## Direction 4: 引导图 (Bootstrap Graph)

### 问题
`app_gradio.py` 和 `src/api/main.py` 各自有 ad-hoc 初始化代码（手动创建 SessionManager、加载 personas、注册 tools），逻辑重复且无错误传播。

### 解决方案

**新增 `src/bootstrap.py`** (200 行)
- `BootstrapStage` 枚举: CONFIG → DATABASE → TOOLS → PERSONAS → SESSION_MGR → BACKGROUND → READY
- `BootstrapGraph` 类:
  - `register_stage(stage, fn, depends_on=[])` — 注册阶段
  - `run_all()` — 按序执行，依赖失败的 stage 自动跳过
  - `is_ready()` / `get_results()` — 状态查询
- `create_default_bootstrap()` 工厂函数 — 注册所有默认阶段
- `get_session_manager()` / `get_bot_pool()` — 全局访问已初始化的单例

**修改 `app_gradio.py`** — 替换手动初始化为:
```python
_bootstrap = create_default_bootstrap()
asyncio.get_event_loop().run_until_complete(_bootstrap.run_all())
sm = get_session_manager()
```

**修改 `src/api/main.py`** — lifespan startup 改为:
```python
bootstrap = create_default_bootstrap()
results = await bootstrap.run_all()
app.state.bootstrap = bootstrap
```
- `/health` 端点增加 bootstrap stage 状态报告

### 阶段依赖图
```
CONFIG (0) ──┬── DATABASE (1)
             ├── TOOLS (2)        # 与 PERSONAS 无依赖
             └── PERSONAS (3)
                    └── SESSION_MGR (4)
                           └── BACKGROUND (5)
                                  └── READY (6)
```

---

## Direction 5: Provider 抽象 + 流式输出

### 问题
`llm_router.py` 中的 provider 调用是散落的独立函数（`_call_anthropic`, `_call_openai_compat`, `_call_gemini`, `_call_hf`），无统一接口。不支持 streaming，PersonaAgent 只能返回完整文本。

### 解决方案

**新增 `src/agents/providers/` 包** (5 个文件, 266 行)

`base.py` — 抽象接口:
```python
class ProviderClient(ABC):
    def chat_sync(model_id, system, messages, ...) -> (text, in_tokens, out_tokens)
    async def chat_stream(model_id, system, messages, ...) -> AsyncIterator[StreamEvent]
    def is_available() -> bool

@dataclass
class StreamEvent:
    event_type: str  # "text_delta" | "usage" | "done"
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
```

`anthropic_provider.py` — `client.messages.stream()` 上下文管理器
`openai_provider.py` — `stream=True`，统一 OpenAI/DeepSeek/Qwen/Local
`gemini_provider.py` — `generate_content_stream()`

**修改 `src/agents/llm_router.py`** (+113 行)
- `UsageRecord` 新增 `per_role` 字段 — 按 agent role 聚合成本
- `LLMRouter` 新增:
  - `_providers: dict[Provider, ProviderClient]` — 懒创建缓存
  - `_get_provider(provider)` — 工厂方法
  - `stream_chat(role, system, messages, ...)` — async generator，支持 fallback 链
- 现有同步 `chat()` 方法**完全不变**

**修改 `src/agents/persona_agent.py`** (+23 行)
- 新增 `generate_response_stream(message, ctx)` — async generator
- 内部调用 `router.stream_chat(role=PERSONA, ...)`
- 流式收集完整文本后更新对话历史

### 流式调用示例
```python
# PersonaAgent streaming
async for chunk in bot.generate_response_stream(message, ctx):
    await websocket.send_text(chunk)  # 实时打字效果
```

---

## 文件变更汇总

### 新增文件 (8 个, ~650 行)
| 文件 | 行数 | 用途 |
|------|------|------|
| `src/agents/hooks.py` | 89 | Hook 协议和注册表 |
| `src/agents/hooks_builtin.py` | 93 | 3 个内置 Hook |
| `src/bootstrap.py` | 200 | 引导图框架 |
| `src/agents/providers/__init__.py` | 5 | Provider 包 |
| `src/agents/providers/base.py` | 49 | 抽象接口 + StreamEvent |
| `src/agents/providers/anthropic_provider.py` | 68 | Anthropic 实现 |
| `src/agents/providers/openai_provider.py` | 75 | OpenAI-compatible 实现 |
| `src/agents/providers/gemini_provider.py` | 69 | Gemini 实现 |

### 修改文件 (10 个, ~250 行变更)
| 文件 | 变更 | 方向 |
|------|------|------|
| `src/agents/tools/registry.py` | +PermissionLevel, +ToolContext, +permission 字段, +max_permission 过滤 | D1 |
| `src/agents/tools/executor.py` | +allowed_permission, +权限检查 | D1 |
| `src/agents/tools/builtin.py` | +权限标注, +ToolContext 对象 | D1 |
| `src/agents/orchestrator.py` | +hook 集成, +_run_with_hooks, +_compact_context_if_needed | D2, D3 |
| `src/agents/agent_context.py` | +compacted_summary, +history_as_messages 摘要注入 | D3 |
| `src/config.py` | +compact_after_turns, +context_token_budget | D3 |
| `app_gradio.py` | 替换手动初始化为 bootstrap | D4 |
| `src/api/main.py` | 替换 lifespan 为 bootstrap, /health 增加状态 | D4 |
| `src/agents/llm_router.py` | +_providers 缓存, +_get_provider, +stream_chat, +per_role 成本 | D5 |
| `src/agents/persona_agent.py` | +generate_response_stream | D5 |

---

## 架构对比

### 优化前
```
app_gradio.py ──(ad-hoc init)──→ SessionManager → OrchestratorAgent
                                                   ├── tools: 无权限
                                                   ├── pipeline: 无 hook
                                                   ├── context: 50 条硬截断
                                                   └── LLM: 同步 only
```

### 优化后
```
app_gradio.py ──→ BootstrapGraph ──→ 7 阶段有序初始化
                                          ↓
                                   OrchestratorAgent
                                    ├── tools: PermissionLevel 门控
                                    ├── pipeline: Hook 管道 (pre/post)
                                    │   ├── LoggingHook (计时)
                                    │   ├── CostTrackingHook (成本)
                                    │   └── ScamRiskGateHook (安全)
                                    ├── context: episodic 摘要压缩
                                    └── LLM Router
                                         ├── chat() 同步 (不变)
                                         ├── stream_chat() 异步流式
                                         └── ProviderClient 抽象层
                                              ├── AnthropicProvider
                                              ├── OpenAICompatProvider
                                              └── GeminiProvider
```

---

## Bonus: ACL Demo 级前端重构

### 问题
原 Gradio UI (`app_gradio.py`) 仅展示 Big Five + MBTI 两个数据维度，未利用后端暴露的 14+ 维度丰富数据。

### 解决方案

**新增 `app_demo.py`** (833 行) — ACL/EMNLP best demo paper 级别的专业 UI

**布局：** 双栏 — 左 40% 聊天 / 右 60% 仪表盘

**6 个实时 Dashboard Tab:**

| Tab | 可视化内容 | 图表类型 |
|-----|-----------|---------|
| 🧬 Personality | Big Five 雷达图 + 置信度叠加 + MBTI/沟通风格/准确率 | Plotly Scatterpolar |
| 💭 Emotion | 情绪效价时间线 + 彩色标记 + 置信度/强度条 + 情感趋势 | Plotly Scatter |
| ❤️ Relationship | 5 阶段进度条 (发光效果) + 类型/情感/能否推进徽章 | HTML + CSS |
| 🧠 Memory | 三层记忆卡片 (工作/情节/语义) + 计数和描述 | HTML Cards |
| ⚙️ Pipeline | Agent 阶段 token 消耗柱状图 (input vs output) | Plotly Bar |
| 📋 Raw Data | 完整 response dict 供研究检查 | gr.JSON |

**设计特点：**
- 暗色专业主题，主色 `#6C63FF`
- Persona 选择卡片网格 (8 角色，4/行)
- 每条消息后全部 dashboard 实时刷新
- 后端不可用时优雅降级 (fallback personas + demo mode)
- 支持 "← Change Partner" 返回角色选择

**启动方式：**
```bash
python app_demo.py
# 或
gradio app_demo.py
```

---

## 向后兼容性

所有改动保持完全向后兼容：
- `ToolExecutor()` 不传 `allowed_permission` 时不做检查
- `ToolRegistry.list_tools()` 不传 `max_permission` 时返回全部
- `router.chat()` 同步接口零改动
- `PersonaAgent.generate_response()` 零改动
- `_tool_context` dict 和 `ToolContext` 对象双轨共存
- Bootstrap 失败时优雅降级，不阻塞启动
