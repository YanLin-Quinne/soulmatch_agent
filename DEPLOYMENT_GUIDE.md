# SoulMatch Agent 部署指南

## 升级完成清单

### ✅ 后端核心升级
1. **SQLite持久化** - `src/api/session_manager.py`
   - 添加SessionStore类,使用SQLite存储session数据
   - 数据库路径: `./data/sessions.db`
   - 支持session跨重启恢复

2. **新增API端点** - `src/api/main.py`
   - `POST /api/profiling/start` - 开始画像模式(30轮推断)
   - `POST /api/profiling/message` - 发送消息获取推断
   - `GET /api/profiling/inference/{session_id}` - 获取推断结果
   - `POST /api/playground/start` - 开始推理游戏(10轮猜人)
   - `POST /api/playground/guess` - 提交猜测
   - `GET /api/pipeline/status/{session_id}` - Pipeline可视化
   - `GET /api/logic-tree/{session_id}` - 逻辑树数据

3. **启用Discussion Room** - `src/config.py`
   - `enable_discussion_room = True`
   - 激活真正的多智能体协商机制

### ✅ 前端开发
1. **15人物数据** - `frontend/src/data/personas.ts`
   - 10个AI人物 + 5个真人
   - 包含详细profile、personality、interests

2. **核心组件** - `frontend/src/components/`
   - `PersonaGrid.tsx` - 15人物卡片网格
   - `ProfilingMode.tsx` - 30句对话推断模式
   - `PlaygroundMode.tsx` - 10句猜人游戏

### ✅ 部署配置
1. **Dockerfile** - 已配置FastAPI启动
2. **环境变量** - `ENABLE_DISCUSSION_ROOM=true`

## HuggingFace Spaces 部署步骤

### 1. 配置API Keys
在HF Spaces Settings中添加以下Secrets:
```bash
ANTHROPIC_API_KEY=your_claude_key
OPENAI_API_KEY=your_gpt_key
GOOGLE_API_KEY=your_gemini_key
QWEN_API_KEY=your_qwen_key
DEEPSEEK_API_KEY=your_deepseek_key
```

### 2. 配置持久化存储
在HF Spaces Settings中:
- 申请Persistent Storage (免费版5GB或Pro版50GB)
- 挂载路径: `/data`

### 3. 部署命令
```bash
cd /Users/quinne/soulmatch_agent
git add .
git commit -m "Upgrade to FastAPI+React with multi-agent discussion"
git push hf main
```

### 4. 验证部署
访问: `https://huggingface.co/spaces/YOUR_USERNAME/soulmatch-agent`

测试端点:
- `GET /health` - 健康检查
- `GET /api` - API信息
- `GET /docs` - Swagger文档

## 本地开发

### 启动后端
```bash
cd /Users/quinne/soulmatch_agent
uvicorn src.api.main:app --reload --port 8000
```

### 启动前端
```bash
cd frontend
npm install
npm run dev
```

### 构建前端
```bash
cd frontend
npm run build
```

## API使用示例

### 画像模式 (Profiling Mode)
```bash
# 1. 开始session
curl -X POST http://localhost:8000/api/profiling/start \
  -H "Content-Type: application/json" \
  -d '{"persona_id": 0}'

# 2. 发送消息
curl -X POST http://localhost:8000/api/profiling/message \
  -H "Content-Type: application/json" \
  -d '{"session_id": "xxx", "message": "Hi, how are you?"}'

# 3. 获取推断
curl http://localhost:8000/api/profiling/inference/xxx
```

### 推理模式 (Playground Mode)
```bash
# 1. 开始游戏
curl -X POST http://localhost:8000/api/playground/start \
  -H "Content-Type: application/json" \
  -d '{"user_id": "player1"}'

# 2. 提交猜测
curl -X POST http://localhost:8000/api/playground/guess \
  -H "Content-Type: application/json" \
  -d '{"session_id": "xxx", "guess_persona_id": 5}'
```

## 核心特性

### 1. 多智能体协同
- 12个agent的DAG pipeline
- 动态讨论和投票机制
- 实时Pipeline可视化

### 2. Bayesian推断
- 30轮对话逐步推断用户特征
- 实时置信度更新
- Conformal Prediction不确定性量化

### 3. 持久化存储
- SQLite存储session数据
- ChromaDB存储记忆向量
- 支持跨session用户研究

### 4. 智能LLM路由
- 5个模型自动选择
- 任务类型匹配
- 失败自动fallback

## 故障排查

### 问题1: API Keys未配置
**症状**: 500错误,日志显示"API key not found"
**解决**: 在HF Spaces Settings添加对应的Secret

### 问题2: 持久化存储未挂载
**症状**: 重启后session丢失
**解决**: 在HF Spaces Settings启用Persistent Storage

### 问题3: 前端未构建
**症状**: 访问根路径显示"Frontend not built"
**解决**: 
```bash
cd frontend && npm install && npm run build
```

### 问题4: Discussion Room未启用
**症状**: Pipeline显示固定顺序
**解决**: 检查`src/config.py`中`enable_discussion_room = True`

## 性能指标

- 单次对话响应: < 3秒
- 并发session: 10+
- 单session内存: < 100MB
- API调用成本: 优先使用Qwen/DeepSeek

## 下一步优化

1. 添加可视化组件:
   - PipelineVisualizer.tsx (D3.js)
   - LogicTreeVisualizer.tsx
   - BayesianUpdater.tsx

2. 实现AI分身系统:
   - CloneSetup.tsx
   - CloneChat.tsx
   - ComparisonView.tsx

3. 增强推断能力:
   - 三段论逻辑推理
   - 跨session用户画像聚合
   - 长期记忆演化

## 联系方式

- GitHub: https://github.com/YOUR_USERNAME/soulmatch_agent
- HF Space: https://huggingface.co/spaces/YOUR_USERNAME/soulmatch-agent
