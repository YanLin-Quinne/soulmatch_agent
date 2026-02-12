# SoulMatch Development Guide

## 快速开始

### 1. 环境设置

```bash
# 克隆项目
cd /Users/quinne/SoulMatch

# 创建Python虚拟环境
python3.12 -m venv venv
source venv/bin/activate

# 安装Python依赖
pip install -r requirements.txt

# 安装前端依赖
cd frontend
npm install
cd ..
```

### 2. 配置环境变量

创建 `.env` 文件：

```bash
cp .env.example .env
```

编辑 `.env`，填入必要的API密钥：

```env
# LLM API Keys (至少配置一个)
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_API_KEY=sk-xxxxx

# Kaggle (用于下载OkCupid数据集)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key

# 其他配置使用默认值即可
```

### 3. 数据准备

```bash
# 下载OkCupid数据集
python scripts/download_okcupid_data.py

# 预处理数据并生成Bot personas（需要LLM API）
python scripts/preprocess_data.py
```

**注意**：`preprocess_data.py` 会调用Claude/GPT API从essay文本提取特征，可能需要几分钟并产生API费用。

### 4. 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_agents.py -v
pytest tests/test_integration.py -v
pytest tests/test_api.py -v

# 运行测试并显示覆盖率
pytest tests/ --cov=src --cov-report=html
```

### 5. 启动服务

**终端1 - 后端服务**：
```bash
cd /Users/quinne/SoulMatch
source venv/bin/activate
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**终端2 - 前端服务**：
```bash
cd /Users/quinne/SoulMatch/frontend
npm run dev
```

访问：
- 前端：http://localhost:3000
- 后端API文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

## 项目结构

```
SoulMatch/
├── src/
│   ├── agents/           # Agent实现
│   │   ├── orchestrator.py         # 主编排器
│   │   ├── persona_agent.py        # Bot角色扮演
│   │   ├── feature_prediction_agent.py  # 特征推断
│   │   ├── emotion_agent.py        # 情绪分析
│   │   ├── scam_detection_agent.py # 诈骗检测
│   │   └── state_machine.py        # 状态机
│   ├── memory/           # 记忆管理
│   │   ├── memory_manager.py       # Memory Manager Agent
│   │   ├── memory_operations.py    # 操作定义
│   │   └── chromadb_client.py      # 向量数据库
│   ├── data/             # 数据处理
│   │   ├── schema.py               # 数据模型
│   │   ├── preprocessor.py         # 数据清洗
│   │   ├── feature_extractor.py    # LLM特征提取
│   │   └── persona_builder.py      # Persona构建
│   ├── training/         # 训练相关
│   │   ├── synthetic_dialogue_generator.py  # 合成对话生成
│   │   └── conversation_simulator.py        # 对话模拟
│   ├── matching/         # 匹配引擎
│   │   ├── matching_engine.py      # 匹配引擎
│   │   └── compatibility_scorer.py # 兼容性评分
│   ├── api/              # FastAPI后端
│   │   ├── main.py                 # 主应用
│   │   ├── websocket.py            # WebSocket端点
│   │   ├── session_manager.py      # 会话管理
│   │   └── chat_handler.py         # 聊天处理
│   └── config.py         # 配置管理
├── frontend/             # React前端
│   ├── src/
│   │   ├── App.tsx                 # 主应用
│   │   └── components/             # 组件
│   └── package.json
├── scripts/              # 工具脚本
│   ├── download_okcupid_data.py    # 数据下载
│   └── preprocess_data.py          # 数据预处理
├── tests/                # 测试
│   ├── test_agents.py              # Agent单元测试
│   ├── test_integration.py         # 集成测试
│   └── test_api.py                 # API测试
└── data/                 # 数据目录
    ├── raw/                        # 原始数据
    ├── processed/                  # 处理后数据
    └── training/                   # 训练数据
```

## 核心工作流

### 对话流程

1. **用户连接WebSocket** → `ws://localhost:8000/ws/{user_id}`
2. **开始对话** → 发送 `{"action": "start"}`
   - Orchestrator 调用 MatchingEngine 推荐Bot
   - Bot发送问候消息
3. **用户发送消息** → 发送 `{"action": "message", "content": "..."}`
   - Orchestrator协调所有子Agent：
     * EmotionAgent：分析情绪
     * ScamDetectionAgent：检测风险
     * FeaturePredictionAgent：更新特征（每3轮）
     * MemoryManager：管理记忆（每5轮）
     * PersonaAgent：生成回复
4. **服务端推送响应**：
   - `{type: "bot_message", message: "..."}`
   - `{type: "emotion", emotion: "joy", ...}`
   - `{type: "warning", level: "high", ...}` (如有风险)
   - `{type: "feature_update", ...}` (特征更新时)

### Agent协调机制

```
OrchestratorAgent
├── PersonaAgent (8个Bot) → 角色扮演
├── FeaturePredictionAgent → 推断用户特征
│   └── BayesianUpdater → 贝叶斯后验更新
├── MemoryManager → 记忆管理
│   └── ChromaDB → 向量存储
├── EmotionAgent → 情绪检测
│   └── EmotionPredictor → 情绪预测
├── ScamDetectionAgent → 诈骗检测
│   ├── ScamDetector (规则引擎)
│   └── SemanticAnalyzer (LLM语义)
└── MatchingEngine → 匹配推荐
    └── CompatibilityScorer → 兼容性评分
```

## 开发技巧

### 调试模式

在 `.env` 中设置：
```env
LOG_LEVEL=DEBUG
```

查看详细日志：
```bash
tail -f logs/soulmatch.log
```

### 测试单个Agent

```python
# 测试Emotion Agent
from src.agents.emotion_agent import EmotionAgent

agent = EmotionAgent(use_claude=True)
result = agent.analyze_message("I'm so excited!")
print(result)
```

### 生成合成对话数据

```python
from src.training.synthetic_dialogue_generator import create_synthetic_dataset

# 生成100条对话
create_synthetic_dataset(
    personas_path="data/processed/bot_personas.json",
    output_path="data/training/synthetic_dialogues.jsonl",
    num_conversations=100
)
```

### API测试（使用curl）

```bash
# 健康检查
curl http://localhost:8000/health

# 创建会话
curl -X POST http://localhost:8000/api/v1/session/start \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user"}'

# WebSocket测试（需要wscat）
npm install -g wscat
wscat -c ws://localhost:8000/ws/test_user
> {"action": "start"}
```

## 常见问题

### Q: ChromaDB初始化失败
**A**: 确保 `chroma_db/` 目录有写权限，或在 `.env` 中修改路径。

### Q: Bot personas加载失败
**A**: 运行 `python scripts/preprocess_data.py` 生成 `data/processed/bot_personas.json`。

### Q: LLM API调用超时
**A**: 检查网络连接和API密钥，或在 `src/config.py` 中增加超时时间。

### Q: 前端连接WebSocket失败
**A**: 确保后端服务运行在 `localhost:8000`，检查CORS配置。

## 贡献指南

1. 创建feature分支
2. 编写测试（tests/目录）
3. 运行测试确保通过
4. 提交PR

## License

MIT License
