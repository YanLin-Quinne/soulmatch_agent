# SoulMatch Agent

基于 OkCupid 数据集的社交匹配 Agent 系统。

## 项目概述

SoulMatch Agent 是一个创新的社交匹配系统，包含虚拟小镇场景，8假2真的人物混合，通过对话推断用户特征，情绪识别，杀猪盘检测，以及使用 SFT+RL 训练记忆管理和特征预测能力。

## 核心特性

- **多Agent架构**: Orchestrator, Persona, Feature Prediction, Memory Manager, Emotion, Scam Detection
- **记忆增强**: 借鉴 Memory-R1 和 ReMemR1 的记忆管理机制
- **特征推断**: 从对话中推断用户22个维度特征，支持贝叶斯更新
- **情绪分析**: 实时8类情绪检测和趋势预测
- **安全保护**: 杀猪盘检测和预警系统
- **智能训练**: SFT冷启动 + GRPO强化学习提升

## 技术栈

- **后端**: Python 3.12, FastAPI, WebSocket
- **前端**: React, TypeScript
- **模型**: Qwen3-0.6B (训练), Claude/GPT API (推理)
- **向量数据库**: ChromaDB
- **训练框架**: PyTorch, Transformers, TRL

## 项目结构

```
SoulMatch/
├── src/
│   ├── agents/           # Agent 实现
│   ├── memory/           # Memory Manager + ChromaDB
│   ├── data/             # OkCupid 数据处理
│   ├── training/         # SFT + RL 训练脚本
│   ├── matching/         # 匹配引擎
│   └── api/              # FastAPI 后端
├── frontend/             # React 前端
├── scripts/              # 数据下载、预处理脚本
├── tests/                # 测试文件
└── data/                 # 数据存储目录
```

## 安装

### 1. 环境配置

```bash
# 创建虚拟环境
python3.12 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 配置 Kaggle API
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key

# 下载 OkCupid 数据集
python scripts/download_okcupid_data.py
```

### 3. 配置环境变量

创建 `.env` 文件：

```env
# LLM API Keys
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key

# Database
CHROMA_DB_PATH=./chroma_db

# Training
MODEL_NAME=Qwen/Qwen2.5-0.5B
DEVICE=mps  # macOS M4 Pro
```

## 使用

### 启动后端服务

```bash
cd src/api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 启动前端

```bash
cd frontend
npm install
npm run dev
```

### 训练模型

```bash
# SFT 冷启动
python src/training/sft_trainer.py

# RL 提升
python src/training/rl_trainer.py
```

## 开发路线

- [x] 项目初始化
- [ ] 数据预处理
- [ ] Agent 实现
- [ ] 记忆管理系统
- [ ] 训练 pipeline
- [ ] 前后端集成
- [ ] 测试和优化

## License

MIT License
