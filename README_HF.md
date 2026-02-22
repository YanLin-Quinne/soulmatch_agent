---
title: SoulMatch Agent
emoji: 💕
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# SoulMatch Agent v2.0

Multi-agent relationship prediction system with conformal uncertainty quantification.

## Features

- 🤖 6 协同Agent（Orchestrator, Feature, Emotion, Scam, Persona, Question）
- 📊 42维用户特征推断（Big Five + MBTI + 依恋风格 + 爱语 + 信任轨迹）
- 🎯 保形预测（APS）不确定性量化
- 🧠 三层记忆管理（Working → Episodic → Semantic）
- 💬 实时WebSocket通信
- 🎨 精美UI设计（social-forecast风格）

## Usage

1. 选择一个Bot角色开始对话
2. 系统实时推断你的特征和情绪
3. 第10/30轮生成里程碑报告
4. 查看关系状态预测和保形预测区间

## Tech Stack

- Backend: FastAPI + Python 3.11
- Frontend: React + TypeScript
- LLM: GPT-5.2, Gemini Flash, DeepSeek
- Memory: ChromaDB
- Calibration: Conformal Prediction (APS)

## Paper

Based on research combining:
- Social Agents (ICLR 2026): Demographic diversity for wisdom of crowds
- Conformal Prediction: Uncertainty quantification with coverage guarantees
