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

- 15 collaborative agents (Orchestrator, Feature, Emotion, Scam, Persona, Relationship, Memory, etc.)
- 42-dimensional user feature inference (Big Five, MBTI, attachment style, love language, trust trajectory)
- Conformal prediction (APS) with coverage guarantees
- Three-layer memory management (Working, Episodic, Semantic)
- Real-time WebSocket communication
- Multi-LLM routing (GPT-5.2, Gemini 3.1 Pro, Claude Opus 4.6, Qwen 3.5 Plus, DeepSeek V3.2)

## Usage

1. Select a bot persona to start a conversation
2. The system infers your features and emotions in real time
3. Milestone reports are generated at turns 10 and 30
4. View relationship state predictions with conformal prediction intervals

## Tech Stack

- Backend: FastAPI + Python 3.11
- Frontend: React + TypeScript
- LLM: GPT-5.2, Gemini 3.1 Pro, Claude Opus 4.6, Qwen 3.5 Plus, DeepSeek V3.2
- Memory: ChromaDB + Three-layer architecture
- Calibration: Conformal Prediction (APS)

## Paper

Based on research combining:
- Social Agents (ICLR 2026): Demographic diversity for wisdom of crowds
- Conformal Prediction (EMNLP 2025): Uncertainty quantification with coverage guarantees
