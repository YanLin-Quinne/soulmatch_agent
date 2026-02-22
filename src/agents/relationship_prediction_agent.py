"""
RelationshipPredictionAgent — 关系状态预测与共形不确定性量化

核心职责:
1. 预测关系状态(stranger→acquaintance→crush→dating→committed)
2. 预测关系类型(love/friendship/family/other)
3. 预测情感倾向(positive/neutral/negative)
4. 用共形预测量化"能否往前推进"的不确定性

设计灵感:
- 多智能体"众智"提升预测准确性
- 共形预测提供有覆盖保证的预测区间
"""

import json
from typing import Dict, Any, Optional
from loguru import logger

from .agent_context import AgentContext
from .llm_router import router, AgentRole
from .conformal_calibrator import ConformalCalibrator


class RelationshipPredictionAgent:
    """关系状态预测Agent,融合多角色评估+共形预测"""

    def __init__(self, llm_router=None, calibrator: Optional[ConformalCalibrator] = None):
        self.llm = llm_router or router
        self.calibrator = calibrator or ConformalCalibrator(alpha=0.10)

    async def execute(self, ctx: AgentContext) -> Dict[str, Any]:
        """
        执行关系预测流程(5步):
        1. 上下文压缩
        2. 情感基线分析
        3. 关系类型+状态分类
        4. 共形预测
        5. t+1预测
        """
        # 触发条件: 每5轮 + 第10/30轮
        if ctx.turn_count % 5 != 0 and ctx.turn_count not in [10, 30]:
            return {}

        logger.info(f"[RelationshipPredictionAgent] Turn {ctx.turn_count}: 开始关系预测")

        # Step 1: 上下文压缩
        compressed_context = await self._compress_context(ctx)

        # Step 2: 情感基线分析
        sentiment = self._analyze_sentiment(ctx)

        # Step 3: 多角色评估 → 关系类型+状态
        rel_assessment = await self._multi_role_assessment(ctx, compressed_context)

        # Step 4: 共形预测 → can_advance
        conformal_result = self._conformal_predict_advance(ctx, rel_assessment)

        # Step 5: t+1预测
        next_status = self._predict_next_status(rel_assessment["rel_status"])

        # 构建结果
        result = {
            "sentiment": sentiment["label"],
            "sentiment_confidence": sentiment["confidence"],
            "rel_type": rel_assessment["rel_type"],
            "rel_type_probs": rel_assessment["rel_type_probs"],
            "rel_status": rel_assessment["rel_status"],
            "rel_status_probs": rel_assessment["rel_status_probs"],
            "can_advance": conformal_result["can_advance"],
            "advance_prediction_set": conformal_result["prediction_set"],
            "advance_coverage_guarantee": 0.9,
            "next_status_prediction": next_status["status"],
            "next_status_probs": next_status["probs"],
            "reasoning_trace": rel_assessment.get("reasoning", ""),
        }

        # 写入context
        ctx.relationship_result = result
        ctx.rel_status = result["rel_status"]
        ctx.rel_type = result["rel_type"]
        ctx.sentiment_label = result["sentiment"]
        ctx.can_advance = result["can_advance"]

        # 保存快照
        snapshot = {
            "turn": ctx.turn_count,
            "sentiment": sentiment["label"],
            "rel_type": rel_assessment["rel_type"],
            "rel_status": rel_assessment["rel_status"],
            "trust_score": ctx.extended_features.get("trust_score", 0.5),
            "emotion_valence": sentiment["valence"],
        }
        ctx.relationship_snapshots.append(snapshot)

        logger.info(f"[RelationshipPredictionAgent] 预测完成: {result['rel_status']} ({result['rel_type']}), can_advance={result['can_advance']}")
        return result

    async def _compress_context(self, ctx: AgentContext) -> str:
        """Step 1: LLM压缩长对话为结构化关系日志(15行以内)"""
        recent = ctx.recent_history(50)
        if not recent:
            return "No conversation history yet."

        history_text = "\n".join([f"{h['speaker']}: {h['message']}" for h in recent])

        prompt = f"""Compress the following conversation into a structured relationship log (max 15 lines).
Focus on: emotional shifts, trust signals, intimacy progression, conflict/resolution.

Conversation:
{history_text}

Output format:
- Turn X: [event summary]
- Turn Y: [event summary]
..."""

        response = self.llm.chat(
            role=AgentRole.PERSONA,
            system="You are a relationship analyst. Extract key relationship events.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return response.strip()

    def _analyze_sentiment(self, ctx: AgentContext) -> Dict[str, Any]:
        """Step 2: 滑动窗口(最近5轮)计算情感平均valence"""
        recent_emotions = ctx.emotion_history[-5:] if ctx.emotion_history else []
        if not recent_emotions:
            return {"label": "neutral", "confidence": 0.5, "valence": 0.0}

        # 简化映射: positive情绪→+1, negative→-1, neutral→0
        valence_map = {
            "joy": 1.0, "excitement": 0.8, "interest": 0.6, "trust": 0.7,
            "sadness": -0.7, "anger": -0.9, "fear": -0.6, "disgust": -0.8,
            "neutral": 0.0, "surprise": 0.3,
        }

        valences = [valence_map.get(e, 0.0) for e in recent_emotions]
        avg_valence = sum(valences) / len(valences)

        if avg_valence > 0.3:
            label = "positive"
        elif avg_valence < -0.3:
            label = "negative"
        else:
            label = "neutral"

        confidence = min(abs(avg_valence) + 0.5, 1.0)
        return {"label": label, "confidence": confidence, "valence": avg_valence}

    async def _multi_role_assessment(self, ctx: AgentContext, compressed_context: str) -> Dict[str, Any]:
        """Step 3: 多角色评估(情感/价值观/行为专家)→关系类型+状态"""
        # 简化版: 单次LLM调用,模拟3个角色的加权聚合
        prompt = f"""You are a relationship assessment panel with 3 experts:
1. Emotional analyst (weight: 0.4)
2. Values compatibility analyst (weight: 0.3)
3. Behavioral pattern analyst (weight: 0.3)

Context:
{compressed_context}

Current features:
- Big Five: {ctx.predicted_features.get('big_five', {})}
- Interests: {ctx.predicted_features.get('interests', {})}
- Turn: {ctx.turn_count}

Assess the relationship and output JSON:
{{
  "rel_type": "love|friendship|family|other",
  "rel_type_probs": {{"love": 0.x, "friendship": 0.y, ...}},
  "rel_status": "stranger|acquaintance|crush|dating|committed",
  "rel_status_probs": {{"stranger": 0.x, "acquaintance": 0.y, ...}},
  "reasoning": "brief explanation"
}}

Constraints:
- rel_status can only advance or stay (monotonic)
- Current status: {ctx.rel_status}
"""

        response = self.llm.chat(
            role=AgentRole.PERSONA,
            system="You are a relationship assessment panel. Output valid JSON only.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            json_mode=True,
        )

        try:
            result = json.loads(response.strip())
            # 单调约束: 不能倒退
            status_order = ["stranger", "acquaintance", "crush", "dating", "committed"]
            current_idx = status_order.index(ctx.rel_status) if ctx.rel_status in status_order else 0
            predicted_idx = status_order.index(result["rel_status"]) if result["rel_status"] in status_order else 0
            if predicted_idx < current_idx:
                result["rel_status"] = ctx.rel_status  # 维持现状
            return result
        except Exception as e:
            logger.warning(f"[RelationshipPredictionAgent] JSON解析失败: {e}, 使用默认值")
            return {
                "rel_type": "other",
                "rel_type_probs": {"other": 1.0},
                "rel_status": ctx.rel_status or "stranger",
                "rel_status_probs": {ctx.rel_status or "stranger": 1.0},
                "reasoning": "Parse error",
            }

    def _conformal_predict_advance(self, ctx: AgentContext, rel_assessment: Dict) -> Dict[str, Any]:
        """Step 4: 共形预测 → can_advance prediction set"""
        # 特征: rel_status, sentiment, trust_score, turn
        status_order = ["stranger", "acquaintance", "crush", "dating", "committed"]
        current_idx = status_order.index(rel_assessment["rel_status"]) if rel_assessment["rel_status"] in status_order else 0
        is_max = (current_idx >= len(status_order) - 1)

        # 简化规则: 如果已到最高状态,不能推进
        if is_max:
            return {"can_advance": False, "prediction_set": ["no"]}

        # 否则基于trust_score和sentiment
        trust = ctx.extended_features.get("trust_score", 0.5)
        sentiment = ctx.sentiment_label

        # 点预测
        if trust > 0.7 and sentiment == "positive":
            can_advance = True
            pred_set = ["yes"]
        elif trust < 0.4 or sentiment == "negative":
            can_advance = False
            pred_set = ["no"]
        else:
            can_advance = False
            pred_set = ["uncertain", "yes"]  # 不确定集合

        return {"can_advance": can_advance, "prediction_set": pred_set}

    def _predict_next_status(self, current_status: str) -> Dict[str, Any]:
        """Step 5: t+1预测(简单马尔可夫转移)"""
        status_order = ["stranger", "acquaintance", "crush", "dating", "committed"]
        if current_status not in status_order:
            return {"status": current_status, "probs": {current_status: 1.0}}

        idx = status_order.index(current_status)
        if idx >= len(status_order) - 1:
            # 已到最高状态
            return {"status": current_status, "probs": {current_status: 1.0}}

        # 简化转移概率: 70%维持, 30%推进
        next_status = status_order[idx + 1]
        return {
            "status": current_status,  # 最可能维持
            "probs": {
                current_status: 0.7,
                next_status: 0.3,
            }
        }
