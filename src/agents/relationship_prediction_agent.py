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
from .social_agents_room import SocialAgentsRoom  # New: Real demographic diversity


class RelationshipPredictionAgent:
    """关系状态预测Agent,融合多角色评估+共形预测"""

    def __init__(self, llm_router=None, calibrator: Optional[ConformalCalibrator] = None, use_social_agents: bool = True):
        self.llm = llm_router or router
        self.calibrator = calibrator or ConformalCalibrator(alpha=0.10)
        self.use_social_agents = use_social_agents
        self.social_agents_room = SocialAgentsRoom(self.llm) if use_social_agents else None

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
        conformal_result = await self._conformal_predict_advance(ctx, rel_assessment, sentiment)

        # Step 5: t+1预测
        next_status = await self._predict_next_status(ctx, rel_assessment["rel_status"], sentiment, rel_assessment)

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
            "advance_coverage_guarantee": conformal_result.get("coverage_guarantee", 0.9),
            "advance_softmax": conformal_result.get("softmax_distribution", {}),
            "blockers": conformal_result.get("blockers", []),
            "catalysts": conformal_result.get("catalysts", []),
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
        """Step 3: Social Agents评估(demographic diverse agents)→关系类型+状态"""

        if self.use_social_agents and self.social_agents_room:
            # 使用Social Agents Room进行demographic diverse评估
            return await self._social_agents_assessment(ctx, compressed_context)
        else:
            # Fallback: 单LLM评估
            return await self._single_llm_assessment(ctx, compressed_context)

    async def _social_agents_assessment(self, ctx: AgentContext, compressed_context: str) -> Dict[str, Any]:
        """使用Social Agents Room进行demographic diverse评估"""
        assert self.social_agents_room is not None, "social_agents_room should not be None when use_social_agents is True"

        relationship_context = {
            "rel_status": ctx.rel_status,
            "sentiment": ctx.sentiment_label,
            "trust_score": ctx.extended_features.get("trust_score", 0.5),
            "turn_count": ctx.turn_count,
        }

        # 调用Social Agents Room
        consensus = await self.social_agents_room.assess_relationship(
            conversation_summary=compressed_context,
            relationship_context=relationship_context
        )

        # 从consensus中提取关系类型和状态
        # 这里简化处理，实际应该让social agents也投票关系类型和状态
        return {
            "rel_type": "love" if consensus.decision == "compatible" else "other",
            "rel_type_probs": {"love": 0.7, "friendship": 0.2, "other": 0.1} if consensus.decision == "compatible" else {"other": 0.7, "friendship": 0.2, "love": 0.1},
            "rel_status": ctx.rel_status,  # 保持当前状态
            "rel_status_probs": {ctx.rel_status: 0.8},
            "rel_status_confidence": consensus.confidence,
            "reasoning": f"Social Agents Consensus ({consensus.vote_distribution}): {consensus.reasoning[:200]}...",
            "social_consensus": consensus,  # 保存完整的consensus结果
        }

    async def _discussion_room_assessment(self, ctx: AgentContext, compressed_context: str) -> Dict[str, Any]:
        """使用Agent讨论室进行多Agent辩论评估"""
        context = {
            "compressed_context": compressed_context,
            "turn_count": ctx.turn_count,
            "current_rel_status": ctx.rel_status,
            "big_five": ctx.predicted_features.get('big_five', {}),
            "interests": ctx.predicted_features.get('interests', {}),
            "trust_score": ctx.extended_features.get('trust_score', 0.5),
        }

        agents = [
            {
                "name": "EmotionExpert",
                "role": AgentRole.EMOTION,
                "expertise": "Emotional dynamics and sentiment analysis",
                "system_prompt": "You are an expert in emotional intelligence and relationship dynamics."
            },
            {
                "name": "ValuesExpert",
                "role": AgentRole.FEATURE,
                "expertise": "Value alignment and compatibility assessment",
                "system_prompt": "You are an expert in personality psychology and value systems."
            },
            {
                "name": "BehaviorExpert",
                "role": AgentRole.GENERAL,
                "expertise": "Behavioral patterns and interaction quality",
                "system_prompt": "You are an expert in behavioral psychology and social interactions."
            }
        ]

        voting_weights = {
            "EmotionExpert": 0.4,
            "ValuesExpert": 0.3,
            "BehaviorExpert": 0.3
        }

        topic = f"""Based on the context, assess:
1. Relationship type (love/friendship/family/other)
2. Relationship status (stranger/acquaintance/crush/dating/committed)

Output JSON:
{{
  "rel_type": "love|friendship|family|other",
  "rel_type_probs": {{"love": 0.x, "friendship": 0.y, ...}},
  "rel_status": "stranger|acquaintance|crush|dating|committed",
  "rel_status_probs": {{"stranger": 0.x, "acquaintance": 0.y, ...}}
}}

Constraint: rel_status can only advance or stay (current: {ctx.rel_status})"""

        consensus = await self.discussion_room.discuss(topic, context, agents, voting_weights)

        # 解析consensus.decision为JSON
        try:
            import re
            clean_decision = re.sub(r'^```json\s*|\s*```$', '', consensus.decision.strip(), flags=re.MULTILINE)
            result = json.loads(clean_decision)
            result["reasoning"] = consensus.reasoning

            # 单调约束
            status_order = ["stranger", "acquaintance", "crush", "dating", "committed"]
            current_idx = status_order.index(ctx.rel_status) if ctx.rel_status in status_order else 0
            predicted_idx = status_order.index(result["rel_status"]) if result["rel_status"] in status_order else 0
            if predicted_idx < current_idx:
                result["rel_status"] = ctx.rel_status

            return result
        except Exception as e:
            logger.warning(f"[RelationshipPredictionAgent] 讨论室结果解析失败: {e}")
            return await self._single_llm_assessment(ctx, compressed_context)

    async def _single_llm_assessment(self, ctx: AgentContext, compressed_context: str) -> Dict[str, Any]:
        """原有的单LLM模拟多角色方法(fallback)"""
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
            # 单调约束
            status_order = ["stranger", "acquaintance", "crush", "dating", "committed"]
            current_idx = status_order.index(ctx.rel_status) if ctx.rel_status in status_order else 0
            predicted_idx = status_order.index(result["rel_status"]) if result["rel_status"] in status_order else 0
            if predicted_idx < current_idx:
                result["rel_status"] = ctx.rel_status
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

    async def _conformal_predict_advance(self, ctx: AgentContext, rel_assessment: Dict, sentiment: Dict) -> Dict[str, Any]:
        """
        Step 4: 真正的共形预测 → can_advance prediction set

        实现完整的共形预测流程:
        1. LLM多次采样获取softmax分布
        2. 调用calibrator.calibrate()生成预测集
        3. 有序边界调整(ordinal boundary)
        4. blockers/catalysts结构化分析
        """
        # 1. 准备特征向量
        features = {
            "relationship_status": rel_assessment["rel_status"],
            "sentiment": sentiment["label"],
            "trust_score": ctx.extended_features.get("trust_score", 0.5),
            "turn": ctx.turn_count,
        }

        # 2. LLM多次采样获取can_advance的softmax分布
        softmax_dist = await self._sample_advance_distribution(ctx, rel_assessment, sentiment, n_samples=5)

        # 3. 构造预测字典和置信度字典
        predictions = {
            "can_advance": self._get_point_prediction_from_softmax(softmax_dist),
            "relationship_status": rel_assessment["rel_status"],
            "sentiment": sentiment["label"],
        }

        llm_confidences = {
            "can_advance": max(softmax_dist.values()) if softmax_dist else 0.5,
            "relationship_status": rel_assessment.get("rel_status_confidence", 0.7),
            "sentiment": sentiment.get("confidence", 0.7),
        }

        # 4. 调用ConformalCalibrator生成预测集
        conformal_result = self.calibrator.calibrate(
            predictions=predictions,
            llm_confidences=llm_confidences,
            turn=ctx.turn_count
        )

        # 5. 提取can_advance的预测集
        can_advance_ps = conformal_result.prediction_sets.get("can_advance")

        if can_advance_ps:
            prediction_set = can_advance_ps.prediction_set
            point_prediction = can_advance_ps.point_prediction
            can_advance = (point_prediction == "yes")
        else:
            # Fallback: 如果没有校准数据,使用规则
            prediction_set = self._fallback_prediction_set(features)
            can_advance = ("yes" in prediction_set)

        # 6. 有序边界调整(ordinal boundary)
        # 关系状态是有序的: stranger < acquaintance < crush < dating < committed
        # 如果已经是最高状态,不能再推进
        status_order = ["stranger", "acquaintance", "crush", "dating", "committed"]
        current_idx = status_order.index(rel_assessment["rel_status"]) if rel_assessment["rel_status"] in status_order else 0
        is_max = (current_idx >= len(status_order) - 1)

        if is_max:
            # 强制约束: 已到最高状态,不能推进
            prediction_set = ["no"]
            can_advance = False

        # 7. blockers/catalysts结构化分析
        blockers, catalysts = await self._analyze_blockers_catalysts(ctx, rel_assessment, sentiment)

        return {
            "can_advance": can_advance,
            "prediction_set": prediction_set,
            "coverage_guarantee": 1 - self.calibrator.alpha,
            "softmax_distribution": softmax_dist,
            "blockers": blockers,
            "catalysts": catalysts,
            "conformal_result": conformal_result,
        }

    async def _sample_advance_distribution(self, ctx: AgentContext, rel_assessment: Dict, sentiment: Dict, n_samples: int = 5) -> Dict[str, float]:
        """
        LLM多次采样获取can_advance的softmax分布

        Args:
            ctx: 对话上下文
            rel_assessment: 关系评估结果
            sentiment: 情感分析结果
            n_samples: 采样次数

        Returns:
            {"yes": 0.6, "no": 0.2, "uncertain": 0.2}
        """
        prompt = f"""
Based on the current relationship assessment, predict whether the relationship can advance to the next stage.

Current Status: {rel_assessment["rel_status"]}
Sentiment: {sentiment["label"]}
Trust Score: {ctx.extended_features.get("trust_score", 0.5):.2f}
Turn: {ctx.turn_count}

Respond with JSON:
{{
    "can_advance": "yes" | "no" | "uncertain",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}
"""

        # 多次采样
        samples = []
        for _ in range(n_samples):
            try:
                response = self.llm.chat(
                    role=AgentRole.GENERAL,
                    system="You are a relationship progression predictor.",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,  # 增加温度以获得多样性
                    max_tokens=200,
                    json_mode=True,
                )

                # 解析JSON
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.startswith("```"):
                    response = response[3:]
                if response.endswith("```"):
                    response = response[:-3]

                result = json.loads(response.strip())
                samples.append(result["can_advance"])
            except Exception as e:
                logger.warning(f"[RelationshipPredictionAgent] Sampling failed: {e}")
                continue

        # 计算softmax分布
        if not samples:
            # Fallback: 均匀分布
            return {"yes": 0.33, "no": 0.33, "uncertain": 0.34}

        # 统计频率
        from collections import Counter
        counts = Counter(samples)
        total = len(samples)

        softmax = {
            "yes": counts.get("yes", 0) / total,
            "no": counts.get("no", 0) / total,
            "uncertain": counts.get("uncertain", 0) / total,
        }

        return softmax

    def _get_point_prediction_from_softmax(self, softmax: Dict[str, float]) -> str:
        """从softmax分布中获取点预测"""
        if not softmax:
            return "uncertain"
        return max(softmax, key=softmax.get)

    def _fallback_prediction_set(self, features: Dict) -> list:
        """Fallback规则(当没有校准数据时)"""
        trust = features.get("trust_score", 0.5)
        sentiment = features.get("sentiment", "neutral")

        if trust > 0.7 and sentiment == "positive":
            return ["yes"]
        elif trust < 0.4 or sentiment == "negative":
            return ["no"]
        else:
            return ["uncertain", "yes"]

    async def _analyze_blockers_catalysts(self, ctx: AgentContext, rel_assessment: Dict, sentiment: Dict) -> tuple:
        """
        结构化分析blockers和catalysts

        Returns:
            (blockers: List[str], catalysts: List[str])
        """
        prompt = f"""
Analyze what factors are blocking or catalyzing relationship progression.

Current Status: {rel_assessment["rel_status"]}
Sentiment: {sentiment["label"]}
Recent Conversation: {ctx.recent_history(5)}

Respond with JSON:
{{
    "blockers": ["factor1", "factor2", ...],
    "catalysts": ["factor1", "factor2", ...]
}}
"""

        try:
            response = self.llm.chat(
                role=AgentRole.GENERAL,
                system="You are a relationship dynamics analyzer.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
                json_mode=True,
            )

            # 解析JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            result = json.loads(response.strip())
            return result.get("blockers", []), result.get("catalysts", [])
        except Exception as e:
            logger.warning(f"[RelationshipPredictionAgent] Blockers/catalysts analysis failed: {e}")
            return [], []

    async def _predict_next_status(self, ctx: AgentContext, current_status: str, sentiment: Dict, rel_assessment: Dict) -> Dict[str, Any]:
        """
        Step 5: t+1预测 - 动态预测下一状态概率分布

        基于:
        - 当前关系状态
        - 信任分数和变化趋势
        - 情感标签和趋势
        - 对话质量
        - 特征匹配度

        使用LLM多次采样获取概率分布
        """
        status_order = ["stranger", "acquaintance", "crush", "dating", "committed"]
        if current_status not in status_order:
            return {"status": current_status, "probs": {current_status: 1.0}}

        idx = status_order.index(current_status)
        if idx >= len(status_order) - 1:
            # 已到最高状态
            return {"status": current_status, "probs": {current_status: 1.0}}

        # 准备上下文特征
        trust_score = ctx.extended_features.get("trust_score", 0.5)
        trust_velocity = ctx.extended_features.get("trust_velocity", 0.0)
        emotion_trend = self._compute_emotion_trend(ctx)

        # LLM多次采样预测下一状态
        next_status_dist = await self._sample_next_status_distribution(
            current_status=current_status,
            trust_score=trust_score,
            trust_velocity=trust_velocity,
            sentiment=sentiment["label"],
            emotion_trend=emotion_trend,
            turn_count=ctx.turn_count,
            n_samples=5
        )

        # 获取最可能的状态
        most_likely = max(next_status_dist, key=next_status_dist.get)

        return {
            "status": most_likely,
            "probs": next_status_dist,
            "features_used": {
                "trust_score": trust_score,
                "trust_velocity": trust_velocity,
                "sentiment": sentiment["label"],
                "emotion_trend": emotion_trend
            }
        }

    async def _sample_next_status_distribution(
        self,
        current_status: str,
        trust_score: float,
        trust_velocity: float,
        sentiment: str,
        emotion_trend: str,
        turn_count: int,
        n_samples: int = 5
    ) -> Dict[str, float]:
        """
        LLM多次采样预测下一状态的概率分布

        Returns:
            {"stranger": 0.1, "acquaintance": 0.6, "crush": 0.3}
        """
        status_order = ["stranger", "acquaintance", "crush", "dating", "committed"]
        current_idx = status_order.index(current_status)

        # 可能的下一状态：维持当前或推进一步
        possible_next = [current_status]
        if current_idx < len(status_order) - 1:
            possible_next.append(status_order[current_idx + 1])

        prompt = f"""Predict the relationship status in the next 5 conversation turns.

Current Status: {current_status}
Trust Score: {trust_score:.2f} (velocity: {trust_velocity:+.2f})
Sentiment: {sentiment}
Emotion Trend: {emotion_trend}
Current Turn: {turn_count}

Based on these indicators, will the relationship:
- Stay at "{current_status}"
- Advance to "{possible_next[-1] if len(possible_next) > 1 else current_status}"

Consider:
- Trust score > 0.7 and positive sentiment → likely to advance
- Trust velocity > 0 and improving emotions → momentum for advancement
- Trust score < 0.5 or negative sentiment → likely to stay
- Low turn count → too early to advance

Respond with JSON:
{{
    "prediction": "{current_status}" | "{possible_next[-1] if len(possible_next) > 1 else current_status}",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}
"""

        # 多次采样
        samples = []
        for _ in range(n_samples):
            try:
                response = self.llm.chat(
                    role=AgentRole.GENERAL,
                    system="You are a relationship progression predictor.",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=200,
                    json_mode=True,
                )

                # 解析JSON
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.startswith("```"):
                    response = response[3:]
                if response.endswith("```"):
                    response = response[:-3]

                result = json.loads(response.strip())
                samples.append(result["prediction"])
            except Exception as e:
                logger.warning(f"[RelationshipPredictionAgent] Next status sampling failed: {e}")
                continue

        # 计算概率分布
        if not samples:
            # Fallback: 基于规则
            if trust_score > 0.7 and sentiment == "positive" and trust_velocity > 0:
                return {current_status: 0.4, possible_next[-1]: 0.6} if len(possible_next) > 1 else {current_status: 1.0}
            elif trust_score < 0.5 or sentiment == "negative":
                return {current_status: 0.9, possible_next[-1]: 0.1} if len(possible_next) > 1 else {current_status: 1.0}
            else:
                return {current_status: 0.7, possible_next[-1]: 0.3} if len(possible_next) > 1 else {current_status: 1.0}

        # 统计频率
        from collections import Counter
        counts = Counter(samples)
        total = len(samples)

        # 构建概率分布（只包含可能的状态）
        distribution = {}
        for status in possible_next:
            distribution[status] = counts.get(status, 0) / total

        # 归一化
        total_prob = sum(distribution.values())
        if total_prob > 0:
            distribution = {k: v / total_prob for k, v in distribution.items()}

        return distribution

    def _compute_emotion_trend(self, ctx: AgentContext) -> str:
        """计算情绪趋势(improving/declining/stable)"""
        if len(ctx.emotion_history) < 3:
            return "stable"

        # 简化映射: positive情绪→+1, negative→-1
        valence_map = {
            "joy": 1.0, "excitement": 0.8, "interest": 0.6, "trust": 0.7,
            "sadness": -0.7, "anger": -0.9, "fear": -0.6, "disgust": -0.8,
            "neutral": 0.0, "surprise": 0.3,
        }

        recent_3 = ctx.emotion_history[-3:]
        valences = [valence_map.get(e, 0.0) for e in recent_3]

        # 简单线性趋势
        if valences[-1] > valences[0] + 0.2:
            return "improving"
        elif valences[-1] < valences[0] - 0.2:
            return "declining"
        else:
            return "stable"
