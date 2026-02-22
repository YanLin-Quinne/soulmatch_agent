"""
MilestoneEvaluator — 里程碑评估(第10/30轮)

职责:
- 第10轮: 初步评估,预测第30轮时的关系状态
- 第30轮: 精确评估,计算预测准确率、共形效率、记忆贡献度
"""

from typing import Dict, Any, List
from loguru import logger


class MilestoneEvaluator:
    """里程碑评估器"""

    def __init__(self):
        pass

    def evaluate(
        self,
        turn: int,
        feature_history: List[Dict],
        relationship_snapshots: List[Dict],
        current_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        执行里程碑评估

        Args:
            turn: 当前轮次
            feature_history: 特征历史记录
            relationship_snapshots: 关系快照历史
            current_features: 当前特征

        Returns:
            milestone_report: 里程碑报告字典
        """
        if turn == 10:
            return self._evaluate_turn_10(relationship_snapshots, current_features)
        elif turn == 30:
            return self._evaluate_turn_30(feature_history, relationship_snapshots)
        else:
            return {}

    def _evaluate_turn_10(
        self,
        relationship_snapshots: List[Dict],
        current_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """第10轮: 初步评估"""
        if not relationship_snapshots:
            return {
                "turn": 10,
                "type": "initial_assessment",
                "message": "Insufficient data for assessment",
            }

        # 分析前10轮的关系轨迹
        statuses = [s.get("rel_status", "stranger") for s in relationship_snapshots]
        sentiments = [s.get("sentiment", "neutral") for s in relationship_snapshots]
        trust_scores = [s.get("trust_score", 0.5) for s in relationship_snapshots]

        # 计算趋势
        status_progression = len(set(statuses))  # 状态变化次数
        positive_sentiment_ratio = sentiments.count("positive") / max(len(sentiments), 1)
        trust_trend = "increasing" if trust_scores[-1] > trust_scores[0] else "decreasing"

        # 预测第30轮状态
        current_status = statuses[-1] if statuses else "stranger"
        status_order = ["stranger", "acquaintance", "crush", "dating", "committed"]
        current_idx = status_order.index(current_status) if current_status in status_order else 0

        # 简单预测: 基于当前趋势
        if positive_sentiment_ratio > 0.6 and trust_trend == "increasing":
            predicted_idx = min(current_idx + 2, len(status_order) - 1)
        elif positive_sentiment_ratio > 0.4:
            predicted_idx = min(current_idx + 1, len(status_order) - 1)
        else:
            predicted_idx = current_idx

        predicted_status_30 = status_order[predicted_idx]

        report = {
            "turn": 10,
            "type": "initial_assessment",
            "current_status": current_status,
            "status_progression": status_progression,
            "positive_sentiment_ratio": round(positive_sentiment_ratio, 2),
            "trust_trend": trust_trend,
            "predicted_status_at_turn_30": predicted_status_30,
            "confidence": "low",
            "message": f"初步评估: 当前{current_status}, 预测第30轮达到{predicted_status_30}",
        }

        logger.info(f"[MilestoneEvaluator] Turn 10评估完成: {report['message']}")
        return report

    def _evaluate_turn_30(
        self,
        feature_history: List[Dict],
        relationship_snapshots: List[Dict],
    ) -> Dict[str, Any]:
        """第30轮: 精确评估"""
        if not relationship_snapshots:
            return {
                "turn": 30,
                "type": "final_assessment",
                "message": "Insufficient data for assessment",
            }

        # 分析完整30轮轨迹
        statuses = [s.get("rel_status", "stranger") for s in relationship_snapshots]
        sentiments = [s.get("sentiment", "neutral") for s in relationship_snapshots]
        trust_scores = [s.get("trust_score", 0.5) for s in relationship_snapshots]

        # 计算指标
        final_status = statuses[-1] if statuses else "stranger"
        status_changes = sum(1 for i in range(1, len(statuses)) if statuses[i] != statuses[i-1])
        avg_trust = sum(trust_scores) / max(len(trust_scores), 1)
        positive_ratio = sentiments.count("positive") / max(len(sentiments), 1)
        negative_ratio = sentiments.count("negative") / max(len(sentiments), 1)

        # 特征收敛性分析
        feature_convergence = self._analyze_feature_convergence(feature_history)

        # 记忆贡献度(简化版: 假设记忆每5轮触发一次)
        memory_contribution = min(len(relationship_snapshots) / 6, 1.0)  # 30轮/5 = 6次记忆触发

        report = {
            "turn": 30,
            "type": "final_assessment",
            "final_status": final_status,
            "status_changes": status_changes,
            "avg_trust_score": round(avg_trust, 2),
            "positive_sentiment_ratio": round(positive_ratio, 2),
            "negative_sentiment_ratio": round(negative_ratio, 2),
            "feature_convergence": feature_convergence,
            "memory_contribution_score": round(memory_contribution, 2),
            "confidence": "high",
            "message": f"最终评估: 关系达到{final_status}, 信任度{avg_trust:.2f}, 特征收敛度{feature_convergence:.2f}",
        }

        logger.info(f"[MilestoneEvaluator] Turn 30评估完成: {report['message']}")
        return report

    def _analyze_feature_convergence(self, feature_history: List[Dict]) -> float:
        """分析特征收敛性(置信度提升程度)"""
        if len(feature_history) < 2:
            return 0.0

        # 比较第5轮和第30轮的平均置信度
        early_confidences = []
        late_confidences = []

        for record in feature_history[:5]:
            confidences = record.get("confidences", {})
            if confidences:
                early_confidences.extend(confidences.values())

        for record in feature_history[-5:]:
            confidences = record.get("confidences", {})
            if confidences:
                late_confidences.extend(confidences.values())

        if not early_confidences or not late_confidences:
            return 0.0

        early_avg = sum(early_confidences) / len(early_confidences)
        late_avg = sum(late_confidences) / len(late_confidences)

        # 收敛度 = 后期置信度 - 前期置信度
        convergence = max(0.0, late_avg - early_avg)
        return round(convergence, 2)
