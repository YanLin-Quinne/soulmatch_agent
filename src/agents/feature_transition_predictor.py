"""
FeatureTransitionPredictor — 特征时序预测

预测下一轮(t+1)哪些特征会变化,以及变化方向。

规则:
- 人格特质(Big Five/MBTI): 极低变化概率(<0.05)
- 信任度: 高频变化,依赖emotion_trend
- 依恋风格: 中低频变化(关系深化时缓慢位移)
"""

from typing import Dict, Any, List
from loguru import logger


class FeatureTransitionPredictor:
    """特征时序预测器"""

    def __init__(self):
        # 特征变化概率基线
        self.change_probs = {
            # 人格特质: 极低变化
            "big_five_openness": 0.02,
            "big_five_conscientiousness": 0.02,
            "big_five_extraversion": 0.03,
            "big_five_agreeableness": 0.03,
            "big_five_neuroticism": 0.04,
            "mbti_type": 0.01,
            "mbti_ei": 0.02,
            "mbti_sn": 0.02,
            "mbti_tf": 0.02,
            "mbti_jp": 0.02,
            # 依恋风格: 中低频
            "attachment_style": 0.10,
            "attachment_anxiety": 0.15,
            "attachment_avoidance": 0.15,
            # 信任度: 高频
            "trust_score": 0.70,
            "trust_velocity": 0.50,
            # 关系状态: 中频
            "relationship_status": 0.20,
            "sentiment_label": 0.40,
        }

    def predict_next(
        self,
        current_features: Dict[str, Any],
        emotion_trend: str,  # "improving" | "declining" | "stable"
        relationship_status: str,
        memory_trigger: bool = False,
    ) -> Dict[str, Any]:
        """
        预测t+1特征变化

        Returns:
        {
          "likely_to_change": ["trust_score", "attachment_anxiety"],
          "predicted_direction": {"trust_score": "+", "attachment_anxiety": "-"},
          "change_probability": {"trust_score": 0.72, "attachment_anxiety": 0.18},
          "stable_features": ["big_five_openness", "mbti_type", ...]
        }
        """
        likely_to_change = []
        predicted_direction = {}
        change_probability = {}
        stable_features = []

        # 遍历所有特征
        for feature, base_prob in self.change_probs.items():
            # 调整概率
            adjusted_prob = base_prob

            # 情绪趋势影响
            if feature == "trust_score":
                if emotion_trend == "improving":
                    adjusted_prob += 0.20
                elif emotion_trend == "declining":
                    adjusted_prob += 0.25  # 下降时变化更剧烈

            if feature == "sentiment_label":
                if emotion_trend != "stable":
                    adjusted_prob += 0.30

            # 关系状态影响
            if feature == "attachment_anxiety":
                if relationship_status in ["crush", "dating"]:
                    adjusted_prob += 0.10  # 关系深化时依恋焦虑可能变化

            if feature == "relationship_status":
                if relationship_status == "committed":
                    adjusted_prob = 0.05  # 已到最高状态,变化概率极低

            # 记忆触发影响
            if memory_trigger:
                adjusted_prob *= 1.3  # 记忆触发时所有特征变化概率提升30%

            # 判断是否会变化
            if adjusted_prob > 0.30:
                likely_to_change.append(feature)
                change_probability[feature] = round(adjusted_prob, 2)

                # 预测方向
                if feature == "trust_score":
                    predicted_direction[feature] = "+" if emotion_trend == "improving" else "-"
                elif feature == "sentiment_label":
                    predicted_direction[feature] = "+" if emotion_trend == "improving" else "-"
                elif feature == "attachment_anxiety":
                    predicted_direction[feature] = "-" if relationship_status in ["dating", "committed"] else "+"
                else:
                    predicted_direction[feature] = "?"
            else:
                stable_features.append(feature)

        logger.info(f"[FeatureTransitionPredictor] 预测 {len(likely_to_change)} 个特征可能变化")

        return {
            "likely_to_change": likely_to_change,
            "predicted_direction": predicted_direction,
            "change_probability": change_probability,
            "stable_features": stable_features,
        }
