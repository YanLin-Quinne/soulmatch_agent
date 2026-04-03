"""Bayesian 更新模块 - 特征推断的后验概率计算"""

from typing import Tuple


class BayesianUpdater:
    """Bayesian 后验更新器"""

    @staticmethod
    def _confidence_to_variance(confidence: float) -> float:
        """Map a [0, 1] confidence score to a small positive variance."""
        confidence = max(0.0, min(1.0, float(confidence)))
        return max(1e-3, 1.0 - confidence)

    @staticmethod
    def _variance_to_confidence(variance: float) -> float:
        """Map posterior variance back to a bounded confidence score."""
        variance = max(0.0, float(variance))
        return max(0.0, min(1.0, 1.0 - variance))

    def update_posterior(
        self,
        prior_mean: float,
        prior_variance: float,
        observation: float,
        observation_variance: float
    ) -> Tuple[float, float]:
        """
        计算后验分布（高斯-高斯共轭）

        Args:
            prior_mean: 先验均值
            prior_variance: 先验方差
            observation: 观测值
            observation_variance: 观测方差

        Returns:
            (posterior_mean, posterior_variance)
        """
        # 精度（方差的倒数）
        prior_precision = 1.0 / prior_variance if prior_variance > 0 else 1e-6
        obs_precision = 1.0 / observation_variance if observation_variance > 0 else 1e-6

        # 后验精度 = 先验精度 + 观测精度
        posterior_precision = prior_precision + obs_precision

        # 后验均值 = 加权平均
        posterior_mean = (
            prior_precision * prior_mean + obs_precision * observation
        ) / posterior_precision

        # 后验方差 = 1 / 后验精度
        posterior_variance = 1.0 / posterior_precision

        return posterior_mean, posterior_variance

    def update_feature(
        self,
        prior_value: float,
        prior_confidence: float,
        observation_value: float,
        observation_confidence: float,
    ) -> Tuple[float, float]:
        """Backward-compatible wrapper used by FeaturePredictionAgent."""
        posterior_mean, posterior_variance = self.update_posterior(
            prior_mean=float(prior_value),
            prior_variance=self._confidence_to_variance(prior_confidence),
            observation=float(observation_value),
            observation_variance=self._confidence_to_variance(observation_confidence),
        )
        return posterior_mean, self._variance_to_confidence(posterior_variance)


# 向后兼容别名
BayesianFeatureUpdater = BayesianUpdater
