"""Bayesian 更新模块 - 特征推断的后验概率计算"""

from typing import Tuple


class BayesianUpdater:
    """Bayesian 后验更新器"""

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


# 向后兼容别名
BayesianFeatureUpdater = BayesianUpdater
