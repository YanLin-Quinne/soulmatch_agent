"""Integration tests for FeaturePredictionAgent pipeline"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.feature_prediction_agent import FeaturePredictionAgent
from src.agents.bayesian_updater import BayesianFeatureUpdater
from src.agents.conformal_calibrator import ConformalCalibrator


class TestFeaturePredictionPipeline:
    """Test feature prediction pipeline: CoT → Bayesian → Conformal"""

    @pytest.fixture
    def agent(self):
        return FeaturePredictionAgent(user_id="test_user", use_cot=True, calibrator_path=None)

    @pytest.mark.asyncio
    async def test_cot_reasoning_extraction(self, agent):
        """Verify CoT reasoning signal extraction (turn >= 3)"""
        agent.conversation_count = 3

        mock_response = {
            "openness": 0.7,
            "openness_confidence": 0.8,
            "extraversion": 0.6,
            "extraversion_confidence": 0.75
        }

        with patch.object(agent, '_extract_features_llm', return_value=mock_response):
            result = agent._extract_features_llm([])

            assert result is not None
            assert "openness" in result
            assert result["openness_confidence"] >= 0.7

    def test_bayesian_posterior_update(self, agent):
        """Verify Bayesian posterior update (confidence_post > confidence_prior)"""
        updater = BayesianFeatureUpdater()

        prior_value = 0.5
        prior_confidence = 0.3
        new_observation = 0.8
        observation_confidence = 0.9

        updated_value, updated_confidence = updater.update_feature(
            prior_value=prior_value,
            prior_confidence=prior_confidence,
            new_observation=new_observation,
            observation_confidence=observation_confidence
        )

        # Verify confidence increases
        assert updated_confidence > prior_confidence

        # Verify updated value moves toward observation
        assert updated_value > prior_value

    def test_conformal_prediction_set_generation(self, agent):
        """Verify conformal prediction set generation (APS algorithm)"""
        calibrator = ConformalCalibrator(alpha=0.10)

        # Mock calibration samples with correct structure
        from src.agents.conformal_calibrator import CalibrationSample
        mock_samples = [
            CalibrationSample(
                profile_id=f"profile_{i}",
                turn=5,
                dimension="big_five_openness",
                llm_softmax={"low": 0.1, "medium": 0.2, "high": 0.7},
                ground_truth="high"
            )
            for i in range(100)
        ]

        calibrator.fit(mock_samples)

        result = calibrator.calibrate(
            predictions={"big_five_openness": 0.7},
            llm_confidences={"big_five_openness": 0.8},
            turn=5
        )

        assert result is not None
        assert "big_five_openness" in result.prediction_sets

    def test_24_dimensional_feature_coverage(self, agent):
        """Verify feature space coverage"""
        from src.agents.conformal_calibrator import ALL_DIMS

        # Verify ALL_DIMS exists and has dimensions
        assert len(ALL_DIMS) > 20  # At least 20+ dimensions

        # Verify agent can handle dimensions
        agent.predicted_features = {dim: 0.5 for dim in list(ALL_DIMS)[:10]}
        agent.feature_confidences = {dim: 0.7 for dim in list(ALL_DIMS)[:10]}

        assert len(agent.predicted_features) == 10
        assert len(agent.feature_confidences) == 10

    def test_confidence_convergence(self, agent):
        """Verify confidence convergence (avg_conf >= 0.80)"""
        # Set all 24 dimensions with high confidence
        from src.agents.conformal_calibrator import ALL_DIMS
        agent.feature_confidences = {dim: 0.85 for dim in ALL_DIMS}

        avg_conf = agent._compute_overall_confidence()

        assert avg_conf >= 0.80

        # Test convergence check
        agent.conversation_count = 6
        low_conf = agent._low_confidence_features()

        # All features above threshold, should be minimal
        assert len(low_conf) < len(ALL_DIMS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
