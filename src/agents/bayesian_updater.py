"""Bayesian updater for feature predictions"""

import numpy as np
from typing import Dict, Optional
from loguru import logger


class BayesianFeatureUpdater:
    """Bayesian posterior update for feature confidence"""
    
    def __init__(self, prior_mean: float = 0.5, prior_std: float = 0.3):
        """
        Initialize with prior distribution
        
        Args:
            prior_mean: Prior mean for features (default 0.5 = neutral)
            prior_std: Prior standard deviation (default 0.3 = moderate uncertainty)
        """
        self.prior_mean = prior_mean
        self.prior_std = prior_std
    
    def update_feature(
        self,
        prior_value: float,
        prior_confidence: float,
        new_observation: float,
        observation_confidence: float
    ) -> tuple[float, float]:
        """
        Update feature value using Bayesian update
        
        Args:
            prior_value: Previous feature value (0-1)
            prior_confidence: Confidence in prior (0-1)
            new_observation: New observed value (0-1)
            observation_confidence: Confidence in new observation (0-1)
        
        Returns:
            (updated_value, updated_confidence)
        """
        
        # Convert confidence to precision (inverse variance)
        prior_precision = self._confidence_to_precision(prior_confidence)
        obs_precision = self._confidence_to_precision(observation_confidence)
        
        # Bayesian update: posterior precision = prior precision + observation precision
        posterior_precision = prior_precision + obs_precision
        
        # Posterior mean: weighted average
        posterior_mean = (
            prior_precision * prior_value + obs_precision * new_observation
        ) / posterior_precision
        
        # Convert precision back to confidence
        posterior_confidence = self._precision_to_confidence(posterior_precision)
        
        # Clip to [0, 1]
        posterior_mean = np.clip(posterior_mean, 0.0, 1.0)
        posterior_confidence = np.clip(posterior_confidence, 0.0, 1.0)
        
        return float(posterior_mean), float(posterior_confidence)
    
    def _confidence_to_precision(self, confidence: float) -> float:
        """Convert confidence [0,1] to precision (higher = more precise)"""
        # Map confidence to variance, then to precision
        # Low confidence → high variance → low precision
        # High confidence → low variance → high precision
        
        # Avoid division by zero
        confidence = max(confidence, 0.01)
        
        # Variance = (1 - confidence)^2
        variance = (1 - confidence) ** 2
        variance = max(variance, 0.01)  # Floor variance
        
        # Precision = 1 / variance
        precision = 1.0 / variance
        
        return precision
    
    def _precision_to_confidence(self, precision: float) -> float:
        """Convert precision to confidence [0,1]"""
        # Inverse of confidence_to_precision
        
        # Variance = 1 / precision
        variance = 1.0 / max(precision, 0.01)
        
        # Confidence = 1 - sqrt(variance)
        confidence = 1.0 - np.sqrt(variance)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def update_features_batch(
        self,
        prior_features: Dict[str, float],
        prior_confidences: Dict[str, float],
        new_observations: Dict[str, float],
        observation_confidences: Dict[str, float]
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        """
        Update multiple features at once
        
        Returns:
            (updated_features, updated_confidences)
        """
        
        updated_features = {}
        updated_confidences = {}
        
        # Update each feature
        for key in prior_features.keys():
            if key in new_observations:
                # Update with new observation
                updated_value, updated_conf = self.update_feature(
                    prior_value=prior_features[key],
                    prior_confidence=prior_confidences.get(key, 0.5),
                    new_observation=new_observations[key],
                    observation_confidence=observation_confidences.get(key, 0.5)
                )
                
                updated_features[key] = updated_value
                updated_confidences[key] = updated_conf
            else:
                # No new observation, keep prior
                updated_features[key] = prior_features[key]
                updated_confidences[key] = prior_confidences.get(key, 0.5)
        
        # Add new features
        for key in new_observations.keys():
            if key not in prior_features:
                updated_features[key] = new_observations[key]
                updated_confidences[key] = observation_confidences.get(key, 0.5)
        
        return updated_features, updated_confidences
    
    def compute_information_gain(
        self,
        prior_confidence: float,
        posterior_confidence: float
    ) -> float:
        """
        Compute information gain from update
        
        Information gain = reduction in uncertainty
        
        Returns:
            Float [0, 1] where higher = more information gained
        """
        
        # Uncertainty = 1 - confidence
        prior_uncertainty = 1.0 - prior_confidence
        posterior_uncertainty = 1.0 - posterior_confidence
        
        # Information gain = reduction in uncertainty
        gain = prior_uncertainty - posterior_uncertainty
        
        # Normalize to [0, 1]
        gain = gain / max(prior_uncertainty, 0.01)
        
        return float(np.clip(gain, 0.0, 1.0))
