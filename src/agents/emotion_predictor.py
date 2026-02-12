"""Emotion Predictor for predicting next emotional state based on history"""

from typing import Dict, List, Optional
from collections import Counter
from loguru import logger


# 8 emotion categories
EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral", "love"]


# Emotion transition patterns (simplified model)
# Each emotion maps to probable next emotions with weights
EMOTION_TRANSITIONS = {
    "joy": {"joy": 0.5, "love": 0.15, "surprise": 0.1, "neutral": 0.15, "sadness": 0.05, "anger": 0.02, "fear": 0.02, "disgust": 0.01},
    "sadness": {"sadness": 0.35, "neutral": 0.25, "anger": 0.15, "fear": 0.1, "joy": 0.1, "disgust": 0.03, "love": 0.01, "surprise": 0.01},
    "anger": {"anger": 0.4, "sadness": 0.2, "disgust": 0.15, "neutral": 0.1, "fear": 0.08, "surprise": 0.04, "joy": 0.02, "love": 0.01},
    "fear": {"fear": 0.35, "neutral": 0.2, "sadness": 0.15, "surprise": 0.1, "anger": 0.1, "disgust": 0.05, "joy": 0.03, "love": 0.02},
    "surprise": {"neutral": 0.25, "joy": 0.2, "surprise": 0.15, "fear": 0.15, "sadness": 0.1, "anger": 0.08, "love": 0.04, "disgust": 0.03},
    "disgust": {"disgust": 0.35, "anger": 0.25, "neutral": 0.15, "sadness": 0.1, "fear": 0.08, "surprise": 0.04, "joy": 0.02, "love": 0.01},
    "neutral": {"neutral": 0.4, "joy": 0.2, "surprise": 0.1, "sadness": 0.1, "love": 0.08, "anger": 0.06, "fear": 0.04, "disgust": 0.02},
    "love": {"love": 0.45, "joy": 0.3, "neutral": 0.1, "surprise": 0.08, "sadness": 0.04, "fear": 0.02, "anger": 0.005, "disgust": 0.005},
}


class EmotionPredictor:
    """Predicts next emotion based on conversation history"""
    
    def __init__(self, history_window: int = 5):
        """
        Initialize emotion predictor
        
        Args:
            history_window: Number of recent emotions to consider for prediction
        """
        self.history_window = history_window
        logger.info(f"EmotionPredictor initialized with history_window={history_window}")
    
    def predict_next_emotion(
        self, 
        emotion_history: List[str],
        return_distribution: bool = True
    ) -> Dict[str, float]:
        """
        Predict the next emotion based on recent emotion history
        
        Args:
            emotion_history: List of recent emotions (most recent last)
            return_distribution: If True, return full probability distribution
        
        Returns:
            Dictionary with predicted emotion and probabilities
            Format: {
                "predicted_emotion": "joy",
                "confidence": 0.65,
                "distribution": {"joy": 0.65, "neutral": 0.2, ...}  # if return_distribution=True
            }
        """
        if not emotion_history:
            logger.warning("Empty emotion history, defaulting to neutral")
            return self._default_prediction(return_distribution)
        
        # Get recent emotions (last N)
        recent_emotions = emotion_history[-self.history_window:]
        
        # Weight recent emotions (more recent = higher weight)
        weighted_predictions = self._compute_weighted_transitions(recent_emotions)
        
        # Get top prediction
        predicted_emotion = max(weighted_predictions.items(), key=lambda x: x[1])[0]
        confidence = weighted_predictions[predicted_emotion]
        
        result = {
            "predicted_emotion": predicted_emotion,
            "confidence": round(confidence, 3)
        }
        
        if return_distribution:
            result["distribution"] = {
                emotion: round(prob, 3) 
                for emotion, prob in sorted(
                    weighted_predictions.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
            }
        
        logger.debug(f"Predicted next emotion: {predicted_emotion} (confidence: {confidence:.2f})")
        return result
    
    def _compute_weighted_transitions(self, recent_emotions: List[str]) -> Dict[str, float]:
        """
        Compute weighted transition probabilities based on recent emotions
        
        More recent emotions have higher weight (exponential decay)
        """
        # Initialize probabilities
        emotion_probs = {emotion: 0.0 for emotion in EMOTIONS}
        
        # Calculate weights (exponential: 0.5, 0.7, 0.85, 0.95, 1.0 for last 5)
        total_weight = 0.0
        for idx, emotion in enumerate(recent_emotions):
            if emotion not in EMOTIONS:
                logger.warning(f"Unknown emotion '{emotion}' in history, skipping")
                continue
            
            # Exponential weight (more recent = higher weight)
            position_weight = 0.5 + (idx / (len(recent_emotions) - 1)) * 0.5 if len(recent_emotions) > 1 else 1.0
            
            # Add weighted transition probabilities
            transitions = EMOTION_TRANSITIONS.get(emotion, {})
            for next_emotion, prob in transitions.items():
                emotion_probs[next_emotion] += prob * position_weight
                total_weight += position_weight
        
        # Normalize probabilities
        if total_weight > 0:
            emotion_probs = {
                emotion: prob / total_weight 
                for emotion, prob in emotion_probs.items()
            }
        
        return emotion_probs
    
    def _default_prediction(self, return_distribution: bool) -> Dict[str, float]:
        """Return default prediction when no history is available"""
        result = {
            "predicted_emotion": "neutral",
            "confidence": 0.4
        }
        
        if return_distribution:
            result["distribution"] = {
                "neutral": 0.4,
                "joy": 0.2,
                "surprise": 0.15,
                "love": 0.1,
                "sadness": 0.08,
                "fear": 0.04,
                "anger": 0.02,
                "disgust": 0.01
            }
        
        return result
    
    def analyze_emotion_trend(self, emotion_history: List[str]) -> Dict[str, any]:
        """
        Analyze overall emotion trend in the conversation
        
        Args:
            emotion_history: List of all emotions in the conversation
        
        Returns:
            Dictionary with trend analysis:
            {
                "dominant_emotion": "joy",
                "emotion_counts": {"joy": 5, "neutral": 3, ...},
                "trend": "positive" | "negative" | "neutral" | "volatile",
                "volatility": 0.65  # emotion change frequency
            }
        """
        if not emotion_history:
            return {
                "dominant_emotion": "neutral",
                "emotion_counts": {},
                "trend": "neutral",
                "volatility": 0.0
            }
        
        # Count emotions
        emotion_counts = Counter(emotion_history)
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        
        # Determine trend (based on positive vs negative emotions)
        positive_emotions = ["joy", "love", "surprise"]
        negative_emotions = ["sadness", "anger", "fear", "disgust"]
        
        positive_count = sum(emotion_counts.get(e, 0) for e in positive_emotions)
        negative_count = sum(emotion_counts.get(e, 0) for e in negative_emotions)
        neutral_count = emotion_counts.get("neutral", 0)
        
        total = len(emotion_history)
        if positive_count / total > 0.5:
            trend = "positive"
        elif negative_count / total > 0.4:
            trend = "negative"
        elif neutral_count / total > 0.6:
            trend = "neutral"
        else:
            trend = "volatile"
        
        # Calculate volatility (emotion change frequency)
        changes = sum(
            1 for i in range(1, len(emotion_history)) 
            if emotion_history[i] != emotion_history[i-1]
        )
        volatility = changes / (len(emotion_history) - 1) if len(emotion_history) > 1 else 0.0
        
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_counts": dict(emotion_counts),
            "trend": trend,
            "volatility": round(volatility, 3)
        }
    
    def get_emotion_momentum(self, emotion_history: List[str], window: int = 3) -> str:
        """
        Determine if emotions are getting more positive or negative
        
        Args:
            emotion_history: List of emotions
            window: Window size for comparison
        
        Returns:
            "improving" | "declining" | "stable"
        """
        if len(emotion_history) < window * 2:
            return "stable"
        
        # Compare recent window vs previous window
        recent_window = emotion_history[-window:]
        previous_window = emotion_history[-window*2:-window]
        
        positive_emotions = ["joy", "love", "surprise"]
        negative_emotions = ["sadness", "anger", "fear", "disgust"]
        
        def emotion_score(emotions):
            positive = sum(1 for e in emotions if e in positive_emotions)
            negative = sum(1 for e in emotions if e in negative_emotions)
            return positive - negative
        
        recent_score = emotion_score(recent_window)
        previous_score = emotion_score(previous_window)
        
        if recent_score > previous_score:
            return "improving"
        elif recent_score < previous_score:
            return "declining"
        else:
            return "stable"
