"""Feature Prediction Agent - Infer user features from conversation"""

import json
from typing import Dict, List, Optional
from loguru import logger

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.agents.bayesian_updater import BayesianFeatureUpdater
from src.config import settings


class FeaturePredictionAgent:
    """
    Agent that predicts user features from conversation history
    
    Features predicted (22 dimensions from OkCupid):
    - Demographics: age, sex, orientation, location
    - Personality: Big Five traits (openness, conscientiousness, extraversion, agreeableness, neuroticism)
    - Lifestyle: diet, drinks, drugs, smokes
    - Values: education, job, religion
    - Communication: style, interests
    """
    
    def __init__(self, user_id: str, use_claude: bool = True):
        self.user_id = user_id
        self.use_claude = use_claude and ANTHROPIC_AVAILABLE
        
        # Initialize LLM client
        if self.use_claude:
            if not settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            logger.info("Using Claude for feature prediction")
        else:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed")
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self.client = openai.OpenAI(api_key=settings.openai_api_key)
            logger.info("Using GPT for feature prediction")
        
        # Bayesian updater
        self.bayesian_updater = BayesianFeatureUpdater()
        
        # Feature state
        self.predicted_features: Dict[str, any] = {}
        self.feature_confidences: Dict[str, float] = {}
        
        # Conversation tracking
        self.conversation_count = 0
        self.max_update_turns = 30  # Update features for first 30 turns
    
    def predict_from_conversation(
        self,
        conversation_history: List[dict]
    ) -> Dict[str, any]:
        """
        Predict features from conversation history
        
        Args:
            conversation_history: List of {"speaker": str, "message": str}
        
        Returns:
            Dict with predicted features and confidences
        """
        
        # Only update for first 30 turns
        if self.conversation_count >= self.max_update_turns:
            logger.info(f"Reached max update turns ({self.max_update_turns}), using cached features")
            return {
                "features": self.predicted_features,
                "confidences": self.feature_confidences,
                "turn": self.conversation_count
            }
        
        # Extract features using LLM
        new_observations = self._extract_features_llm(conversation_history)
        
        if not new_observations:
            logger.warning("No features extracted from conversation")
            return {
                "features": self.predicted_features,
                "confidences": self.feature_confidences,
                "turn": self.conversation_count
            }
        
        # Bayesian update
        updated_features, updated_confidences = self._update_features(new_observations)
        
        self.predicted_features = updated_features
        self.feature_confidences = updated_confidences
        self.conversation_count += 1
        
        logger.info(f"Updated features at turn {self.conversation_count}")
        
        return {
            "features": self.predicted_features,
            "confidences": self.feature_confidences,
            "turn": self.conversation_count,
            "information_gain": new_observations.get("_meta", {}).get("information_gain", 0.0)
        }
    
    def _extract_features_llm(
        self,
        conversation_history: List[dict]
    ) -> Optional[Dict[str, any]]:
        """Extract features from conversation using LLM"""
        
        # Build conversation context
        conversation_text = "\n".join([
            f"{msg['speaker']}: {msg['message']}"
            for msg in conversation_history[-10:]  # Last 10 messages
        ])
        
        prompt = f"""Analyze this dating app conversation and infer the user's features. Respond ONLY with valid JSON.

Conversation:
{conversation_text}

Infer the following features (use null if uncertain):

{{
  "age": <estimated age 18-80 or null>,
  "sex": "<m/f or null>",
  "orientation": "<straight/gay/bisexual or null>",
  "location": "<city or null>",
  
  "big_five": {{
    "openness": <0.0-1.0 or null>,
    "conscientiousness": <0.0-1.0 or null>,
    "extraversion": <0.0-1.0 or null>,
    "agreeableness": <0.0-1.0 or null>,
    "neuroticism": <0.0-1.0 or null>
  }},
  
  "lifestyle": {{
    "diet": "<omnivore/vegetarian/vegan/other or null>",
    "drinks": "<not_at_all/rarely/socially/often/very_often or null>",
    "smokes": "<no/sometimes/yes or null>",
    "drugs": "<never/sometimes or null>"
  }},
  
  "background": {{
    "education": "<high_school/some_college/bachelors/masters/phd or null>",
    "job": "<field or null>",
    "religion": "<atheist/agnostic/christian/jewish/muslim/hindu/buddhist/other or null>"
  }},
  
  "interests": {{
    "music": <0.0-1.0>,
    "sports": <0.0-1.0>,
    "travel": <0.0-1.0>,
    "food": <0.0-1.0>,
    "arts": <0.0-1.0>,
    "tech": <0.0-1.0>,
    "outdoors": <0.0-1.0>,
    "books": <0.0-1.0>
  }},
  
  "communication_style": "<direct/indirect/humorous/serious/casual/formal or null>",
  "relationship_goals": "<casual/serious/friendship/unsure or null>",
  
  "_confidence": {{
    "age": <0.0-1.0>,
    "sex": <0.0-1.0>,
    "orientation": <0.0-1.0>,
    "big_five_openness": <0.0-1.0>,
    "big_five_conscientiousness": <0.0-1.0>,
    "big_five_extraversion": <0.0-1.0>,
    "big_five_agreeableness": <0.0-1.0>,
    "big_five_neuroticism": <0.0-1.0>,
    "lifestyle_diet": <0.0-1.0>,
    "lifestyle_drinks": <0.0-1.0>,
    "background_education": <0.0-1.0>,
    "interests_music": <0.0-1.0>,
    "communication_style": <0.0-1.0>,
    "relationship_goals": <0.0-1.0>
  }}
}}

Guidelines:
- Only infer what can be reasonably deduced from the conversation
- Set confidence based on evidence strength (0.0 = no evidence, 1.0 = very strong evidence)
- Use null for features with no evidence
- Big Five: 0=low, 0.5=average, 1=high
- Interests: strength of interest (0=none, 1=very strong)

Respond with ONLY the JSON object."""

        try:
            if self.use_claude:
                response = self.client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=1024,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1024
                )
                content = response.choices[0].message.content
            
            # Parse JSON
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            features = json.loads(content)
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def _update_features(
        self,
        new_observations: Dict[str, any]
    ) -> tuple[Dict[str, any], Dict[str, float]]:
        """Update features using Bayesian update"""
        
        # Extract confidences
        confidences = new_observations.pop("_confidence", {})
        
        # Flatten nested structure for updating
        flat_observations = {}
        flat_confidences = {}
        
        # Simple features
        for key in ["age", "sex", "orientation", "location", 
                   "communication_style", "relationship_goals"]:
            if key in new_observations and new_observations[key] is not None:
                flat_observations[key] = new_observations[key]
                flat_confidences[key] = confidences.get(key, 0.5)
        
        # Big Five
        if "big_five" in new_observations:
            for trait, value in new_observations["big_five"].items():
                if value is not None:
                    flat_observations[f"big_five_{trait}"] = value
                    flat_confidences[f"big_five_{trait}"] = confidences.get(f"big_five_{trait}", 0.5)
        
        # Lifestyle
        if "lifestyle" in new_observations:
            for key, value in new_observations["lifestyle"].items():
                if value is not None:
                    flat_observations[f"lifestyle_{key}"] = value
                    flat_confidences[f"lifestyle_{key}"] = confidences.get(f"lifestyle_{key}", 0.5)
        
        # Background
        if "background" in new_observations:
            for key, value in new_observations["background"].items():
                if value is not None:
                    flat_observations[f"background_{key}"] = value
                    flat_confidences[f"background_{key}"] = confidences.get(f"background_{key}", 0.5)
        
        # Interests (numeric)
        if "interests" in new_observations:
            for interest, value in new_observations["interests"].items():
                flat_observations[f"interest_{interest}"] = value
                flat_confidences[f"interest_{interest}"] = confidences.get(f"interests_{interest}", 0.5)
        
        # Bayesian update for numeric features
        numeric_keys = [k for k in flat_observations.keys() 
                       if k.startswith("big_five_") or k.startswith("interest_")]
        
        updated_numeric_features = {}
        updated_numeric_confidences = {}
        
        for key in numeric_keys:
            prior_value = self.predicted_features.get(key, 0.5)
            prior_conf = self.feature_confidences.get(key, 0.3)
            
            new_value = flat_observations[key]
            new_conf = flat_confidences[key]
            
            updated_value, updated_conf = self.bayesian_updater.update_feature(
                prior_value=prior_value,
                prior_confidence=prior_conf,
                new_observation=new_value,
                observation_confidence=new_conf
            )
            
            updated_numeric_features[key] = updated_value
            updated_numeric_confidences[key] = updated_conf
        
        # For categorical features, use new observation if confidence is higher
        categorical_keys = [k for k in flat_observations.keys() if k not in numeric_keys]
        
        updated_categorical_features = {}
        updated_categorical_confidences = {}
        
        for key in categorical_keys:
            new_value = flat_observations[key]
            new_conf = flat_confidences[key]
            
            if key in self.predicted_features:
                prior_conf = self.feature_confidences.get(key, 0.3)
                
                # Use new value if confidence is higher
                if new_conf > prior_conf:
                    updated_categorical_features[key] = new_value
                    updated_categorical_confidences[key] = new_conf
                else:
                    updated_categorical_features[key] = self.predicted_features[key]
                    updated_categorical_confidences[key] = prior_conf
            else:
                updated_categorical_features[key] = new_value
                updated_categorical_confidences[key] = new_conf
        
        # Combine
        updated_features = {**updated_numeric_features, **updated_categorical_features}
        updated_confidences = {**updated_numeric_confidences, **updated_categorical_confidences}
        
        return updated_features, updated_confidences
    
    def get_feature_summary(self) -> Dict[str, any]:
        """Get human-readable feature summary"""
        
        summary = {
            "demographics": {},
            "personality": {},
            "lifestyle": {},
            "interests": {},
            "overall_confidence": self._compute_overall_confidence()
        }
        
        # Demographics
        for key in ["age", "sex", "orientation", "location"]:
            if key in self.predicted_features:
                summary["demographics"][key] = {
                    "value": self.predicted_features[key],
                    "confidence": self.feature_confidences.get(key, 0.0)
                }
        
        # Personality (Big Five)
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            key = f"big_five_{trait}"
            if key in self.predicted_features:
                summary["personality"][trait] = {
                    "value": self.predicted_features[key],
                    "confidence": self.feature_confidences.get(key, 0.0)
                }
        
        # Interests
        for interest in ["music", "sports", "travel", "food", "arts", "tech", "outdoors", "books"]:
            key = f"interest_{interest}"
            if key in self.predicted_features:
                summary["interests"][interest] = {
                    "value": self.predicted_features[key],
                    "confidence": self.feature_confidences.get(key, 0.0)
                }
        
        return summary
    
    def _compute_overall_confidence(self) -> float:
        """Compute overall confidence across all features"""
        
        if not self.feature_confidences:
            return 0.0
        
        return sum(self.feature_confidences.values()) / len(self.feature_confidences)
