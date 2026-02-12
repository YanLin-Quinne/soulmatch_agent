"""Compatibility Scorer - Compute compatibility between user features and bot personas"""

import numpy as np
from typing import Dict, Tuple, Optional
from loguru import logger


class CompatibilityScorer:
    """
    Compute compatibility scores between user features and bot personas
    
    Scoring dimensions:
    - Personality match (Big Five): 40% weight
    - Interest overlap: 30% weight
    - Communication style: 20% weight
    - Relationship goals: 10% weight
    """
    
    # Feature vector indices (matching persona_builder.py)
    BIG_FIVE_START = 0
    BIG_FIVE_END = 5
    COMM_STYLE_START = 5
    COMM_STYLE_END = 11
    INTERESTS_START = 11
    INTERESTS_END = 19
    GOALS_START = 19
    GOALS_END = 23
    
    # Scoring weights
    PERSONALITY_WEIGHT = 0.40
    INTERESTS_WEIGHT = 0.30
    COMMUNICATION_WEIGHT = 0.20
    GOALS_WEIGHT = 0.10
    
    # Communication styles (one-hot encoded order)
    COMM_STYLES = ["direct", "indirect", "humorous", "serious", "casual", "formal"]
    
    # Relationship goals (one-hot encoded order)
    GOAL_TYPES = ["casual", "serious", "friendship", "unsure"]
    
    # Interest categories (order from persona_builder.py)
    INTEREST_CATEGORIES = ["music", "sports", "travel", "food", "arts", "tech", "outdoors", "books"]
    
    # Personality traits (Big Five order)
    BIG_FIVE_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    
    def __init__(self, 
                 personality_weight: float = 0.40,
                 interests_weight: float = 0.30,
                 communication_weight: float = 0.20,
                 goals_weight: float = 0.10):
        """
        Initialize compatibility scorer
        
        Args:
            personality_weight: Weight for personality matching (default 0.40)
            interests_weight: Weight for interest overlap (default 0.30)
            communication_weight: Weight for communication style (default 0.20)
            goals_weight: Weight for relationship goals (default 0.10)
        """
        # Validate weights sum to 1.0
        total_weight = personality_weight + interests_weight + communication_weight + goals_weight
        if not np.isclose(total_weight, 1.0):
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            norm = total_weight
            personality_weight /= norm
            interests_weight /= norm
            communication_weight /= norm
            goals_weight /= norm
        
        self.personality_weight = personality_weight
        self.interests_weight = interests_weight
        self.communication_weight = communication_weight
        self.goals_weight = goals_weight
    
    def compute_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors
        
        Args:
            vector1: First feature vector (normalized)
            vector2: Second feature vector (normalized)
        
        Returns:
            Cosine similarity in range [-1, 1]
        """
        # Handle edge cases
        if len(vector1) != len(vector2):
            logger.error(f"Vector length mismatch: {len(vector1)} vs {len(vector2)}")
            return 0.0
        
        if len(vector1) == 0:
            return 0.0
        
        # Compute cosine similarity
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Clamp to [-1, 1] to handle numerical errors
        similarity = np.clip(similarity, -1.0, 1.0)
        
        return float(similarity)
    
    def compute_personality_match(
        self, 
        user_vector: np.ndarray, 
        bot_vector: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute personality match score based on Big Five traits
        
        Uses complementarity for some traits (e.g., extraversion-introversion)
        and similarity for others (e.g., agreeableness, openness)
        
        Args:
            user_vector: User feature vector
            bot_vector: Bot feature vector
        
        Returns:
            Tuple of (overall_score, trait_scores_dict)
        """
        user_big_five = user_vector[self.BIG_FIVE_START:self.BIG_FIVE_END]
        bot_big_five = bot_vector[self.BIG_FIVE_START:self.BIG_FIVE_END]
        
        trait_scores = {}
        
        # Define matching strategy for each trait
        # True = similarity preferred, False = complementarity preferred
        similarity_preferred = {
            "openness": True,           # Similar openness works well
            "conscientiousness": True,  # Similar conscientiousness works well
            "extraversion": False,      # Complementarity can work (intro-extro balance)
            "agreeableness": True,      # Similar agreeableness works well
            "neuroticism": True         # Similar emotional stability preferred
        }
        
        for idx, trait in enumerate(self.BIG_FIVE_TRAITS):
            user_val = user_big_five[idx]
            bot_val = bot_big_five[idx]
            
            if similarity_preferred[trait]:
                # Similarity: 1 - normalized distance
                score = 1.0 - abs(user_val - bot_val)
            else:
                # Complementarity: favor moderate differences (not too extreme)
                diff = abs(user_val - bot_val)
                # Optimal at 0.3-0.5 difference, penalize extremes
                if 0.2 <= diff <= 0.6:
                    score = 1.0 - abs(diff - 0.4) / 0.4
                else:
                    score = 1.0 - diff
            
            trait_scores[trait] = float(score)
        
        # Weighted average (all traits equal for now)
        overall_score = np.mean(list(trait_scores.values()))
        
        return float(overall_score), trait_scores
    
    def compute_interest_overlap(
        self,
        user_vector: np.ndarray,
        bot_vector: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute interest overlap score
        
        Args:
            user_vector: User feature vector
            bot_vector: Bot feature vector
        
        Returns:
            Tuple of (overall_score, interest_scores_dict)
        """
        user_interests = user_vector[self.INTERESTS_START:self.INTERESTS_END]
        bot_interests = bot_vector[self.INTERESTS_START:self.INTERESTS_END]
        
        interest_scores = {}
        
        # Compute element-wise similarity
        for idx, interest in enumerate(self.INTEREST_CATEGORIES):
            user_val = user_interests[idx]
            bot_val = bot_interests[idx]
            
            # Both interested: high score
            # Neither interested: neutral (0.5)
            # One interested: low score
            if user_val > 0.3 and bot_val > 0.3:
                # Both interested: product of interest levels
                score = user_val * bot_val
            elif user_val < 0.3 and bot_val < 0.3:
                # Neither interested: neutral
                score = 0.5
            else:
                # Mismatch: penalize
                score = 0.3
            
            interest_scores[interest] = float(score)
        
        # Overall: weighted by interest strength
        interest_weights = user_interests + bot_interests
        interest_weights = interest_weights / (interest_weights.sum() + 1e-6)  # Avoid div by zero
        
        scores_array = np.array(list(interest_scores.values()))
        overall_score = np.sum(scores_array * interest_weights)
        
        return float(overall_score), interest_scores
    
    def compute_communication_match(
        self,
        user_vector: np.ndarray,
        bot_vector: np.ndarray
    ) -> Tuple[float, str, str]:
        """
        Compute communication style match
        
        Args:
            user_vector: User feature vector
            bot_vector: Bot feature vector
        
        Returns:
            Tuple of (match_score, user_style, bot_style)
        """
        user_comm = user_vector[self.COMM_STYLE_START:self.COMM_STYLE_END]
        bot_comm = bot_vector[self.COMM_STYLE_START:self.COMM_STYLE_END]
        
        # Get dominant styles
        user_style_idx = np.argmax(user_comm)
        bot_style_idx = np.argmax(bot_comm)
        
        user_style = self.COMM_STYLES[user_style_idx]
        bot_style = self.COMM_STYLES[bot_style_idx]
        
        # Define compatibility matrix
        # Some styles are compatible (casual-humorous), others clash (formal-casual)
        compatibility_matrix = {
            ("direct", "direct"): 0.9,
            ("direct", "indirect"): 0.3,
            ("direct", "humorous"): 0.7,
            ("direct", "serious"): 0.8,
            ("direct", "casual"): 0.7,
            ("direct", "formal"): 0.6,
            
            ("indirect", "indirect"): 0.8,
            ("indirect", "humorous"): 0.6,
            ("indirect", "serious"): 0.7,
            ("indirect", "casual"): 0.8,
            ("indirect", "formal"): 0.7,
            
            ("humorous", "humorous"): 0.95,
            ("humorous", "serious"): 0.4,
            ("humorous", "casual"): 0.9,
            ("humorous", "formal"): 0.3,
            
            ("serious", "serious"): 0.9,
            ("serious", "casual"): 0.5,
            ("serious", "formal"): 0.8,
            
            ("casual", "casual"): 0.95,
            ("casual", "formal"): 0.3,
            
            ("formal", "formal"): 0.85,
        }
        
        # Get compatibility score (symmetrical)
        pair = (user_style, bot_style)
        reverse_pair = (bot_style, user_style)
        
        score = compatibility_matrix.get(pair, compatibility_matrix.get(reverse_pair, 0.5))
        
        return float(score), user_style, bot_style
    
    def compute_goals_match(
        self,
        user_vector: np.ndarray,
        bot_vector: np.ndarray
    ) -> Tuple[float, str, str]:
        """
        Compute relationship goals match
        
        Args:
            user_vector: User feature vector
            bot_vector: Bot feature vector
        
        Returns:
            Tuple of (match_score, user_goal, bot_goal)
        """
        user_goals = user_vector[self.GOALS_START:self.GOALS_END]
        bot_goals = bot_vector[self.GOALS_START:self.GOALS_END]
        
        # Get dominant goals
        user_goal_idx = np.argmax(user_goals)
        bot_goal_idx = np.argmax(bot_goals)
        
        user_goal = self.GOAL_TYPES[user_goal_idx]
        bot_goal = self.GOAL_TYPES[bot_goal_idx]
        
        # Define compatibility matrix
        goals_compatibility = {
            ("casual", "casual"): 1.0,
            ("casual", "serious"): 0.2,
            ("casual", "friendship"): 0.6,
            ("casual", "unsure"): 0.7,
            
            ("serious", "serious"): 1.0,
            ("serious", "friendship"): 0.4,
            ("serious", "unsure"): 0.6,
            
            ("friendship", "friendship"): 0.95,
            ("friendship", "unsure"): 0.7,
            
            ("unsure", "unsure"): 0.8,
        }
        
        # Get compatibility score (symmetrical)
        pair = (user_goal, bot_goal)
        reverse_pair = (bot_goal, user_goal)
        
        score = goals_compatibility.get(pair, goals_compatibility.get(reverse_pair, 0.5))
        
        return float(score), user_goal, bot_goal
    
    def compute_compatibility(
        self,
        user_features: np.ndarray,
        bot_persona: np.ndarray
    ) -> Tuple[float, Dict[str, any]]:
        """
        Compute overall compatibility score between user and bot
        
        Args:
            user_features: User feature vector (23 dimensions)
            bot_persona: Bot persona feature vector (23 dimensions)
        
        Returns:
            Tuple of (overall_score [0-1], breakdown_dict)
        """
        # Validate input
        if len(user_features) != 23 or len(bot_persona) != 23:
            logger.error(f"Invalid feature vector length: user={len(user_features)}, bot={len(bot_persona)}")
            return 0.0, {}
        
        # Convert to numpy arrays
        user_vec = np.array(user_features)
        bot_vec = np.array(bot_persona)
        
        # Compute component scores
        personality_score, personality_breakdown = self.compute_personality_match(user_vec, bot_vec)
        interest_score, interest_breakdown = self.compute_interest_overlap(user_vec, bot_vec)
        comm_score, user_comm_style, bot_comm_style = self.compute_communication_match(user_vec, bot_vec)
        goals_score, user_goal, bot_goal = self.compute_goals_match(user_vec, bot_vec)
        
        # Weighted combination
        overall_score = (
            self.personality_weight * personality_score +
            self.interests_weight * interest_score +
            self.communication_weight * comm_score +
            self.goals_weight * goals_score
        )
        
        # Build breakdown
        breakdown = {
            "overall_score": float(overall_score),
            "components": {
                "personality": {
                    "score": float(personality_score),
                    "weight": self.personality_weight,
                    "weighted_score": float(self.personality_weight * personality_score),
                    "breakdown": personality_breakdown
                },
                "interests": {
                    "score": float(interest_score),
                    "weight": self.interests_weight,
                    "weighted_score": float(self.interests_weight * interest_score),
                    "breakdown": interest_breakdown
                },
                "communication": {
                    "score": float(comm_score),
                    "weight": self.communication_weight,
                    "weighted_score": float(self.communication_weight * comm_score),
                    "user_style": user_comm_style,
                    "bot_style": bot_comm_style
                },
                "goals": {
                    "score": float(goals_score),
                    "weight": self.goals_weight,
                    "weighted_score": float(self.goals_weight * goals_score),
                    "user_goal": user_goal,
                    "bot_goal": bot_goal
                }
            }
        }
        
        return float(overall_score), breakdown
