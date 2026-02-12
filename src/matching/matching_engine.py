"""Matching Engine - Recommend best bot personas for users"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from loguru import logger

from src.matching.compatibility_scorer import CompatibilityScorer
from src.data.schema import PersonaProfile


class MatchingEngine:
    """
    Matching engine for recommending compatible bot personas
    
    Features:
    - Rank bot personas by compatibility score
    - Recommend top N matches
    - Explain why matches are compatible
    - Support for batch scoring
    """
    
    def __init__(self, 
                 scorer: Optional[CompatibilityScorer] = None,
                 min_score_threshold: float = 0.3):
        """
        Initialize matching engine
        
        Args:
            scorer: Custom CompatibilityScorer (if None, uses default)
            min_score_threshold: Minimum compatibility score to recommend (default 0.3)
        """
        self.scorer = scorer if scorer is not None else CompatibilityScorer()
        self.min_score_threshold = min_score_threshold
        
        # Learning history (for future adaptive weighting)
        self.match_history: List[Dict] = []
    
    def rank_candidates(
        self,
        user_features: np.ndarray,
        bot_personas: List[PersonaProfile],
        exclude_ids: Optional[List[str]] = None
    ) -> List[Tuple[PersonaProfile, float, Dict]]:
        """
        Rank bot personas by compatibility score
        
        Args:
            user_features: User feature vector (23 dimensions)
            bot_personas: List of bot PersonaProfile objects
            exclude_ids: List of profile IDs to exclude from ranking
        
        Returns:
            List of (persona, score, breakdown) tuples, sorted by score descending
        """
        if len(bot_personas) == 0:
            logger.warning("No bot personas provided for ranking")
            return []
        
        # Filter excluded personas
        if exclude_ids:
            bot_personas = [p for p in bot_personas if p.profile_id not in exclude_ids]
        
        if len(bot_personas) == 0:
            logger.warning("All personas excluded from ranking")
            return []
        
        # Score each persona
        ranked_matches = []
        
        for persona in bot_personas:
            if persona.feature_vector is None:
                logger.warning(f"Persona {persona.profile_id} has no feature vector, skipping")
                continue
            
            # Compute compatibility
            score, breakdown = self.scorer.compute_compatibility(
                user_features=user_features,
                bot_persona=np.array(persona.feature_vector)
            )
            
            ranked_matches.append((persona, score, breakdown))
        
        # Sort by score descending
        ranked_matches.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Ranked {len(ranked_matches)} personas (scores: {[f'{s:.2f}' for _, s, _ in ranked_matches[:5]]})")
        
        return ranked_matches
    
    def recommend_top_n(
        self,
        user_features: np.ndarray,
        bot_personas: List[PersonaProfile],
        n: int = 3,
        exclude_ids: Optional[List[str]] = None
    ) -> List[Tuple[PersonaProfile, float, Dict]]:
        """
        Recommend top N compatible bot personas
        
        Args:
            user_features: User feature vector (23 dimensions)
            bot_personas: List of bot PersonaProfile objects
            n: Number of recommendations (default 3)
            exclude_ids: List of profile IDs to exclude
        
        Returns:
            List of top N (persona, score, breakdown) tuples
        """
        # Get ranked list
        ranked_matches = self.rank_candidates(
            user_features=user_features,
            bot_personas=bot_personas,
            exclude_ids=exclude_ids
        )
        
        # Filter by minimum threshold
        filtered_matches = [
            (persona, score, breakdown)
            for persona, score, breakdown in ranked_matches
            if score >= self.min_score_threshold
        ]
        
        if len(filtered_matches) < n:
            logger.warning(
                f"Only {len(filtered_matches)} personas meet threshold {self.min_score_threshold}, "
                f"requested {n}"
            )
        
        # Return top N
        top_n = filtered_matches[:n]
        
        logger.info(f"Recommended {len(top_n)} personas: {[(p.profile_id, f'{s:.2f}') for p, s, _ in top_n]}")
        
        return top_n
    
    def explain_match(
        self,
        user_features: np.ndarray,
        bot_persona: PersonaProfile
    ) -> str:
        """
        Generate human-readable explanation for why this match is compatible
        
        Args:
            user_features: User feature vector (23 dimensions)
            bot_persona: Bot PersonaProfile
        
        Returns:
            Explanation string
        """
        if bot_persona.feature_vector is None:
            return "Unable to explain match: bot persona has no feature vector"
        
        # Compute compatibility
        score, breakdown = self.scorer.compute_compatibility(
            user_features=user_features,
            bot_persona=np.array(bot_persona.feature_vector)
        )
        
        components = breakdown.get("components", {})
        
        # Build explanation parts
        explanation_parts = []
        
        # Overall score
        score_pct = int(score * 100)
        explanation_parts.append(f"**Compatibility: {score_pct}%**\n")
        
        # Personality
        personality = components.get("personality", {})
        personality_score = personality.get("score", 0.0)
        personality_breakdown = personality.get("breakdown", {})
        
        if personality_score > 0.7:
            # Find strongest matching traits
            top_traits = sorted(
                personality_breakdown.items(),
                key=lambda x: x[1],
                reverse=True
            )[:2]
            trait_names = [self._trait_name(trait) for trait, _ in top_traits]
            explanation_parts.append(
                f"✨ **Great personality match** - You both share similar {' and '.join(trait_names)}."
            )
        elif personality_score > 0.5:
            explanation_parts.append(
                f"😊 **Good personality compatibility** - Your personalities complement each other well."
            )
        else:
            explanation_parts.append(
                f"🔄 **Balanced personalities** - You bring different perspectives to conversations."
            )
        
        # Interests
        interests = components.get("interests", {})
        interest_score = interests.get("score", 0.0)
        interest_breakdown = interests.get("breakdown", {})
        
        # Find shared interests (both > 0.5)
        shared_interests = [
            interest for interest, score in interest_breakdown.items()
            if score > 0.6
        ]
        
        if len(shared_interests) >= 2:
            interest_list = ", ".join(shared_interests[:3])
            explanation_parts.append(
                f"🎯 **Shared interests** - You both enjoy {interest_list}!"
            )
        elif len(shared_interests) == 1:
            explanation_parts.append(
                f"🎯 **Common ground** - You share an interest in {shared_interests[0]}."
            )
        else:
            explanation_parts.append(
                f"🌟 **Diverse interests** - You can learn new things from each other."
            )
        
        # Communication
        communication = components.get("communication", {})
        comm_score = communication.get("score", 0.0)
        user_style = communication.get("user_style", "unknown")
        bot_style = communication.get("bot_style", "unknown")
        
        if comm_score > 0.8:
            if user_style == bot_style:
                explanation_parts.append(
                    f"💬 **Communication style match** - You both have a {user_style} style."
                )
            else:
                explanation_parts.append(
                    f"💬 **Compatible communication** - Your {user_style} and their {bot_style} "
                    f"styles work well together."
                )
        elif comm_score < 0.5:
            explanation_parts.append(
                f"🗣️ **Different communication styles** - Your {user_style} and their {bot_style} "
                f"styles may need some adjustment."
            )
        
        # Relationship goals
        goals = components.get("goals", {})
        goals_score = goals.get("score", 0.0)
        user_goal = goals.get("user_goal", "unknown")
        bot_goal = goals.get("bot_goal", "unknown")
        
        if goals_score > 0.8:
            if user_goal == bot_goal:
                explanation_parts.append(
                    f"🎯 **Aligned goals** - You're both looking for {user_goal} connections."
                )
            else:
                explanation_parts.append(
                    f"✅ **Compatible goals** - Your {user_goal} and their {bot_goal} "
                    f"goals can work together."
                )
        elif goals_score < 0.5:
            explanation_parts.append(
                f"⚠️ **Different goals** - You're seeking {user_goal} while they want {bot_goal}."
            )
        
        # Add persona summary
        if bot_persona.features.personality_summary:
            summary = bot_persona.features.personality_summary[:200]
            if len(bot_persona.features.personality_summary) > 200:
                summary += "..."
            explanation_parts.append(f"\n**About them:**\n{summary}")
        
        return "\n\n".join(explanation_parts)
    
    def batch_score(
        self,
        user_features: np.ndarray,
        bot_personas: List[PersonaProfile]
    ) -> Dict[str, float]:
        """
        Batch score multiple bot personas (optimized for performance)
        
        Args:
            user_features: User feature vector (23 dimensions)
            bot_personas: List of bot PersonaProfile objects
        
        Returns:
            Dict mapping profile_id to compatibility score
        """
        scores = {}
        
        for persona in bot_personas:
            if persona.feature_vector is None:
                continue
            
            score, _ = self.scorer.compute_compatibility(
                user_features=user_features,
                bot_persona=np.array(persona.feature_vector)
            )
            
            scores[persona.profile_id] = score
        
        return scores
    
    def record_match_outcome(
        self,
        user_id: str,
        bot_id: str,
        compatibility_score: float,
        conversation_success: bool,
        conversation_length: int
    ):
        """
        Record match outcome for learning (future adaptive weighting)
        
        Args:
            user_id: User identifier
            bot_id: Bot persona identifier
            compatibility_score: Predicted compatibility score
            conversation_success: Whether conversation was successful
            conversation_length: Number of messages exchanged
        """
        outcome = {
            "user_id": user_id,
            "bot_id": bot_id,
            "predicted_score": compatibility_score,
            "success": conversation_success,
            "conversation_length": conversation_length
        }
        
        self.match_history.append(outcome)
        
        logger.info(
            f"Recorded match outcome: user={user_id}, bot={bot_id}, "
            f"predicted={compatibility_score:.2f}, success={conversation_success}, "
            f"length={conversation_length}"
        )
    
    def get_match_statistics(self) -> Dict[str, any]:
        """
        Get statistics about match history (for monitoring)
        
        Returns:
            Dict with match statistics
        """
        if len(self.match_history) == 0:
            return {
                "total_matches": 0,
                "success_rate": 0.0,
                "avg_conversation_length": 0.0
            }
        
        total_matches = len(self.match_history)
        successful_matches = sum(1 for m in self.match_history if m["success"])
        total_length = sum(m["conversation_length"] for m in self.match_history)
        
        return {
            "total_matches": total_matches,
            "success_rate": successful_matches / total_matches,
            "avg_conversation_length": total_length / total_matches,
            "successful_matches": successful_matches
        }
    
    def _trait_name(self, trait: str) -> str:
        """Convert trait key to readable name"""
        trait_names = {
            "openness": "openness to new experiences",
            "conscientiousness": "conscientiousness",
            "extraversion": "extraversion",
            "agreeableness": "agreeableness",
            "neuroticism": "emotional stability"
        }
        return trait_names.get(trait, trait)
    
    def update_scoring_weights(
        self,
        personality_weight: Optional[float] = None,
        interests_weight: Optional[float] = None,
        communication_weight: Optional[float] = None,
        goals_weight: Optional[float] = None
    ):
        """
        Update scoring weights based on learning
        
        Args:
            personality_weight: New weight for personality (optional)
            interests_weight: New weight for interests (optional)
            communication_weight: New weight for communication (optional)
            goals_weight: New weight for goals (optional)
        """
        # Create new scorer with updated weights
        new_personality = personality_weight if personality_weight is not None else self.scorer.personality_weight
        new_interests = interests_weight if interests_weight is not None else self.scorer.interests_weight
        new_communication = communication_weight if communication_weight is not None else self.scorer.communication_weight
        new_goals = goals_weight if goals_weight is not None else self.scorer.goals_weight
        
        self.scorer = CompatibilityScorer(
            personality_weight=new_personality,
            interests_weight=new_interests,
            communication_weight=new_communication,
            goals_weight=new_goals
        )
        
        logger.info(
            f"Updated scoring weights: personality={new_personality:.2f}, "
            f"interests={new_interests:.2f}, communication={new_communication:.2f}, "
            f"goals={new_goals:.2f}"
        )
