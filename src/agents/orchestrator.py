"""Orchestrator Agent - Coordinates all sub-agents"""

from typing import Dict, List, Optional, Any
from loguru import logger

from src.agents.state_machine import ConversationStateMachine, ConversationState
from src.agents.persona_agent import PersonaAgent, PersonaAgentPool
from src.agents.feature_prediction_agent import FeaturePredictionAgent
from src.agents.emotion_agent import EmotionAgent
from src.agents.scam_detection_agent import ScamDetectionAgent
from src.memory.memory_manager import MemoryManager
from src.matching.matching_engine import MatchingEngine
from src.data.schema import PersonaProfile


class OrchestratorAgent:
    """
    Main orchestrator that coordinates all sub-agents
    
    Sub-agents:
    - PersonaAgent: Bot role-playing
    - FeaturePredictionAgent: User feature inference
    - MemoryManager: Memory management
    - EmotionAgent: Emotion detection and prediction
    - ScamDetectionAgent: Scam detection
    - MatchingEngine: Compatibility matching
    """
    
    def __init__(
        self,
        user_id: str,
        bot_personas_pool: PersonaAgentPool,
        use_claude: bool = True
    ):
        self.user_id = user_id
        
        # State machine
        self.state_machine = ConversationStateMachine(user_id)
        
        # Sub-agents
        self.bot_pool = bot_personas_pool
        self.current_bot: Optional[PersonaAgent] = None
        
        self.feature_agent = FeaturePredictionAgent(user_id, use_claude=use_claude)
        self.memory_manager = MemoryManager(user_id, use_llm=use_claude)
        self.emotion_agent = EmotionAgent(use_claude=use_claude)
        self.scam_agent = ScamDetectionAgent(use_semantic=True, use_claude=use_claude)
        self.matching_engine = MatchingEngine()
        
        # Conversation history (shared across agents)
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info(f"Orchestrator initialized for user {user_id}")
    
    def start_new_conversation(self) -> Dict[str, Any]:
        """
        Start a new conversation with a matched bot
        
        Returns:
            Dict with greeting and match info
        """
        
        # Get all bots
        bot_summaries = self.bot_pool.get_agent_summaries()
        
        if not bot_summaries:
            logger.error("No bot personas available")
            return {
                "success": False,
                "error": "No bots available"
            }
        
        # Get user features (might be empty initially)
        user_feature_summary = self.feature_agent.get_feature_summary()
        
        # Recommend a match
        # For initial match with no user features, use random or first bot
        if not user_feature_summary.get("personality"):
            # Random match for new users
            import random
            selected_bot_id = random.choice(list(bot_summaries.keys()))
            compatibility_score = 0.5
            match_explanation = "Let's see if you two click! 🎲"
        else:
            # Smart match based on features
            # Convert user features to vector (simplified)
            user_vector = self._convert_features_to_vector(
                self.feature_agent.predicted_features
            )
            
            bot_personas = [
                self.bot_pool.get_agent(bot_id).persona 
                for bot_id in bot_summaries.keys()
            ]
            
            recommendations = self.matching_engine.recommend_top_n(
                user_features=user_vector,
                candidates=bot_personas,
                n=1
            )
            
            if recommendations:
                selected_bot_id = recommendations[0]["profile_id"]
                compatibility_score = recommendations[0]["score"]
                match_explanation = recommendations[0]["explanation"]
            else:
                # Fallback
                selected_bot_id = list(bot_summaries.keys())[0]
                compatibility_score = 0.5
                match_explanation = "Let's give this a try! ✨"
        
        # Get bot agent
        self.current_bot = self.bot_pool.get_agent(selected_bot_id)
        
        # Start conversation in state machine
        self.state_machine.start_conversation(selected_bot_id, compatibility_score)
        
        # Generate greeting
        greeting = self.current_bot.generate_greeting()
        
        # Add to conversation history
        self._add_to_history("bot", greeting)
        
        logger.info(f"Started conversation with {selected_bot_id}, compatibility: {compatibility_score:.2f}")
        
        return {
            "success": True,
            "bot_id": selected_bot_id,
            "bot_profile": bot_summaries[selected_bot_id],
            "compatibility_score": compatibility_score,
            "match_explanation": match_explanation,
            "greeting": greeting
        }
    
    def process_user_message(self, message: str) -> Dict[str, Any]:
        """
        Process a user message and orchestrate all sub-agents
        
        Args:
            message: User's message
        
        Returns:
            Dict with bot response and agent outputs
        """
        
        if not self.current_bot:
            return {
                "success": False,
                "error": "No active conversation. Call start_new_conversation() first."
            }
        
        # Add user message to history
        self._add_to_history("user", message)
        self.memory_manager.add_conversation_turn("user", message)
        
        # Get actions from state machine
        actions = self.state_machine.handle_user_message()
        
        logger.info(f"Processing user message, actions: {actions}")
        
        # Initialize response dict
        response = {
            "success": True,
            "turn": self.state_machine.context.turn_count,
        }
        
        # Execute actions
        for action in actions:
            if action == "scam_check":
                scam_result = self._check_scam(message)
                response["scam_detection"] = scam_result
                
                # Handle scam detection
                if scam_result["warning_level"] in ["high", "critical"]:
                    self.state_machine.handle_scam_detected(scam_result["warning_level"])
                    response["warning"] = scam_result.get("message", {}).get("en", "")
            
            elif action == "feature_update":
                feature_result = self._update_features()
                response["feature_update"] = feature_result
                self.state_machine.handle_feature_updated()
            
            elif action == "memory_update":
                memory_result = self._update_memory()
                response["memory_update"] = memory_result
                self.state_machine.handle_memory_updated()
            
            elif action == "emotion_analysis":
                emotion_result = self._analyze_emotion(message)
                response["emotion"] = emotion_result
                
                # Update state machine
                if emotion_result.get("current_emotion"):
                    self.state_machine.context.current_user_emotion = emotion_result["current_emotion"]["emotion"]
                    self.state_machine.context.add_emotion(emotion_result["current_emotion"]["emotion"])
            
            elif action == "bot_response":
                bot_response = self._generate_bot_response(message)
                response["bot_message"] = bot_response
                
                # Add to history
                self._add_to_history("bot", bot_response)
                self.memory_manager.add_conversation_turn("bot", bot_response)
                
                self.state_machine.handle_bot_message()
        
        # Add context
        response["context"] = {
            "state": self.state_machine.context.current_state,
            "turn_count": self.state_machine.context.turn_count,
            "risk_level": self.state_machine.context.current_risk_level,
            "user_emotion": self.state_machine.context.current_user_emotion,
        }
        
        return response
    
    def _add_to_history(self, speaker: str, message: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            "speaker": speaker,
            "message": message,
            "turn": self.state_machine.context.turn_count
        })
        
        # Keep last 50 messages
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
    
    def _check_scam(self, message: str) -> Dict[str, Any]:
        """Run scam detection"""
        result = self.scam_agent.analyze_message(message)
        logger.info(f"Scam check: risk={result['risk_score']:.2f}, level={result['warning_level']}")
        return result
    
    def _update_features(self) -> Dict[str, Any]:
        """Update user features"""
        result = self.feature_agent.predict_from_conversation(self.conversation_history)
        logger.info(f"Feature update: turn={result['turn']}, confidence={result.get('confidences', {})}")
        return result
    
    def _update_memory(self) -> Dict[str, Any]:
        """Update memory"""
        # Get recent messages
        recent_messages = self.conversation_history[-5:]
        
        # Get current features
        current_features = self.feature_agent.predicted_features
        
        # Decide memory action
        action = self.memory_manager.decide_memory_action(
            recent_messages=recent_messages,
            current_features=current_features
        )
        
        # Execute action
        memories = self.memory_manager.execute_action(action)
        
        result = {
            "operation": action.operation,
            "reasoning": action.reasoning,
            "memories_affected": len(memories) if memories else 0
        }
        
        logger.info(f"Memory update: operation={action.operation}")
        return result
    
    def _analyze_emotion(self, message: str) -> Dict[str, Any]:
        """Analyze emotion"""
        result = self.emotion_agent.analyze_message(message)
        logger.info(f"Emotion: {result.get('current_emotion', {}).get('emotion', 'unknown')}")
        return result
    
    def _generate_bot_response(self, user_message: str) -> str:
        """Generate bot response"""
        
        # Get emotion context
        emotion_context = None
        if self.state_machine.context.current_user_emotion:
            emotion_result = self.emotion_agent.suggest_reply_strategy(
                self.state_machine.context.current_user_emotion
            )
            emotion_context = emotion_result.get("approach", "")
        
        # Generate response
        response = self.current_bot.generate_response(
            user_message=user_message,
            emotion_hint=emotion_context
        )
        
        return response
    
    def _convert_features_to_vector(self, features: Dict[str, Any]) -> Optional[list[float]]:
        """Convert predicted features to feature vector"""
        
        # This is a simplified conversion
        # In practice, should match PersonaBuilder.create_feature_vector format
        
        vector = []
        
        # Big Five (5 dims)
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            key = f"big_five_{trait}"
            vector.append(features.get(key, 0.5))
        
        # Communication style (6 dims, one-hot)
        comm_style = features.get("communication_style", "casual")
        styles = ["direct", "indirect", "humorous", "serious", "casual", "formal"]
        for style in styles:
            vector.append(1.0 if comm_style == style else 0.0)
        
        # Interests (8 dims)
        interests = ["music", "sports", "travel", "food", "arts", "tech", "outdoors", "books"]
        for interest in interests:
            key = f"interest_{interest}"
            vector.append(features.get(key, 0.0))
        
        # Relationship goals (4 dims, one-hot)
        goal = features.get("relationship_goals", "unsure")
        goals = ["casual", "serious", "friendship", "unsure"]
        for g in goals:
            vector.append(1.0 if goal == g else 0.0)
        
        # Normalize
        import numpy as np
        vector_array = np.array(vector)
        norm = np.linalg.norm(vector_array)
        if norm > 0:
            vector_array = vector_array / norm
        
        return vector_array.tolist() if len(vector) == 23 else None
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation"""
        
        return {
            "user_id": self.user_id,
            "bot_id": self.state_machine.context.bot_id,
            "state": self.state_machine.context.current_state,
            "turn_count": self.state_machine.context.turn_count,
            "message_count": len(self.conversation_history),
            "compatibility_score": self.state_machine.context.compatibility_score,
            "risk_level": self.state_machine.context.current_risk_level,
            "warnings_count": self.state_machine.context.scam_warnings_count,
            "user_features": self.feature_agent.get_feature_summary(),
            "emotion_history": self.state_machine.context.emotion_history,
        }
    
    def reset_conversation(self):
        """Reset current conversation"""
        self.state_machine.reset()
        self.conversation_history = []
        self.current_bot = None
        self.memory_manager = MemoryManager(self.user_id)
        logger.info(f"Conversation reset for user {self.user_id}")
