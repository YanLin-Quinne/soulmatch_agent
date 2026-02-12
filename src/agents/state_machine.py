"""Conversation state machine for orchestrator"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel
from datetime import datetime


class ConversationState(str, Enum):
    """Conversation states"""
    INIT = "INIT"  # Initial state, no conversation yet
    MATCHING = "MATCHING"  # Finding a match
    GREETING = "GREETING"  # Bot sends greeting
    ACTIVE = "ACTIVE"  # Active conversation
    FEATURE_UPDATE = "FEATURE_UPDATE"  # Updating user features
    MEMORY_UPDATE = "MEMORY_UPDATE"  # Memory operations
    SCAM_CHECK = "SCAM_CHECK"  # Scam detection check
    WARNING = "WARNING"  # Warning issued
    ENDED = "ENDED"  # Conversation ended


class ConversationContext(BaseModel):
    """Context for conversation state machine"""
    
    # State
    current_state: ConversationState = ConversationState.INIT
    
    # Participants
    user_id: str
    bot_id: Optional[str] = None
    
    # Conversation tracking
    turn_count: int = 0
    message_count: int = 0
    started_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None
    
    # Features
    user_features_initialized: bool = False
    last_feature_update_turn: int = 0
    
    # Memory
    last_memory_update_turn: int = 0
    
    # Emotion
    current_user_emotion: Optional[str] = None
    current_bot_emotion: Optional[str] = None
    emotion_history: list[str] = []
    
    # Scam detection
    scam_warnings_count: int = 0
    last_scam_check_turn: int = 0
    current_risk_level: str = "safe"
    
    # Matching
    compatibility_score: Optional[float] = None
    
    def update_state(self, new_state: ConversationState):
        """Update conversation state"""
        self.current_state = new_state
    
    def increment_turn(self):
        """Increment turn counter"""
        self.turn_count += 1
        self.message_count += 1
        self.last_message_at = datetime.now()
    
    def should_update_features(self, update_frequency: int = 3) -> bool:
        """Check if features should be updated"""
        # Update every N turns for first 30 turns
        if self.turn_count >= 30:
            return False
        return (self.turn_count - self.last_feature_update_turn) >= update_frequency
    
    def should_update_memory(self, update_frequency: int = 5) -> bool:
        """Check if memory should be updated"""
        return (self.turn_count - self.last_memory_update_turn) >= update_frequency
    
    def should_check_scam(self, check_frequency: int = 2) -> bool:
        """Check if scam detection should run"""
        return (self.turn_count - self.last_scam_check_turn) >= check_frequency
    
    def record_feature_update(self):
        """Record that features were updated"""
        self.last_feature_update_turn = self.turn_count
    
    def record_memory_update(self):
        """Record that memory was updated"""
        self.last_memory_update_turn = self.turn_count
    
    def record_scam_check(self, risk_level: str):
        """Record scam check"""
        self.last_scam_check_turn = self.turn_count
        self.current_risk_level = risk_level
    
    def add_emotion(self, emotion: str):
        """Add emotion to history"""
        self.emotion_history.append(emotion)
        # Keep last 20 emotions
        if len(self.emotion_history) > 20:
            self.emotion_history = self.emotion_history[-20:]
    
    def issue_warning(self):
        """Issue a scam warning"""
        self.scam_warnings_count += 1


class ConversationStateMachine:
    """State machine for managing conversation flow"""
    
    def __init__(self, user_id: str):
        self.context = ConversationContext(user_id=user_id)
    
    def start_conversation(self, bot_id: str, compatibility_score: float):
        """Start a new conversation"""
        self.context.bot_id = bot_id
        self.context.compatibility_score = compatibility_score
        self.context.started_at = datetime.now()
        self.context.update_state(ConversationState.GREETING)
    
    def handle_user_message(self) -> list[str]:
        """
        Handle incoming user message and determine next actions
        
        Returns:
            List of actions to perform (e.g., ["feature_update", "scam_check", "bot_response"])
        """
        actions = []
        
        # Increment turn
        self.context.increment_turn()
        
        # State transitions
        if self.context.current_state == ConversationState.GREETING:
            self.context.update_state(ConversationState.ACTIVE)
        
        # Check if features should be updated
        if self.context.should_update_features():
            actions.append("feature_update")
        
        # Check for scam signals
        if self.context.should_check_scam():
            actions.append("scam_check")
        
        # Check if memory should be updated
        if self.context.should_update_memory():
            actions.append("memory_update")
        
        # Always generate bot response
        actions.append("bot_response")
        
        # Emotion analysis
        actions.append("emotion_analysis")
        
        return actions
    
    def handle_bot_message(self):
        """Handle bot message sent"""
        self.context.increment_turn()
    
    def handle_scam_detected(self, risk_level: str):
        """Handle scam detection result"""
        self.context.record_scam_check(risk_level)
        
        if risk_level in ["high", "critical"]:
            self.context.update_state(ConversationState.WARNING)
            self.context.issue_warning()
    
    def handle_feature_updated(self):
        """Handle feature update completion"""
        self.context.record_feature_update()
        if not self.context.user_features_initialized:
            self.context.user_features_initialized = True
    
    def handle_memory_updated(self):
        """Handle memory update completion"""
        self.context.record_memory_update()
    
    def end_conversation(self):
        """End the conversation"""
        self.context.update_state(ConversationState.ENDED)
    
    def get_context(self) -> ConversationContext:
        """Get current context"""
        return self.context
    
    def reset(self):
        """Reset state machine"""
        user_id = self.context.user_id
        self.context = ConversationContext(user_id=user_id)
