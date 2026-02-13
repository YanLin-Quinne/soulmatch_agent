"""Unit tests for individual agents"""

import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.emotion_agent import EmotionAgent
from src.agents.scam_detection_agent import ScamDetectionAgent
from src.memory.memory_operations import Memory, MemoryAction, MemoryOperation
from src.memory.memory_manager import MemoryManager


class TestEmotionAgent:
    """Test EmotionAgent functionality"""
    
    @pytest.fixture
    def emotion_agent(self):
        agent = EmotionAgent(use_claude=False)
        return agent
    
    def test_emotion_detection_fallback(self, emotion_agent):
        """Test keyword-based fallback emotion detection"""
        
        # Test joy
        result = emotion_agent._detect_emotion_from_keywords("I'm so happy and excited!")
        assert result["emotion"] == "joy"
        
        # Test sadness
        result = emotion_agent._detect_emotion_from_keywords("I feel sad and lonely")
        assert result["emotion"] == "sadness"
        
        # Test anger
        result = emotion_agent._detect_emotion_from_keywords("I'm so angry and frustrated!")
        assert result["emotion"] == "anger"
        
        # Test neutral
        result = emotion_agent._detect_emotion_from_keywords("The weather is normal today")
        assert result["emotion"] == "neutral"


class TestScamDetectionAgent:
    """Test ScamDetectionAgent functionality"""
    
    @pytest.fixture
    def scam_agent(self):
        return ScamDetectionAgent(use_semantic=False)
    
    def test_love_bombing_detection(self, scam_agent):
        """Test love bombing pattern detection"""
        
        message = "You are my soulmate, my everything, my destiny!"
        result = scam_agent.analyze_message(message)
        
        assert "LOVE_BOMBING" in [p.name for p in result["detected_patterns"]]
        assert result["risk_score"] > 0.0
    
    def test_money_request_detection(self, scam_agent):
        """Test money request pattern detection"""
        
        message = "Can you send me money? I need $500 wire transfer urgently."
        result = scam_agent.analyze_message(message)
        
        detected = [p.name for p in result["detected_patterns"]]
        assert "MONEY_REQUEST" in detected or "URGENCY_PRESSURE" in detected
        assert result["risk_score"] > 0.0
    
    def test_external_link_detection(self, scam_agent):
        """Test external link detection"""
        
        message = "Add me on WhatsApp: +1234567890"
        result = scam_agent.analyze_message(message)
        
        assert "EXTERNAL_LINKS" in [p.name for p in result["detected_patterns"]]
    
    def test_safe_message(self, scam_agent):
        """Test safe message with no scam signals"""
        
        message = "Hey, how was your day? I went hiking today."
        result = scam_agent.analyze_message(message)
        
        assert result["warning_level"] == "safe"
        assert result["risk_score"] < 0.3


class TestMemoryOperations:
    """Test memory operations and data models"""
    
    def test_memory_creation(self):
        """Test Memory object creation"""
        
        memory = Memory(
            memory_id="test_123",
            content="User likes hiking",
            memory_type="fact",
            importance=0.8
        )
        
        assert memory.memory_id == "test_123"
        assert memory.content == "User likes hiking"
        assert memory.importance == 0.8
    
    def test_memory_action_validation(self):
        """Test MemoryAction validation"""
        
        # Valid ADD action
        action = MemoryAction(
            operation=MemoryOperation.ADD,
            content="Test memory",
            memory_type="fact"
        )
        assert action.validate_action() is True
        
        # Invalid ADD (missing content)
        action = MemoryAction(
            operation=MemoryOperation.ADD,
            memory_type="fact"
        )
        assert action.validate_action() is False
        
        # Valid UPDATE
        action = MemoryAction(
            operation=MemoryOperation.UPDATE,
            memory_id="123",
            new_content="Updated content"
        )
        assert action.validate_action() is True
        
        # Valid NOOP
        action = MemoryAction(operation=MemoryOperation.NOOP)
        assert action.validate_action() is True


class TestMemoryManager:
    """Test MemoryManager"""
    
    @pytest.fixture
    def memory_manager(self):
        # Mock ChromaDB to avoid actual database operations
        with patch('src.memory.memory_manager.ChromaDBClient'):
            manager = MemoryManager(user_id="test_user", use_llm=False)
            return manager
    
    def test_conversation_tracking(self, memory_manager):
        """Test conversation history tracking"""
        
        memory_manager.add_conversation_turn("user", "Hello")
        memory_manager.add_conversation_turn("bot", "Hi there!")
        
        assert memory_manager.current_turn == 2
        assert len(memory_manager.conversation_history) == 2
    
    def test_simple_memory_decision(self, memory_manager):
        """Test simple heuristic memory decision"""
        
        # Should add memory every 5 turns
        memory_manager.current_turn = 5
        
        recent_messages = [
            {"speaker": "user", "message": "I love traveling"}
        ]
        
        action = memory_manager.decide_memory_action(recent_messages)
        
        assert action.operation == MemoryOperation.ADD
        assert action.content is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
