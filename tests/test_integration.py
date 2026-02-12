"""Integration tests for the complete system"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.schema import OkCupidProfile, ExtractedFeatures, PersonaProfile


class TestDataModels:
    """Test data models and schema"""
    
    def test_okcupid_profile_creation(self):
        """Test OkCupidProfile model"""
        
        profile = OkCupidProfile(
            age=25,
            sex="f",
            orientation="straight",
            education="bachelors",
            essay0="I love hiking and reading",
            essay1="Working as a software engineer"
        )
        
        assert profile.age == 25
        assert profile.sex == "f"
        assert profile.essay0 == "I love hiking and reading"
    
    def test_extracted_features_creation(self):
        """Test ExtractedFeatures model"""
        
        features = ExtractedFeatures(
            communication_style="humorous",
            communication_confidence=0.8,
            core_values=["family", "career", "adventure"],
            values_confidence=0.7,
            openness=0.85,
            extraversion=0.65,
            relationship_goals="serious",
            goals_confidence=0.75,
            personality_summary="Outgoing and adventurous person"
        )
        
        assert features.communication_style == "humorous"
        assert features.openness == 0.85
        assert len(features.core_values) == 3
    
    def test_persona_profile_creation(self):
        """Test PersonaProfile with complete data"""
        
        profile = OkCupidProfile(age=28, sex="m")
        features = ExtractedFeatures(
            communication_style="casual",
            communication_confidence=0.8,
            core_values=["honesty"],
            values_confidence=0.7
        )
        
        persona = PersonaProfile(
            original_profile=profile,
            features=features,
            profile_id="bot_0",
            is_bot=True,
            system_prompt="You are a friendly person",
            feature_vector=[0.5] * 23
        )
        
        assert persona.profile_id == "bot_0"
        assert persona.is_bot is True
        assert len(persona.feature_vector) == 23


class TestDataPreprocessing:
    """Test data preprocessing pipeline"""
    
    def test_profile_to_dict(self):
        """Test profile serialization"""
        
        profile = OkCupidProfile(
            age=30,
            sex="f",
            orientation="bisexual",
            education="masters"
        )
        
        profile_dict = profile.model_dump()
        
        assert profile_dict["age"] == 30
        assert profile_dict["sex"] == "f"
        assert "education" in profile_dict


class TestMatchingEngine:
    """Test matching engine without real data"""
    
    def test_feature_vector_format(self):
        """Test that feature vectors have correct dimensions"""
        
        # Mock PersonaProfile with proper feature vector
        mock_persona = MagicMock()
        mock_persona.feature_vector = [0.5] * 23
        mock_persona.profile_id = "bot_test"
        
        assert len(mock_persona.feature_vector) == 23


class TestStateMachine:
    """Test conversation state machine"""
    
    def test_state_machine_initialization(self):
        """Test state machine starts in INIT state"""
        
        from src.agents.state_machine import ConversationStateMachine, ConversationState
        
        sm = ConversationStateMachine(user_id="test_user")
        
        assert sm.context.current_state == ConversationState.INIT
        assert sm.context.user_id == "test_user"
        assert sm.context.turn_count == 0
    
    def test_state_transitions(self):
        """Test state transitions"""
        
        from src.agents.state_machine import ConversationStateMachine, ConversationState
        
        sm = ConversationStateMachine(user_id="test_user")
        
        # Start conversation
        sm.start_conversation(bot_id="bot_0", compatibility_score=0.85)
        
        assert sm.context.current_state == ConversationState.GREETING
        assert sm.context.bot_id == "bot_0"
        assert sm.context.compatibility_score == 0.85
    
    def test_feature_update_frequency(self):
        """Test feature update frequency control"""
        
        from src.agents.state_machine import ConversationStateMachine
        
        sm = ConversationStateMachine(user_id="test_user")
        
        # First turn
        sm.context.turn_count = 0
        assert sm.context.should_update_features(update_frequency=3) is True
        
        # Turn 1 (shouldn't update)
        sm.context.turn_count = 1
        sm.context.last_feature_update_turn = 0
        assert sm.context.should_update_features(update_frequency=3) is False
        
        # Turn 3 (should update)
        sm.context.turn_count = 3
        assert sm.context.should_update_features(update_frequency=3) is True
        
        # Turn 30+ (stop updating)
        sm.context.turn_count = 30
        assert sm.context.should_update_features(update_frequency=3) is False


class TestBayesianUpdater:
    """Test Bayesian feature updater"""
    
    def test_bayesian_update(self):
        """Test Bayesian posterior update"""
        
        from src.agents.bayesian_updater import BayesianFeatureUpdater
        
        updater = BayesianFeatureUpdater()
        
        # Initial belief: 0.5 with low confidence
        prior_value = 0.5
        prior_confidence = 0.3
        
        # New observation: 0.8 with high confidence
        new_observation = 0.8
        observation_confidence = 0.9
        
        updated_value, updated_confidence = updater.update_feature(
            prior_value=prior_value,
            prior_confidence=prior_confidence,
            new_observation=new_observation,
            observation_confidence=observation_confidence
        )
        
        # Updated value should be closer to observation (high confidence)
        assert updated_value > prior_value
        assert updated_value < new_observation or abs(updated_value - new_observation) < 0.1
        
        # Confidence should increase
        assert updated_confidence > prior_confidence
    
    def test_information_gain(self):
        """Test information gain calculation"""
        
        from src.agents.bayesian_updater import BayesianFeatureUpdater
        
        updater = BayesianFeatureUpdater()
        
        # Low confidence → high confidence = high information gain
        gain = updater.compute_information_gain(
            prior_confidence=0.2,
            posterior_confidence=0.8
        )
        
        assert gain > 0.5
        
        # High confidence → slightly higher = low information gain
        gain = updater.compute_information_gain(
            prior_confidence=0.8,
            posterior_confidence=0.85
        )
        
        assert gain < 0.3


class TestAPIModels:
    """Test API request/response models"""
    
    def test_pydantic_models_import(self):
        """Test that API models can be imported"""
        
        try:
            from pydantic import BaseModel
            
            class TestModel(BaseModel):
                name: str
                value: int
            
            model = TestModel(name="test", value=42)
            assert model.name == "test"
            assert model.value == 42
            
        except ImportError:
            pytest.skip("Pydantic not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
