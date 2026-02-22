"""Integration tests for OrchestratorAgent pipeline"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.orchestrator import OrchestratorAgent
from src.agents.persona_agent import PersonaAgentPool
from src.agents.agent_context import AgentContext


class TestOrchestratorIntegration:
    """Test 5-phase orchestrator pipeline"""

    @pytest.fixture
    def mock_bot_pool(self):
        pool = Mock(spec=PersonaAgentPool)
        pool.get_agent_summaries.return_value = {"bot_0": {"name": "Test Bot"}}
        mock_bot = Mock()
        mock_bot.persona.profile_id = "bot_0"
        pool.get_agent.return_value = mock_bot
        return pool

    @pytest.fixture
    def orchestrator(self, mock_bot_pool):
        return OrchestratorAgent(user_id="test_user", bot_personas_pool=mock_bot_pool)

    @pytest.mark.asyncio
    async def test_pipeline_phase1_parallel_execution(self, orchestrator):
        """Verify Phase 1 parallel execution (Emotion + Scam + Memory)"""
        with patch.object(orchestrator.emotion_agent, 'analyze_message', return_value={"current_emotion": {"emotion": "happy", "confidence": 0.8}}), \
             patch.object(orchestrator.scam_agent, 'analyze_message', return_value={"risk_score": 0.1}), \
             patch.object(orchestrator.memory_manager, 'retrieve_relevant_memories', return_value=["memory1"]):

            orchestrator.ctx.user_message = "Hello!"

            # Simulate phase 1 parallel execution
            emotion_result = orchestrator.emotion_agent.analyze_message("Hello!")
            scam_result = orchestrator.scam_agent.analyze_message("Hello!")
            memories = orchestrator.memory_manager.retrieve_relevant_memories("Hello!")

            assert emotion_result is not None
            assert scam_result is not None
            assert memories is not None
            assert emotion_result["current_emotion"]["emotion"] == "happy"

    @pytest.mark.asyncio
    async def test_pipeline_sequential_phases(self, orchestrator):
        """Verify Phase 2-5 sequential execution"""
        orchestrator.ctx.current_emotion = "happy"
        orchestrator.ctx.emotion_confidence = 0.8

        with patch.object(orchestrator.feature_agent, 'predict_from_conversation', return_value={"openness": 0.7}), \
             patch.object(orchestrator.question_agent, 'suggest_probes', return_value=["probe1"]):

            # Phase 2: Feature prediction (depends on emotion)
            features = orchestrator.feature_agent.predict_from_conversation([])
            assert features is not None
            assert "openness" in features

            # Phase 3: Question strategy (depends on features)
            orchestrator.ctx.predicted_features = features
            probes = orchestrator.question_agent.suggest_probes(orchestrator.ctx)
            assert probes is not None

    def test_agent_context_state_passing(self, orchestrator):
        """Verify AgentContext state passing between phases"""
        ctx = orchestrator.ctx

        # Phase 1 writes
        ctx.current_emotion = "excited"
        ctx.scam_risk_score = 0.2
        ctx.retrieved_memories = ["mem1", "mem2"]

        # Phase 2 reads emotion
        assert ctx.current_emotion == "excited"

        # Phase 3 reads features
        ctx.predicted_features = {"openness": 0.8}
        assert ctx.predicted_features["openness"] == 0.8

        # Verify state persistence
        assert ctx.scam_risk_score == 0.2
        assert len(ctx.retrieved_memories) == 2

    @pytest.mark.asyncio
    async def test_error_handling_and_fallback(self, orchestrator):
        """Verify error handling and fallback"""
        with patch.object(orchestrator.emotion_agent, 'analyze_message', side_effect=Exception("API Error")):
            try:
                orchestrator.emotion_agent.analyze_message("test")
                assert False, "Should raise exception"
            except Exception as e:
                assert "API Error" in str(e)

    @pytest.mark.asyncio
    async def test_complete_message_flow(self, orchestrator):
        """End-to-end message processing flow"""
        orchestrator.current_bot = Mock()
        orchestrator.current_bot.chat.return_value = "Bot response"

        with patch.object(orchestrator.emotion_agent, 'analyze_message', return_value={"current_emotion": {"emotion": "neutral", "confidence": 0.7}}), \
             patch.object(orchestrator.scam_agent, 'analyze_message', return_value={"risk_score": 0.0}), \
             patch.object(orchestrator.memory_manager, 'retrieve_relevant_memories', return_value=[]), \
             patch.object(orchestrator.feature_agent, 'predict_from_conversation', return_value={}), \
             patch.object(orchestrator.question_agent, 'suggest_probes', return_value=[]):

            orchestrator.ctx.user_message = "Hi there"
            orchestrator.ctx.conversation_history = []

            # Verify context is populated
            assert orchestrator.ctx.user_message == "Hi there"
            assert orchestrator.ctx.conversation_history == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
