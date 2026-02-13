"""Agent modules"""

from src.agents.llm_router import LLMRouter, router, AgentRole, Provider
from src.agents.agent_context import AgentContext
from src.agents.orchestrator import OrchestratorAgent
from src.agents.persona_agent import PersonaAgent, PersonaAgentPool, create_agent_pool_from_file
from src.agents.feature_prediction_agent import FeaturePredictionAgent
from src.agents.emotion_agent import EmotionAgent, EmotionDetector
from src.agents.scam_detection_agent import ScamDetectionAgent, ScamDetector, SemanticScamAnalyzer
from src.agents.question_strategy_agent import QuestionStrategyAgent
from src.agents.reasoning import ChainOfThought, ReActReasoner, ReasoningTrace, cot_reason, react_reason
from src.agents.prompt_generator import ConversationHistory, Message
from src.agents.emotion_predictor import EmotionPredictor, EMOTIONS

__all__ = [
    "LLMRouter", "router", "AgentRole", "Provider",
    "AgentContext",
    "OrchestratorAgent",
    "PersonaAgent", "PersonaAgentPool", "create_agent_pool_from_file",
    "FeaturePredictionAgent",
    "EmotionAgent", "EmotionDetector",
    "ScamDetectionAgent", "ScamDetector", "SemanticScamAnalyzer",
    "QuestionStrategyAgent",
    "ChainOfThought", "ReActReasoner", "ReasoningTrace", "cot_reason", "react_reason",
    "ConversationHistory", "Message",
    "EmotionPredictor", "EMOTIONS",
]
