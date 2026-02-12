"""Agent modules"""

from src.agents.persona_agent import (
    PersonaAgent,
    PersonaAgentPool,
    create_agent_pool_from_file
)
from src.agents.prompt_generator import (
    ConversationHistory,
    Message,
    format_conversation_context,
    create_greeting_prompt,
    enhance_prompt_with_context,
    extract_system_prompt_summary
)
from src.agents.emotion_agent import (
    EmotionAgent,
    EmotionDetector
)
from src.agents.emotion_predictor import (
    EmotionPredictor,
    EMOTIONS,
    EMOTION_TRANSITIONS
)
from src.agents.scam_detection_agent import (
    ScamDetectionAgent,
    ScamDetector,
    SemanticScamAnalyzer
)
from src.agents.scam_patterns import (
    ScamPattern,
    PATTERN_RULES,
    RISK_THRESHOLDS,
    WARNING_MESSAGES,
    get_pattern_description
)

__all__ = [
    # Persona Agent
    "PersonaAgent",
    "PersonaAgentPool",
    "create_agent_pool_from_file",
    
    # Prompt Utilities
    "ConversationHistory",
    "Message",
    "format_conversation_context",
    "create_greeting_prompt",
    "enhance_prompt_with_context",
    "extract_system_prompt_summary",
    
    # Emotion Agent
    "EmotionAgent",
    "EmotionDetector",
    "EmotionPredictor",
    "EMOTIONS",
    "EMOTION_TRANSITIONS",
    
    # Scam Detection Agent
    "ScamDetectionAgent",
    "ScamDetector",
    "SemanticScamAnalyzer",
    "ScamPattern",
    "PATTERN_RULES",
    "RISK_THRESHOLDS",
    "WARNING_MESSAGES",
    "get_pattern_description",
]
