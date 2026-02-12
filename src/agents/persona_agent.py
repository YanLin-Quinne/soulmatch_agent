"""Persona Agent for role-playing dating app conversations"""

import json
from pathlib import Path
from typing import Optional, Union
from loguru import logger

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic package not available")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai package not available")

from src.data.schema import PersonaProfile
from src.config import settings
from src.agents.prompt_generator import ConversationHistory


class PersonaAgent:
    """Agent that role-plays as a specific persona in conversations"""
    
    def __init__(
        self, 
        persona: PersonaProfile,
        use_claude: bool = True,
        model_name: Optional[str] = None,
        temperature: float = 0.8,
        max_tokens: int = 300
    ):
        """
        Initialize persona agent
        
        Args:
            persona: PersonaProfile with system_prompt and features
            use_claude: Use Claude API (default) vs OpenAI GPT
            model_name: Override default model (claude-3-5-sonnet-20241022 or gpt-4o-mini)
            temperature: Response creativity (0.0-1.0, higher = more creative)
            max_tokens: Maximum response length
        """
        self.persona = persona
        self.use_claude = use_claude and ANTHROPIC_AVAILABLE
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize API client
        if self.use_claude:
            if not settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not configured")
            self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            self.model = model_name or "claude-3-5-sonnet-20241022"
            logger.info(f"PersonaAgent initialized with Claude model: {self.model}")
        else:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed")
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not configured")
            self.client = openai.OpenAI(api_key=settings.openai_api_key)
            self.model = model_name or "gpt-4o-mini"
            logger.info(f"PersonaAgent initialized with GPT model: {self.model}")
        
        # Conversation tracking
        self.conversation_history = ConversationHistory(max_messages=20)  # 10 rounds
        
        logger.info(f"PersonaAgent created for profile: {persona.profile_id}")
    
    def generate_response(
        self, 
        message: str,
        conversation_history: Optional[ConversationHistory] = None
    ) -> str:
        """
        Generate a response to a message, staying in character
        
        Args:
            message: User's message
            conversation_history: Optional conversation history (uses internal if None)
        
        Returns:
            Agent's response as the persona
        """
        # Use provided history or internal tracking
        history = conversation_history or self.conversation_history
        
        # Add user message to history
        history.add_user_message(message)
        
        try:
            if self.use_claude:
                response_text = self._generate_claude_response(message, history)
            else:
                response_text = self._generate_gpt_response(message, history)
            
            # Add response to history
            history.add_assistant_message(response_text)
            
            logger.debug(f"[{self.persona.profile_id}] Generated response: {response_text[:100]}...")
            return response_text
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            # Fallback response
            fallback = self._get_fallback_response()
            history.add_assistant_message(fallback)
            return fallback
    
    def _generate_claude_response(
        self, 
        message: str, 
        history: ConversationHistory
    ) -> str:
        """Generate response using Claude API"""
        
        # Build messages (history + current)
        messages = history.to_api_format()
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.persona.system_prompt,  # System prompt defines the persona
            messages=messages
        )
        
        return response.content[0].text
    
    def _generate_gpt_response(
        self, 
        message: str, 
        history: ConversationHistory
    ) -> str:
        """Generate response using OpenAI GPT API"""
        
        # Build messages: system + history
        messages = [{"role": "system", "content": self.persona.system_prompt}]
        messages.extend(history.to_api_format())
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    def _get_fallback_response(self) -> str:
        """Get a fallback response when API fails"""
        
        # Use personality to craft a generic but in-character response
        style = self.persona.features.communication_style
        
        fallbacks = {
            "humorous": "Haha, my brain just blue-screened for a second there! What were we talking about? 😅",
            "direct": "Sorry, I lost my train of thought. Could you repeat that?",
            "casual": "Oops, spaced out for a sec! What did you say?",
            "formal": "I apologize, I seem to have lost track of our conversation. Could you please rephrase that?",
            "serious": "I'm sorry, I need a moment to collect my thoughts. Could you say that again?",
            "indirect": "Hmm, I'm not sure I caught all of that. Mind saying it again?"
        }
        
        return fallbacks.get(style, "Sorry, could you say that again?")
    
    def generate_greeting(self) -> str:
        """Generate an opening message to start a conversation"""
        
        prompt = "Generate a warm, natural first message to start a conversation on a dating app. Be authentic, show your personality, and keep it casual (2-3 sentences max). Don't ask generic questions."
        
        # Create temporary history for greeting
        temp_history = ConversationHistory(max_messages=2)
        temp_history.add_user_message(prompt)
        
        try:
            if self.use_claude:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=150,
                    temperature=0.9,  # More creative for greetings
                    system=self.persona.system_prompt,
                    messages=temp_history.to_api_format()
                )
                greeting = response.content[0].text
            else:
                messages = [{"role": "system", "content": self.persona.system_prompt}]
                messages.extend(temp_history.to_api_format())
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.9,
                    max_tokens=150
                )
                greeting = response.choices[0].message.content
            
            logger.info(f"[{self.persona.profile_id}] Generated greeting")
            return greeting
            
        except Exception as e:
            logger.error(f"Failed to generate greeting: {e}")
            return "Hey! How's your day going? 😊"
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history.clear()
        logger.debug(f"[{self.persona.profile_id}] Conversation reset")
    
    def get_persona_summary(self) -> dict:
        """Get a summary of the persona for debugging/display"""
        return {
            "profile_id": self.persona.profile_id,
            "age": self.persona.original_profile.age,
            "sex": self.persona.original_profile.sex,
            "location": self.persona.original_profile.location,
            "communication_style": self.persona.features.communication_style,
            "core_values": self.persona.features.core_values,
            "interests": list(self.persona.features.interest_categories.keys())[:3],
            "relationship_goals": self.persona.features.relationship_goals,
            "personality_summary": self.persona.features.personality_summary
        }


class PersonaAgentPool:
    """Manages multiple persona agents (bot pool)"""
    
    def __init__(
        self, 
        personas: Optional[list[PersonaProfile]] = None,
        use_claude: bool = True,
        temperature: float = 0.8
    ):
        """
        Initialize agent pool
        
        Args:
            personas: List of PersonaProfile objects (or load from file)
            use_claude: Use Claude API vs OpenAI
            temperature: Response creativity
        """
        self.use_claude = use_claude
        self.temperature = temperature
        self.agents: dict[str, PersonaAgent] = {}
        
        if personas:
            self.load_personas(personas)
            logger.info(f"PersonaAgentPool initialized with {len(self.agents)} agents")
    
    def load_personas(self, personas: list[PersonaProfile]):
        """Load personas into the pool"""
        for persona in personas:
            agent = PersonaAgent(
                persona=persona,
                use_claude=self.use_claude,
                temperature=self.temperature
            )
            self.agents[persona.profile_id] = agent
        
        logger.info(f"Loaded {len(personas)} personas into pool")
    
    def load_from_file(self, personas_path: Union[str, Path]):
        """Load personas from JSON file"""
        personas_path = Path(personas_path)
        
        if not personas_path.exists():
            raise FileNotFoundError(f"Personas file not found: {personas_path}")
        
        with open(personas_path, 'r', encoding='utf-8') as f:
            personas_dict = json.load(f)
        
        personas = [PersonaProfile(**p) for p in personas_dict]
        self.load_personas(personas)
        
        logger.info(f"Loaded {len(personas)} personas from {personas_path}")
    
    def get_agent(self, profile_id: str) -> Optional[PersonaAgent]:
        """Get agent by profile ID"""
        agent = self.agents.get(profile_id)
        if not agent:
            logger.warning(f"No agent found for profile_id: {profile_id}")
        return agent
    
    def get_all_agents(self) -> list[PersonaAgent]:
        """Get all agents"""
        return list(self.agents.values())
    
    def get_agent_summaries(self) -> list[dict]:
        """Get summaries of all agents"""
        return [agent.get_persona_summary() for agent in self.agents.values()]
    
    def reset_all_conversations(self):
        """Reset all conversation histories"""
        for agent in self.agents.values():
            agent.reset_conversation()
        logger.info("All agent conversations reset")
    
    def __len__(self) -> int:
        """Get number of agents in pool"""
        return len(self.agents)
    
    def __contains__(self, profile_id: str) -> bool:
        """Check if profile_id exists in pool"""
        return profile_id in self.agents


# Convenience function for quick setup
def create_agent_pool_from_file(
    personas_path: Union[str, Path] = "data/processed/bot_personas.json",
    use_claude: bool = True,
    temperature: float = 0.8
) -> PersonaAgentPool:
    """
    Quick setup: Create agent pool from bot personas file
    
    Args:
        personas_path: Path to bot_personas.json
        use_claude: Use Claude API (default) vs OpenAI
        temperature: Response creativity
    
    Returns:
        Initialized PersonaAgentPool
    """
    pool = PersonaAgentPool(use_claude=use_claude, temperature=temperature)
    pool.load_from_file(personas_path)
    return pool
