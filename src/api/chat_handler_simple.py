"""Simple Chat Handler - Uses SimplePersonaAgent for reliable conversations

This is a simplified version that uses SimplePersonaAgent instead of the
full orchestrator pipeline. It provides:
- Guaranteed message format
- Fallback to rules if API fails
- No complex multi-agent coordination
- Faster response times
"""

from typing import Dict, Any
from loguru import logger

from src.agents.simple_persona import SimplePersonaAgent
from src.agents.persona_agent import PersonaAgentPool


class SimpleChatHandler:
    """
    Simplified chat handler using SimplePersonaAgent.

    Differences from full ChatHandler:
    - No orchestrator, just direct persona agent
    - No feature prediction, emotion analysis, etc.
    - Simpler state management
    - More reliable message format
    """

    def __init__(self):
        """Initialize simple chat handler"""
        self.sessions: Dict[str, SimplePersonaAgent] = {}
        self.bot_pool: PersonaAgentPool = None
        logger.info("SimpleChatHandler initialized")

    def set_bot_pool(self, pool: PersonaAgentPool):
        """Set the bot personas pool"""
        self.bot_pool = pool
        logger.info(f"Bot pool set with {len(pool)} personas")

    async def handle_start_conversation(self, user_id: str) -> Dict[str, Any]:
        """
        Start a new conversation for a user

        Args:
            user_id: User identifier

        Returns:
            Dict with conversation start result
        """
        try:
            if not self.bot_pool:
                return {
                    "success": False,
                    "error": "Bot pool not initialized"
                }

            # Get available bots
            bot_summaries = self.bot_pool.get_agent_summaries()
            if not bot_summaries:
                return {
                    "success": False,
                    "error": "No bots available"
                }

            # For now, randomly select a bot
            # TODO: Add matching logic later
            import random
            selected_bot_id = random.choice(list(bot_summaries.keys()))

            # Get the persona from pool
            bot_agent = self.bot_pool.get_agent(selected_bot_id)
            if not bot_agent:
                return {
                    "success": False,
                    "error": f"Bot {selected_bot_id} not found"
                }

            # Create SimplePersonaAgent from the persona
            simple_agent = SimplePersonaAgent(
                persona=bot_agent.persona,
                temperature=0.8
            )

            # Store in sessions
            self.sessions[user_id] = simple_agent

            # Generate greeting
            greeting = simple_agent.get_greeting()

            bot_summary = bot_summaries[selected_bot_id]

            logger.info(f"Started simple conversation for user {user_id} with bot {selected_bot_id}")

            return {
                "success": True,
                "bot_id": selected_bot_id,
                "bot_profile": bot_summary,
                "compatibility_score": 0.75,  # Placeholder
                "match_explanation": "Let's see if you two click!",
                "greeting": greeting
            }

        except Exception as e:
            logger.error(f"Error starting conversation for user {user_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": f"Failed to start conversation: {str(e)}"
            }

    async def handle_user_message(
        self,
        user_id: str,
        message: str
    ) -> Dict[str, Any]:
        """
        Process a user message

        Args:
            user_id: User identifier
            message: User's message content

        Returns:
            Dict with bot response
        """
        try:
            # Get session
            agent = self.sessions.get(user_id)

            if not agent:
                return {
                    "success": False,
                    "error": "No active session. Please start a conversation first."
                }

            # Validate message
            if not message or not message.strip():
                return {
                    "success": False,
                    "error": "Message cannot be empty"
                }

            # Generate response
            response = agent.generate_response(message.strip())

            # Calculate turn count
            turn = len(agent.messages) // 2

            logger.info(f"Processed message for user {user_id}: turn {turn}")

            return {
                "success": True,
                "bot_message": response,
                "turn": turn
            }

        except Exception as e:
            logger.error(f"Error processing message for user {user_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": f"Failed to process message: {str(e)}"
            }

    async def reset_conversation(self, user_id: str) -> Dict[str, Any]:
        """
        Reset conversation for a user

        Args:
            user_id: User identifier

        Returns:
            Result dict
        """
        try:
            agent = self.sessions.get(user_id)

            if not agent:
                return {
                    "success": False,
                    "error": "No active session found"
                }

            agent.reset()

            logger.info(f"Reset conversation for user {user_id}")
            return {
                "success": True,
                "message": "Conversation reset successfully"
            }

        except Exception as e:
            logger.error(f"Error resetting conversation for user {user_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to reset conversation: {str(e)}"
            }

    async def get_conversation_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get summary of current conversation

        Args:
            user_id: User identifier

        Returns:
            Conversation summary dict
        """
        try:
            agent = self.sessions.get(user_id)

            if not agent:
                return {
                    "success": False,
                    "error": "No active session found"
                }

            summary = {
                "total_messages": len(agent.messages),
                "turns": len(agent.messages) // 2,
                "persona_id": agent.persona.profile_id
            }

            return {
                "success": True,
                "summary": summary
            }

        except Exception as e:
            logger.error(f"Error getting conversation summary for user {user_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to get summary: {str(e)}"
            }


# Global simple chat handler instance
simple_chat_handler = SimpleChatHandler()
