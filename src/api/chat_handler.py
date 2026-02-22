"""Chat Handler - Handles chat logic and orchestrator interactions"""

from typing import Dict, Any, Optional
from loguru import logger

from src.api.session_manager import session_manager
from src.agents.orchestrator import OrchestratorAgent


class ChatHandler:
    """
    Handles chat operations by coordinating with OrchestratorAgent
    
    Provides error handling and response formatting for chat interactions
    """
    
    def __init__(self):
        """Initialize chat handler"""
        logger.info("ChatHandler initialized")
    
    async def handle_start_conversation(self, user_id: str) -> Dict[str, Any]:
        """
        Start a new conversation for a user
        
        Args:
            user_id: User identifier
        
        Returns:
            Dict with conversation start result
        """
        try:
            # Get or create session
            orchestrator = session_manager.get_session(user_id)
            
            if not orchestrator:
                # Create new session if doesn't exist
                session_manager.create_session(user_id)
                orchestrator = session_manager.get_session(user_id)
            
            if not orchestrator:
                return {
                    "success": False,
                    "error": "Failed to create session"
                }
            
            # Start new conversation
            result = orchestrator.start_new_conversation()
            
            logger.info(f"Started conversation for user {user_id}: {result.get('success', False)}")
            return result
            
        except Exception as e:
            logger.error(f"Error starting conversation for user {user_id}: {e}")
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
            Dict with bot response and agent outputs
        """
        try:
            # Get session
            orchestrator = session_manager.get_session(user_id)
            
            if not orchestrator:
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
            
            # Process message through orchestrator
            result = await orchestrator.process_user_message(message.strip())
            
            logger.info(f"Processed message for user {user_id}: turn {result.get('turn', 0)}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing message for user {user_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to process message: {str(e)}"
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
            orchestrator = session_manager.get_session(user_id)
            
            if not orchestrator:
                return {
                    "success": False,
                    "error": "No active session found"
                }
            
            summary = orchestrator.get_conversation_summary()
            
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
    
    async def reset_conversation(self, user_id: str) -> Dict[str, Any]:
        """
        Reset conversation for a user
        
        Args:
            user_id: User identifier
        
        Returns:
            Result dict
        """
        try:
            orchestrator = session_manager.get_session(user_id)
            
            if not orchestrator:
                return {
                    "success": False,
                    "error": "No active session found"
                }
            
            orchestrator.reset_conversation()
            
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
    
    async def get_user_features(self, user_id: str) -> Dict[str, Any]:
        """
        Get predicted user features
        
        Args:
            user_id: User identifier
        
        Returns:
            User features dict
        """
        try:
            orchestrator = session_manager.get_session(user_id)
            
            if not orchestrator:
                return {
                    "success": False,
                    "error": "No active session found"
                }
            
            features = orchestrator.feature_agent.get_feature_summary()
            
            return {
                "success": True,
                "features": features
            }
            
        except Exception as e:
            logger.error(f"Error getting user features for user {user_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to get user features: {str(e)}"
            }


# Global chat handler instance
chat_handler = ChatHandler()
