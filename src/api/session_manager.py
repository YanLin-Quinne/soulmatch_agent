"""Session Manager - Manages user sessions and Orchestrator instances"""

import time
from typing import Dict, Optional
from threading import Lock
from loguru import logger

from src.agents.orchestrator import OrchestratorAgent
from src.agents.persona_agent import PersonaAgentPool


class SessionManager:
    """
    Singleton session manager that maps user_id to OrchestratorAgent instances
    
    Thread-safe implementation with session timeout support
    """
    
    _instance: Optional["SessionManager"] = None
    _lock = Lock()
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize session manager (only once)"""
        if self._initialized:
            return
            
        self.sessions: Dict[str, OrchestratorAgent] = {}
        self.session_metadata: Dict[str, dict] = {}  # user_id -> {session_id, created_at, last_active}
        self.bot_personas_pool: Optional[PersonaAgentPool] = None
        self._initialized = True
        
        logger.info("SessionManager initialized")
    
    def set_bot_personas_pool(self, pool: PersonaAgentPool):
        """
        Set the global bot personas pool
        
        Args:
            pool: PersonaAgentPool instance loaded from bot_personas.json
        """
        self.bot_personas_pool = pool
        logger.info(f"Bot personas pool set with {len(pool)} personas")
    
    def create_session(self, user_id: str, bot_id: Optional[str] = None) -> str:
        """
        Create a new session for a user.

        Args:
            user_id: User identifier
            bot_id: Optional bot profile ID to use (if None, random selection)

        Returns:
            session_id (same as user_id in this simple implementation)
        """
        if not self.bot_personas_pool:
            raise RuntimeError("Bot personas pool not initialized. Call set_bot_personas_pool() first.")

        if user_id in self.sessions:
            logger.info(f"Session already exists for user {user_id}, reusing")
            self.session_metadata[user_id]["last_active"] = time.time()
            return user_id

        orchestrator = OrchestratorAgent(
            user_id=user_id,
            bot_personas_pool=self.bot_personas_pool,
            bot_id=bot_id,
        )
        
        self.sessions[user_id] = orchestrator
        self.session_metadata[user_id] = {
            "session_id": user_id,
            "created_at": time.time(),
            "last_active": time.time()
        }
        
        logger.info(f"Created new session for user {user_id}")
        return user_id
    
    def get_session(self, session_id: str) -> Optional[OrchestratorAgent]:
        """
        Get orchestrator for a session
        
        Args:
            session_id: Session identifier (user_id)
        
        Returns:
            OrchestratorAgent instance or None if not found
        """
        orchestrator = self.sessions.get(session_id)
        
        if orchestrator:
            # Update last active time
            if session_id in self.session_metadata:
                self.session_metadata[session_id]["last_active"] = time.time()
        else:
            logger.warning(f"Session not found: {session_id}")
        
        return orchestrator
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if deleted, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            if session_id in self.session_metadata:
                del self.session_metadata[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        else:
            logger.warning(f"Session not found for deletion: {session_id}")
            return False
    
    def get_session_info(self, session_id: str) -> Optional[dict]:
        """
        Get session metadata
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session metadata dict or None
        """
        return self.session_metadata.get(session_id)
    
    def cleanup_inactive_sessions(self, timeout_seconds: int = 3600):
        """
        Clean up sessions that have been inactive for too long
        
        Args:
            timeout_seconds: Inactivity timeout in seconds (default: 1 hour)
        
        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        expired_sessions = []
        
        for session_id, metadata in self.session_metadata.items():
            if current_time - metadata["last_active"] > timeout_seconds:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} inactive sessions")
        
        return len(expired_sessions)
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len(self.sessions)
    
    def list_all_sessions(self) -> list[dict]:
        """
        List all active sessions
        
        Returns:
            List of session metadata dicts
        """
        return [
            {
                "session_id": sid,
                **metadata
            }
            for sid, metadata in self.session_metadata.items()
        ]


# Global session manager instance
session_manager = SessionManager()
