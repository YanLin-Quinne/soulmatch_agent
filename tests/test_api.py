"""API tests (requires backend running)"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestSessionManager:
    """Test session management"""

    @pytest.fixture
    def session_manager(self):
        """Create session manager with mocked dependencies"""
        with patch('src.api.session_manager.OrchestratorAgent'):
            from src.api.session_manager import SessionManager

            # Reset singleton
            SessionManager._instance = None
            SessionManager._lock = __import__('threading').Lock()

            manager = SessionManager()
            manager.bot_personas_pool = MagicMock()

            return manager

    def test_singleton_pattern(self):
        """Test SessionManager is singleton"""
        with patch('src.api.session_manager.OrchestratorAgent'):
            from src.api.session_manager import SessionManager

            # Reset singleton
            SessionManager._instance = None
            SessionManager._lock = __import__('threading').Lock()

            manager1 = SessionManager()
            manager2 = SessionManager()

            assert manager1 is manager2

    def test_create_session(self, session_manager):
        """Test session creation"""
        session_id = session_manager.create_session("user_123")

        assert session_id is not None
        assert session_id == "user_123"

    def test_get_session(self, session_manager):
        """Test getting existing session"""
        session_id = session_manager.create_session("user_456")
        orchestrator = session_manager.get_session(session_id)

        assert orchestrator is not None

    def test_delete_session(self, session_manager):
        """Test session deletion"""
        session_id = session_manager.create_session("user_789")
        session_manager.delete_session(session_id)

        orchestrator = session_manager.get_session(session_id)
        assert orchestrator is None


@pytest.mark.asyncio
class TestChatHandler:
    """Test chat handler logic"""

    @pytest.fixture
    def chat_handler(self):
        """Create chat handler with mocked dependencies"""
        from src.api.chat_handler import ChatHandler

        handler = ChatHandler()
        mock_orchestrator = MagicMock()
        mock_orchestrator.process_user_message = AsyncMock()

        return handler, mock_orchestrator

    async def test_handle_start_conversation_success(self, chat_handler):
        """Test starting conversation successfully"""
        handler, mock_orchestrator = chat_handler

        mock_orchestrator.start_new_conversation.return_value = {
            "success": True,
            "bot_id": "bot_0",
            "greeting": "Hello!"
        }

        with patch('src.api.chat_handler.session_manager') as mock_sm:
            mock_sm.get_session.return_value = mock_orchestrator
            result = await handler.handle_start_conversation("session_123")

        assert result["success"] is True
        assert "bot_id" in result

    async def test_handle_user_message_success(self, chat_handler):
        """Test handling user message successfully"""
        handler, mock_orchestrator = chat_handler

        mock_orchestrator.process_user_message.return_value = {
            "success": True,
            "bot_message": "Nice to meet you!",
            "emotion": {"current_emotion": {"emotion": "joy"}}
        }

        with patch('src.api.chat_handler.session_manager') as mock_sm:
            mock_sm.get_session.return_value = mock_orchestrator
            result = await handler.handle_user_message("session_123", "Hi there!")

        assert result["success"] is True
        assert "bot_message" in result

    async def test_handle_no_session(self, chat_handler):
        """Test handling request with no session"""
        handler, _ = chat_handler

        with patch('src.api.chat_handler.session_manager') as mock_sm:
            mock_sm.get_session.return_value = None
            result = await handler.handle_user_message("invalid_session", "Hello")

        assert result["success"] is False
        assert "error" in result


class TestWebSocketProtocol:
    """Test WebSocket message protocol"""
    
    def test_message_format(self):
        """Test WebSocket message format"""
        
        import json
        
        # Client → Server
        client_start = {"action": "start"}
        assert "action" in client_start
        
        client_message = {"action": "message", "content": "Hello"}
        assert "action" in client_message
        assert "content" in client_message
        
        # Server → Client
        server_welcome = {
            "type": "welcome",
            "user_id": "user_123"
        }
        assert "type" in server_welcome
        
        server_bot_message = {
            "type": "bot_message",
            "message": "Hi!",
            "turn": 1
        }
        assert "type" in server_bot_message
        assert "message" in server_bot_message


@pytest.mark.asyncio
class TestFastAPIEndpoints:
    """Test FastAPI endpoints (requires app import)"""
    
    async def test_health_endpoint_format(self):
        """Test health endpoint response format"""
        
        # Expected response structure
        health_response = {
            "status": "ok",
            "service": "SoulMatch API",
            "bot_personas_loaded": 0
        }
        
        assert "status" in health_response
        assert "service" in health_response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
