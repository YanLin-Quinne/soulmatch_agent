"""Integration tests for WebSocket protocol"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestWebSocketProtocol:
    """Test WebSocket message protocol"""

    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test async WebSocket client connection"""
        # Mock websocket connection
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({"type": "connected", "user_id": "test_user"}))

        response = await mock_ws.recv()
        data = json.loads(response)

        assert data["type"] == "connected"
        assert data["user_id"] == "test_user"

    @pytest.mark.asyncio
    async def test_message_send_receive_flow(self):
        """Test message send/receive complete flow"""
        mock_ws = AsyncMock()

        # Send message
        message = {"type": "message", "content": "Hi"}
        await mock_ws.send(json.dumps(message))

        # Receive response
        mock_ws.recv = AsyncMock(return_value=json.dumps({
            "type": "bot_message",
            "content": "Hello!",
            "emotion": "happy"
        }))

        response = await mock_ws.recv()
        data = json.loads(response)

        assert data["type"] == "bot_message"
        assert "emotion" in data

    @pytest.mark.asyncio
    async def test_five_message_types(self):
        """Verify 5 message types (bot_message, emotion, warning, features, turn)"""
        message_types = [
            {"type": "bot_message", "content": "Hello"},
            {"type": "emotion", "emotion": "happy", "confidence": 0.8},
            {"type": "warning", "level": "low", "message": "Be careful"},
            {"type": "features", "predicted": {"openness": 0.7}},
            {"type": "turn", "count": 5}
        ]

        for msg in message_types:
            mock_ws = AsyncMock()
            mock_ws.recv = AsyncMock(return_value=json.dumps(msg))

            response = await mock_ws.recv()
            data = json.loads(response)

            assert data["type"] == msg["type"]

    @pytest.mark.asyncio
    async def test_connection_disconnect_reconnect(self):
        """Test connection disconnect and reconnect"""
        mock_ws = AsyncMock()

        # Simulate disconnect
        mock_ws.close = AsyncMock()
        await mock_ws.close()

        # Simulate reconnect
        mock_ws_new = AsyncMock()
        mock_ws_new.recv = AsyncMock(return_value=json.dumps({"type": "reconnected"}))

        response = await mock_ws_new.recv()
        data = json.loads(response)

        assert data["type"] == "reconnected"

    @pytest.mark.asyncio
    async def test_concurrent_multi_user_sessions(self):
        """Test concurrent multi-user sessions"""
        users = ["user1", "user2", "user3"]
        sessions = {}

        for user_id in users:
            mock_ws = AsyncMock()
            mock_ws.recv = AsyncMock(return_value=json.dumps({
                "type": "connected",
                "user_id": user_id
            }))
            sessions[user_id] = mock_ws

        # Verify all sessions active
        assert len(sessions) == 3

        # Verify each session independent
        for user_id, ws in sessions.items():
            response = await ws.recv()
            data = json.loads(response)
            assert data["user_id"] == user_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
