"""Simple WebSocket endpoint using SimpleChatHandler

This is a simplified WebSocket endpoint that uses SimplePersonaAgent
instead of the full orchestrator pipeline.
"""

import json
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from src.api.chat_handler_simple import simple_chat_handler


class SimpleConnectionManager:
    """Manages WebSocket connections for simple chat"""

    def __init__(self):
        # user_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        """Accept and register a WebSocket connection"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"[Simple] WebSocket connected: user_id={user_id}")

    def disconnect(self, user_id: str):
        """Unregister a WebSocket connection"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"[Simple] WebSocket disconnected: user_id={user_id}")

    async def send_message(self, user_id: str, message: dict):
        """Send a message to a specific user"""
        if user_id in self.active_connections:
            websocket = self.active_connections[user_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"[Simple] Error sending message to user {user_id}: {e}")
                self.disconnect(user_id)

    def is_connected(self, user_id: str) -> bool:
        """Check if user is connected"""
        return user_id in self.active_connections


# Global connection manager
simple_manager = SimpleConnectionManager()


async def websocket_simple_endpoint(websocket: WebSocket, user_id: str):
    """
    Simple WebSocket endpoint for real-time chat

    Protocol:
        Client -> Server:
            {"action": "start"} - Start a new conversation
            {"action": "message", "content": "..."} - Send a message
            {"action": "summary"} - Get conversation summary
            {"action": "reset"} - Reset conversation

        Server -> Client:
            {"type": "welcome", "message": "..."} - Welcome message
            {"type": "conversation_started", "data": {...}} - Conversation started
            {"type": "bot_message", "message": "..."} - Bot's reply
            {"type": "error", "message": "..."} - Error message
    """

    await simple_manager.connect(user_id, websocket)

    # Send welcome message
    await simple_manager.send_message(user_id, {
        "type": "welcome",
        "message": f"Connected to SoulMatch (Simple Mode)! Ready to chat.",
        "user_id": user_id,
        "mode": "simple"
    })

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await simple_manager.send_message(user_id, {
                    "type": "error",
                    "message": "Invalid JSON format"
                })
                continue

            action = message.get("action")

            if not action:
                await simple_manager.send_message(user_id, {
                    "type": "error",
                    "message": "Missing 'action' field"
                })
                continue

            # Handle different actions
            if action == "start":
                # Start a new conversation
                result = await simple_chat_handler.handle_start_conversation(user_id)

                if result.get("success"):
                    await simple_manager.send_message(user_id, {
                        "type": "conversation_started",
                        "data": {
                            "bot_id": result.get("bot_id"),
                            "bot_profile": result.get("bot_profile"),
                            "compatibility_score": result.get("compatibility_score"),
                            "match_explanation": result.get("match_explanation"),
                            "greeting": result.get("greeting")
                        }
                    })
                else:
                    await simple_manager.send_message(user_id, {
                        "type": "error",
                        "message": result.get("error", "Failed to start conversation")
                    })

            elif action == "message":
                # Process user message
                content = message.get("content", "").strip()

                if not content:
                    await simple_manager.send_message(user_id, {
                        "type": "error",
                        "message": "Message content cannot be empty"
                    })
                    continue

                result = await simple_chat_handler.handle_user_message(user_id, content)

                if result.get("success"):
                    # Send bot response
                    await simple_manager.send_message(user_id, {
                        "type": "bot_message",
                        "message": result["bot_message"],
                        "turn": result.get("turn")
                    })
                else:
                    await simple_manager.send_message(user_id, {
                        "type": "error",
                        "message": result.get("error", "Failed to process message")
                    })

            elif action == "summary":
                # Get conversation summary
                result = await simple_chat_handler.get_conversation_summary(user_id)

                if result.get("success"):
                    await simple_manager.send_message(user_id, {
                        "type": "summary",
                        "data": result.get("summary")
                    })
                else:
                    await simple_manager.send_message(user_id, {
                        "type": "error",
                        "message": result.get("error", "Failed to get summary")
                    })

            elif action == "reset":
                # Reset conversation
                result = await simple_chat_handler.reset_conversation(user_id)

                await simple_manager.send_message(user_id, {
                    "type": "reset_complete" if result.get("success") else "error",
                    "message": result.get("message" if result.get("success") else "error")
                })

            else:
                await simple_manager.send_message(user_id, {
                    "type": "error",
                    "message": f"Unknown action: {action}"
                })

    except WebSocketDisconnect:
        logger.info(f"[Simple] WebSocket disconnected normally: user_id={user_id}")
        simple_manager.disconnect(user_id)

    except Exception as e:
        logger.error(f"[Simple] WebSocket error for user {user_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        simple_manager.disconnect(user_id)
        try:
            await websocket.close()
        except:
            pass
