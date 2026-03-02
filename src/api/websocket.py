"""WebSocket endpoint for real-time chat"""

import asyncio
import json
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from src.api.chat_handler import chat_handler
from src.api.session_manager import session_manager


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        # user_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, user_id: str, websocket: WebSocket):
        """Accept and register a WebSocket connection"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"WebSocket connected: user_id={user_id}")
    
    def disconnect(self, user_id: str):
        """Unregister a WebSocket connection"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"WebSocket disconnected: user_id={user_id}")
    
    async def send_message(self, user_id: str, message: dict):
        """Send a message to a specific user"""
        if user_id in self.active_connections:
            websocket = self.active_connections[user_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to user {user_id}: {e}")
                self.disconnect(user_id)
    
    def is_connected(self, user_id: str) -> bool:
        """Check if user is connected"""
        return user_id in self.active_connections


# Global connection manager
manager = ConnectionManager()


async def _send_heartbeat(user_id: str):
    """Send heartbeat every 5s to keep WebSocket alive during long processing."""
    try:
        while True:
            await asyncio.sleep(5)
            await manager.send_message(user_id, {"type": "heartbeat"})
    except asyncio.CancelledError:
        pass


async def _run_background_prediction(user_id: str):
    """Run relationship prediction in background and send results via WebSocket."""
    try:
        orchestrator = session_manager.get_session(user_id)
        if not orchestrator:
            return
        result = await orchestrator.run_relationship_prediction()
        if result.get("relationship_prediction"):
            await manager.send_message(user_id, {
                "type": "relationship_prediction",
                "data": result["relationship_prediction"]
            })
        if result.get("milestone_report"):
            await manager.send_message(user_id, {
                "type": "milestone_report",
                "data": result["milestone_report"]
            })
        if result.get("love_prediction"):
            await manager.send_message(user_id, {
                "type": "love_prediction",
                "data": result["love_prediction"]
            })
    except Exception as e:
        logger.error(f"Background relationship prediction failed for {user_id}: {e}")


async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time chat
    
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
            {"type": "emotion", "data": {...}} - Emotion analysis
            {"type": "warning", "data": {...}} - Scam warning
            {"type": "feature_update", "data": {...}} - Feature prediction update
            {"type": "context", "data": {...}} - Conversation context
            {"type": "summary", "data": {...}} - Conversation summary
            {"type": "error", "message": "..."} - Error message
    """
    
    await manager.connect(user_id, websocket)
    
    # Send welcome message
    await manager.send_message(user_id, {
        "type": "welcome",
        "message": f"Connected to SoulMatch! Ready to chat.",
        "user_id": user_id
    })
    
    # Ensure session exists
    try:
        if not session_manager.get_session(user_id):
            session_manager.create_session(user_id)
            logger.info(f"Created session for new WebSocket user: {user_id}")
    except Exception as e:
        logger.error(f"Error creating session for user {user_id}: {e}")
        await manager.send_message(user_id, {
            "type": "error",
            "message": "Failed to initialize session"
        })
        await websocket.close()
        return
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await manager.send_message(user_id, {
                    "type": "error",
                    "message": "Invalid JSON format"
                })
                continue
            
            action = message.get("action")
            
            if not action:
                await manager.send_message(user_id, {
                    "type": "error",
                    "message": "Missing 'action' field"
                })
                continue
            
            # Handle different actions
            if action == "typing":
                # Broadcast typing indicator (for future multi-user)
                is_typing = message.get("is_typing", False)
                # Store in orchestrator context if available
                orchestrator = session_manager.get_session(user_id)
                if orchestrator:
                    orchestrator.ctx.peer_typing = is_typing
                # Echo back as typing_indicator for UI consistency
                await manager.send_message(user_id, {
                    "type": "typing_indicator",
                    "data": {"user_id": user_id, "is_typing": is_typing}
                })

            elif action == "start":
                # Start a new conversation
                bot_id = message.get("bot_id")  # Optional: if provided, use specific bot
                result = await chat_handler.handle_start_conversation(user_id, bot_id=bot_id)
                
                if result.get("success"):
                    await manager.send_message(user_id, {
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
                    await manager.send_message(user_id, {
                        "type": "error",
                        "message": result.get("error", "Failed to start conversation")
                    })
            
            elif action == "ping":
                await manager.send_message(user_id, {"type": "pong"})

            elif action == "message":
                # Process user message
                content = message.get("content", "").strip()
                quote_id = message.get("quote_id")
                quote_text = message.get("quote_text")

                if not content:
                    await manager.send_message(user_id, {
                        "type": "error",
                        "message": "Message content cannot be empty"
                    })
                    continue

                # Store quote context in orchestrator
                orchestrator = session_manager.get_session(user_id)
                if orchestrator and quote_text:
                    orchestrator.ctx.quoted_message = {"id": quote_id, "content": quote_text}

                # Start heartbeat to keep connection alive during processing
                heartbeat_task = asyncio.create_task(_send_heartbeat(user_id))
                try:
                    result = await chat_handler.handle_user_message(user_id, content)
                finally:
                    heartbeat_task.cancel()

                if result.get("success"):
                    # Conversational pacing: delay before sending bot response
                    delay = result.get("reply_delay", 0)
                    if delay > 0:
                        await asyncio.sleep(delay)

                    # Send bot response
                    if "bot_message" in result:
                        bot_msg_payload = {
                            "type": "bot_message",
                            "message": result["bot_message"],
                            "turn": result.get("turn")
                        }
                        # Attach quote context if user quoted a message
                        if quote_text:
                            bot_msg_payload["quoted"] = {"id": quote_id, "content": quote_text}
                        await manager.send_message(user_id, bot_msg_payload)
                        # Clear quote after use
                        if orchestrator:
                            orchestrator.ctx.quoted_message = None

                    # Send emotion analysis if available
                    if "emotion" in result:
                        await manager.send_message(user_id, {
                            "type": "emotion",
                            "data": result["emotion"]
                        })

                    # Send scam warning if detected
                    if "scam_detection" in result:
                        scam_data = result["scam_detection"]
                        if scam_data.get("warning_level") in ["medium", "high", "critical"]:
                            await manager.send_message(user_id, {
                                "type": "warning",
                                "data": {
                                    "level": scam_data.get("warning_level"),
                                    "message": scam_data.get("message", {}).get("en", ""),
                                    "risk_score": scam_data.get("risk_score")
                                }
                            })

                    # Send feature update notification
                    if "feature_update" in result:
                        await manager.send_message(user_id, {
                            "type": "feature_update",
                            "data": result["feature_update"]
                        })

                    # Send conversation sentiment
                    if "conversation_sentiment" in result:
                        await manager.send_message(user_id, {
                            "type": "conversation_sentiment",
                            "data": result["conversation_sentiment"]
                        })

                    # Send threshold reached notification
                    if "threshold_reached" in result:
                        await manager.send_message(user_id, {
                            "type": "threshold_reached",
                            "data": result["threshold_reached"]
                        })

                    # Send conversation hints
                    if "conversation_hints" in result:
                        await manager.send_message(user_id, {
                            "type": "conversation_hints",
                            "data": result["conversation_hints"]
                        })

                    # Send digital twin auto-created notification
                    if "digital_twin_created" in result:
                        await manager.send_message(user_id, {
                            "type": "digital_twin_created",
                            "data": result["digital_twin_created"]
                        })

                    # Send context
                    if "context" in result:
                        await manager.send_message(user_id, {
                            "type": "context",
                            "data": result["context"]
                        })

                    # Send memory stats (three-layer memory visibility)
                    if "memory_stats" in result:
                        await manager.send_message(user_id, {
                            "type": "memory_stats",
                            "data": result["memory_stats"]
                        })

                    # Send warning message if present
                    if "warning" in result:
                        await manager.send_message(user_id, {
                            "type": "system_warning",
                            "message": result["warning"]
                        })

                    # Fire background relationship prediction if needed
                    orchestrator = session_manager.get_session(user_id)
                    if orchestrator and getattr(orchestrator, '_pending_relationship_turn', False):
                        asyncio.create_task(_run_background_prediction(user_id))
                else:
                    await manager.send_message(user_id, {
                        "type": "error",
                        "message": result.get("error", "Failed to process message")
                    })
            
            elif action == "summary":
                # Get conversation summary
                result = await chat_handler.get_conversation_summary(user_id)
                
                if result.get("success"):
                    await manager.send_message(user_id, {
                        "type": "summary",
                        "data": result.get("summary")
                    })
                else:
                    await manager.send_message(user_id, {
                        "type": "error",
                        "message": result.get("error", "Failed to get summary")
                    })
            
            elif action == "reset":
                # Reset conversation
                result = await chat_handler.reset_conversation(user_id)
                
                await manager.send_message(user_id, {
                    "type": "reset_complete" if result.get("success") else "error",
                    "message": result.get("message" if result.get("success") else "error")
                })
            
            elif action == "features":
                # Get user features
                result = await chat_handler.get_user_features(user_id)
                
                if result.get("success"):
                    await manager.send_message(user_id, {
                        "type": "user_features",
                        "data": result.get("features")
                    })
                else:
                    await manager.send_message(user_id, {
                        "type": "error",
                        "message": result.get("error", "Failed to get features")
                    })
            
            elif action == "create_twin":
                # Create digital twin from current predicted features
                orchestrator = session_manager.get_session(user_id)
                if orchestrator:
                    try:
                        twin_profile = orchestrator.digital_twin_agent.create_twin(orchestrator.ctx)
                        await manager.send_message(user_id, {
                            "type": "digital_twin_created",
                            "data": twin_profile
                        })
                    except Exception as e:
                        logger.error(f"Digital twin creation failed for {user_id}: {e}")
                        await manager.send_message(user_id, {
                            "type": "error",
                            "message": f"Failed to create digital twin: {str(e)}"
                        })
                else:
                    await manager.send_message(user_id, {
                        "type": "error",
                        "message": "No active session for twin creation"
                    })

            elif action == "twin_message":
                # Chat with the digital twin
                content = message.get("content", "").strip()
                orchestrator = session_manager.get_session(user_id)
                if orchestrator and orchestrator.ctx.digital_twin:
                    try:
                        # Echo user message back
                        await manager.send_message(user_id, {
                            "type": "twin_message",
                            "data": {"sender": "user", "content": content}
                        })
                        # Generate twin response
                        twin_reply = await orchestrator.digital_twin_agent.chat_with_twin(
                            orchestrator.ctx.digital_twin, content
                        )
                        await manager.send_message(user_id, {
                            "type": "twin_message",
                            "data": {"sender": "twin", "content": twin_reply}
                        })
                    except Exception as e:
                        logger.error(f"Twin chat failed for {user_id}: {e}")
                        await manager.send_message(user_id, {
                            "type": "error",
                            "message": f"Twin chat error: {str(e)}"
                        })
                else:
                    await manager.send_message(user_id, {
                        "type": "error",
                        "message": "No digital twin exists yet. Create one first."
                    })

            elif action == "compare_perception":
                # Compare user's self-perception with predicted features
                perception = message.get("perception", {})
                orchestrator = session_manager.get_session(user_id)
                if orchestrator:
                    try:
                        comparison = orchestrator.digital_twin_agent.compare_perception(
                            orchestrator.ctx, perception
                        )
                        await manager.send_message(user_id, {
                            "type": "perception_comparison",
                            "data": comparison
                        })
                    except Exception as e:
                        logger.error(f"Perception comparison failed for {user_id}: {e}")
                        await manager.send_message(user_id, {
                            "type": "error",
                            "message": f"Perception comparison error: {str(e)}"
                        })
                else:
                    await manager.send_message(user_id, {
                        "type": "error",
                        "message": "No active session for perception comparison"
                    })

            else:
                await manager.send_message(user_id, {
                    "type": "error",
                    "message": f"Unknown action: {action}"
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected normally: user_id={user_id}")
        manager.disconnect(user_id)
    
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        manager.disconnect(user_id)
        try:
            await websocket.close()
        except:
            pass
