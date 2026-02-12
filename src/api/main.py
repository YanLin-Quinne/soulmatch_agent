"""FastAPI main application - SoulMatch backend API"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

from src.config import settings
from src.api.session_manager import session_manager
from src.api.chat_handler import chat_handler
from src.api.websocket import websocket_endpoint
from src.agents.persona_agent import create_agent_pool_from_file


# ============================================
# Pydantic Models
# ============================================

class SessionStartRequest(BaseModel):
    """Request to start a new session"""
    user_id: str = Field(..., description="Unique user identifier")
    use_claude: bool = Field(True, description="Use Claude API (default) vs OpenAI")


class SessionStartResponse(BaseModel):
    """Response for session start"""
    success: bool
    session_id: str | None = None
    message: str | None = None
    error: str | None = None


class SessionInfoResponse(BaseModel):
    """Session information"""
    success: bool
    session_id: str | None = None
    created_at: float | None = None
    last_active: float | None = None
    error: str | None = None


class UserSummaryResponse(BaseModel):
    """User feature summary"""
    success: bool
    user_id: str
    summary: dict | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str = "1.0.0"
    bot_personas_loaded: int = 0


# ============================================
# Lifespan Management
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown
    
    Startup:
        - Load bot personas from data/processed/bot_personas.json
        - Initialize session manager with personas pool
    
    Shutdown:
        - Clean up resources
    """
    # Startup
    logger.info("Starting up SoulMatch API...")
    
    # Load bot personas
    personas_path = settings.data_dir / "processed" / "bot_personas.json"
    
    if not personas_path.exists():
        logger.warning(f"Bot personas file not found: {personas_path}")
        logger.warning("API will start but conversations cannot be started until personas are loaded")
        # Don't fail startup - allow API to run for health checks
    else:
        try:
            # Create agent pool from file
            bot_pool = create_agent_pool_from_file(
                personas_path=personas_path,
                use_claude=True,  # Default to Claude
                temperature=0.8
            )
            
            # Set pool in session manager
            session_manager.set_bot_personas_pool(bot_pool)
            
            logger.info(f"✅ Loaded {len(bot_pool)} bot personas from {personas_path}")
            
        except Exception as e:
            logger.error(f"Failed to load bot personas: {e}")
            logger.warning("API will start but conversations cannot be started")
    
    logger.info("✅ SoulMatch API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SoulMatch API...")
    
    # Clean up all sessions
    active_count = session_manager.get_active_sessions_count()
    if active_count > 0:
        logger.info(f"Cleaning up {active_count} active sessions...")
        # Could save session state here if needed
    
    logger.info("✅ SoulMatch API shutdown complete")


# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="SoulMatch API",
    description="AI-powered dating app backend with persona-based conversations",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Health Check
# ============================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint
    
    Returns API status and bot personas count
    """
    bot_count = 0
    if session_manager.bot_personas_pool:
        bot_count = len(session_manager.bot_personas_pool)
    
    return HealthResponse(
        status="ok",
        bot_personas_loaded=bot_count
    )


# ============================================
# Session Management Endpoints
# ============================================

@app.post("/api/v1/session/start", response_model=SessionStartResponse, tags=["Session"])
async def start_session(request: SessionStartRequest):
    """
    Start a new session and conversation
    
    Creates a session and immediately starts a conversation with a matched bot
    """
    try:
        # Create session
        session_id = session_manager.create_session(
            user_id=request.user_id,
            use_claude=request.use_claude
        )
        
        # Start conversation immediately
        result = await chat_handler.handle_start_conversation(request.user_id)
        
        if result.get("success"):
            return SessionStartResponse(
                success=True,
                session_id=session_id,
                message=f"Session created and conversation started with {result.get('bot_id')}"
            )
        else:
            return SessionStartResponse(
                success=False,
                error=result.get("error", "Failed to start conversation")
            )
    
    except Exception as e:
        logger.error(f"Error starting session for user {request.user_id}: {e}")
        return SessionStartResponse(
            success=False,
            error=str(e)
        )


@app.get("/api/v1/session/{session_id}", response_model=SessionInfoResponse, tags=["Session"])
async def get_session_info(session_id: str):
    """
    Get information about a session
    
    Returns session metadata including creation time and last active time
    """
    try:
        info = session_manager.get_session_info(session_id)
        
        if info:
            return SessionInfoResponse(
                success=True,
                session_id=info["session_id"],
                created_at=info["created_at"],
                last_active=info["last_active"]
            )
        else:
            return SessionInfoResponse(
                success=False,
                error="Session not found"
            )
    
    except Exception as e:
        logger.error(f"Error getting session info for {session_id}: {e}")
        return SessionInfoResponse(
            success=False,
            error=str(e)
        )


@app.delete("/api/v1/session/{session_id}", tags=["Session"])
async def delete_session(session_id: str):
    """
    Delete a session
    
    Removes session and cleans up resources
    """
    try:
        success = session_manager.delete_session(session_id)
        
        if success:
            return JSONResponse(
                content={
                    "success": True,
                    "message": f"Session {session_id} deleted"
                }
            )
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# User Features Endpoint
# ============================================

@app.get("/api/v1/users/{user_id}/summary", response_model=UserSummaryResponse, tags=["Users"])
async def get_user_summary(user_id: str):
    """
    Get user's predicted feature summary
    
    Returns personality traits, interests, and other inferred characteristics
    """
    try:
        result = await chat_handler.get_user_features(user_id)
        
        if result.get("success"):
            return UserSummaryResponse(
                success=True,
                user_id=user_id,
                summary=result.get("features")
            )
        else:
            return UserSummaryResponse(
                success=False,
                user_id=user_id,
                error=result.get("error", "Failed to get user summary")
            )
    
    except Exception as e:
        logger.error(f"Error getting user summary for {user_id}: {e}")
        return UserSummaryResponse(
            success=False,
            user_id=user_id,
            error=str(e)
        )


# ============================================
# WebSocket Endpoint
# ============================================

@app.websocket("/ws/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time chat
    
    Connect via: ws://localhost:8000/ws/{user_id}
    
    Protocol:
        - Client sends: {"action": "start"} to start conversation
        - Client sends: {"action": "message", "content": "..."} to chat
        - Server sends various message types (bot_message, emotion, warning, etc.)
    """
    await websocket_endpoint(websocket, user_id)


# ============================================
# Admin/Debug Endpoints (optional)
# ============================================

@app.get("/api/v1/admin/sessions", tags=["Admin"])
async def list_sessions():
    """
    List all active sessions (admin endpoint)
    """
    try:
        sessions = session_manager.list_all_sessions()
        return {
            "success": True,
            "count": len(sessions),
            "sessions": sessions
        }
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/admin/cleanup", tags=["Admin"])
async def cleanup_inactive_sessions(timeout_seconds: int = 3600):
    """
    Clean up inactive sessions (admin endpoint)
    
    Args:
        timeout_seconds: Inactivity timeout in seconds (default: 1 hour)
    """
    try:
        count = session_manager.cleanup_inactive_sessions(timeout_seconds)
        return {
            "success": True,
            "message": f"Cleaned up {count} inactive sessions"
        }
    except Exception as e:
        logger.error(f"Error cleaning up sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Root Endpoint
# ============================================

@app.get("/", tags=["System"])
async def root():
    """
    Root endpoint - API info
    """
    return {
        "name": "SoulMatch API",
        "version": "1.0.0",
        "description": "AI-powered dating app backend",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "websocket": "/ws/{user_id}"
        }
    }


# ============================================
# Run with: uvicorn src.api.main:app --reload
# ============================================
