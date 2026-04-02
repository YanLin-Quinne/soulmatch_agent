"""FastAPI main application - AI YOU backend API"""

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from loguru import logger

from src.config import settings
from src.api.session_manager import session_manager
from src.api.chat_handler import chat_handler
from src.api.websocket import websocket_endpoint
from src.api.file_handler import extract_text
from src.bootstrap import create_default_bootstrap, get_session_manager as _bootstrap_get_sm
from src.agents.llm_router import router


# ============================================
# Pydantic Models
# ============================================

class SessionStartRequest(BaseModel):
    """Request to start a new session"""
    user_id: str = Field(..., description="Unique user identifier")


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


class UserAggregateResponse(BaseModel):
    """User aggregate data"""
    success: bool
    data: dict | None = None
    error: str | None = None


class UserSummaryResponse(BaseModel):
    """User feature summary"""
    success: bool
    user_id: str
    summary: dict | None = None
    error: str | None = None


class ProfilingStartRequest(BaseModel):
    """Start profiling mode session"""
    persona_id: int = Field(..., description="Persona ID (0-14)")


class ProfilingMessageRequest(BaseModel):
    """Send message in profiling mode"""
    session_id: str
    message: str


class PlaygroundStartRequest(BaseModel):
    """Start playground mode game"""
    user_id: str


class PlaygroundGuessRequest(BaseModel):
    """Submit guess in playground mode"""
    session_id: str
    guess_persona_id: int


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
    logger.info("Starting up AI YOU API...")

    bootstrap = create_default_bootstrap()
    results = await bootstrap.run_all()
    for r in results:
        if not r.success:
            logger.error(f"Bootstrap stage {r.stage.name} failed: {r.error}")
    app.state.bootstrap = bootstrap

    logger.info("✅ AI YOU API started successfully")

    # Start background session cleanup task (every 30 minutes)
    async def _periodic_session_cleanup():
        while True:
            await asyncio.sleep(1800)  # 30 minutes
            try:
                count = session_manager.cleanup_inactive_sessions(timeout_seconds=3600)
                if count > 0:
                    logger.info(f"Periodic cleanup: removed {count} stale sessions")
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")

    cleanup_task = asyncio.create_task(_periodic_session_cleanup())

    yield

    # Shutdown
    cleanup_task.cancel()
    logger.info("Shutting down AI YOU API...")
    
    # Clean up all sessions
    active_count = session_manager.get_active_sessions_count()
    if active_count > 0:
        logger.info(f"Cleaning up {active_count} active sessions...")
        # Could save session state here if needed
    
    logger.info("✅ AI YOU API shutdown complete")


# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="AI YOU API",
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

@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint
    
    Returns API status and bot personas count
    """
    bot_count = 0
    if session_manager.bot_personas_pool:
        bot_count = len(session_manager.bot_personas_pool)

    bootstrap_results = (
        app.state.bootstrap.get_results() if hasattr(app.state, "bootstrap") else {}
    )

    return {
        "status": "ok",
        "version": "1.0.0",
        "bot_personas_loaded": bot_count,
        "bootstrap": bootstrap_results,
    }


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
        session_id = session_manager.create_session(user_id=request.user_id)
        
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
# File Upload Endpoint
# ============================================

@app.post("/api/v1/upload", tags=["Upload"])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file (PDF, TXT, DOCX) and extract text content.

    Returns extracted text for injection into chat context.
    Max file size: 5MB. Max text length: 10000 characters.
    """
    text = await extract_text(file)
    return {
        "success": True,
        "text": text,
        "char_count": len(text),
        "filename": file.filename,
    }


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

@app.get("/api/v1/admin/usage", tags=["Admin"])
async def get_llm_usage():
    """Get LLM usage statistics across all providers."""
    return {"success": True, "usage": router.get_usage_report()}


@app.get("/api/v1/admin/tools", tags=["Admin"])
async def list_tools():
    """List all registered agent tools."""
    from src.agents.tools.registry import tool_registry
    return {
        "success": True,
        "count": len(tool_registry),
        "tools": tool_registry.get_schemas(),
    }


@app.get("/api", tags=["System"])
async def api_root():
    """
    API info endpoint
    """
    return {
        "name": "AI YOU API",
        "version": "2.0.0",
        "description": "AI-powered dating prediction agent with multi-LLM routing and tool calling",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "websocket": "/ws/{user_id}",
            "usage": "/api/v1/admin/usage",
            "tools": "/api/v1/admin/tools",
        }
    }


# ============================================
# Static Files (Frontend)
# ============================================

# Mount static files if frontend build exists
frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/")
    async def serve_frontend():
        """Serve frontend index.html"""
        return FileResponse(frontend_dist / "index.html")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve SPA - return index.html for all non-API routes"""
        if full_path.startswith("api/") or full_path.startswith("ws/") or full_path == "health" or full_path == "docs":
            raise HTTPException(status_code=404)
        file_path = frontend_dist / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")
else:
    @app.get("/")
    async def root():
        """Root endpoint when frontend not built"""
        return {
            "name": "AI YOU API",
            "version": "2.0.0",
            "message": "Frontend not built. Run: cd frontend && npm install && npm run build"
        }


# ============================================
# Run with: uvicorn src.api.main:app --reload
# ============================================


@app.get("/api/v1/debug/personas-file", tags=["Debug"])
async def check_personas_file():
    """Debug endpoint to check if personas file exists"""
    personas_path = settings.data_dir / "processed" / "bot_personas.json"
    return {
        "personas_path": str(personas_path),
        "exists": personas_path.exists(),
        "data_dir": str(settings.data_dir),
        "data_dir_exists": settings.data_dir.exists(),
        "processed_dir_exists": (settings.data_dir / "processed").exists(),
        "files_in_processed": list((settings.data_dir / "processed").iterdir()) if (settings.data_dir / "processed").exists() else []
    }


# ============================================
# Profiling Mode Endpoints
# ============================================

@app.post("/api/profiling/start", tags=["Profiling"])
async def start_profiling_session(request: ProfilingStartRequest):
    """Start a profiling mode session (30-turn inference)"""
    try:
        user_id = f"profiling_{request.persona_id}_{int(time.time())}"
        session_id = session_manager.create_session(user_id=user_id, bot_id=str(request.persona_id))
        return {"success": True, "session_id": session_id, "persona_id": request.persona_id}
    except Exception as e:
        logger.error(f"Error starting profiling session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/profiling/message", tags=["Profiling"])
async def send_profiling_message(request: ProfilingMessageRequest):
    """Send message and get inference update"""
    try:
        orchestrator = session_manager.get_session(request.session_id)
        if not orchestrator:
            raise HTTPException(status_code=404, detail="Session not found")

        result = await orchestrator.process_user_message(request.message)

        return {
            "success": True,
            "bot_message": result.get("bot_message"),
            "turn": result.get("turn"),
            "inferred_traits": orchestrator.ctx.predicted_features,
            "confidence": orchestrator.feature_agent._compute_overall_confidence()
        }
    except Exception as e:
        logger.error(f"Error processing profiling message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/profiling/inference/{session_id}", tags=["Profiling"])
async def get_inference_result(session_id: str):
    """Get current inference result"""
    try:
        orchestrator = session_manager.get_session(session_id)
        if not orchestrator:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "success": True,
            "inferred_traits": orchestrator.ctx.predicted_features,
            "confidence": orchestrator.feature_agent._compute_overall_confidence(),
            "turn_count": orchestrator.ctx.turn_count
        }
    except Exception as e:
        logger.error(f"Error getting inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Playground Mode Endpoints
# ============================================

@app.post("/api/playground/start", tags=["Playground"])
async def start_playground_game(request: PlaygroundStartRequest):
    """Start a playground mode game (10-turn guessing)"""
    try:
        import random
        target_persona = random.randint(0, 14)
        session_id = session_manager.create_session(user_id=request.user_id, bot_id=str(target_persona))

        return {
            "success": True,
            "session_id": session_id,
            "message": "Guess which persona I am in 10 messages!"
        }
    except Exception as e:
        logger.error(f"Error starting playground game: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/playground/guess", tags=["Playground"])
async def submit_playground_guess(request: PlaygroundGuessRequest):
    """Submit a guess and get score"""
    try:
        orchestrator = session_manager.get_session(request.session_id)
        if not orchestrator:
            raise HTTPException(status_code=404, detail="Session not found")

        actual_persona_id = int(orchestrator.preferred_bot_id or 0)
        correct = (request.guess_persona_id == actual_persona_id)

        return {
            "success": True,
            "correct": correct,
            "actual_persona_id": actual_persona_id,
            "score": 100 if correct else max(0, 100 - abs(request.guess_persona_id - actual_persona_id) * 10)
        }
    except Exception as e:
        logger.error(f"Error submitting guess: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Visualization Endpoints
# ============================================

@app.get("/api/pipeline/status/{session_id}", tags=["Visualization"])
async def get_pipeline_status(session_id: str):
    """Get agent pipeline status for visualization"""
    try:
        orchestrator = session_manager.get_session(session_id)
        if not orchestrator:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "success": True,
            "pipeline": {
                "emotion_agent": {"status": "active", "last_result": orchestrator.ctx.current_emotion},
                "scam_agent": {"status": "active", "risk_score": orchestrator.ctx.scam_risk_score},
                "feature_agent": {"status": "active", "confidence": orchestrator.feature_agent._compute_overall_confidence()},
                "memory_manager": {"status": "active", "stats": orchestrator.memory_manager.get_memory_stats()},
                "persona_agent": {"status": "active", "bot_id": orchestrator.preferred_bot_id}
            }
        }
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logic-tree/{session_id}", tags=["Visualization"])
async def get_logic_tree(session_id: str):
    """Get reasoning logic tree"""
    try:
        orchestrator = session_manager.get_session(session_id)
        if not orchestrator:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "success": True,
            "tree": {
                "root": "User personality inference",
                "branches": [
                    {"node": "Emotion analysis", "confidence": orchestrator.ctx.emotion_confidence},
                    {"node": "Feature prediction", "confidence": orchestrator.feature_agent._compute_overall_confidence()},
                    {"node": "Memory retrieval", "count": orchestrator.memory_manager.get_memory_stats().get("total_memories", 0)}
                ]
            }
        }
    except Exception as e:
        logger.error(f"Error getting logic tree: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/research/aggregate/{user_id}", tags=["Research"])
async def get_user_aggregate(user_id: str) -> UserAggregateResponse:
    """获取用户聚合数据"""
    try:
        from src.persistence.session_store import SessionStore
        store = SessionStore()
        data = store.aggregate_user_sessions(user_id)
        return UserAggregateResponse(success=True, data=data)
    except Exception as e:
        logger.error(f"Error aggregating user data: {e}")
        return UserAggregateResponse(success=False, error=str(e))
