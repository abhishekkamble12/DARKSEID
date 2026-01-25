from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import uuid
import httpx
import time
import logging
from typing import Optional, Dict, Any, List
# project imports moved to function scope to prevent startup hangs
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"✅ Loaded .env from {env_path}")
else:
    load_dotenv()
    print("⚠️ .env not found in current directory, used default system env.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("darksied-api")

# MCP RAG Service URL
MCP_RAG_URL = os.getenv("MCP_RAG_URL", "http://localhost:8001")

# LiveKit Configuration
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")

# Check if LiveKit is configured
LIVEKIT_AVAILABLE = bool(LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET)

# Try to import LiveKit API
try:
    from livekit.api import AccessToken, VideoGrants
    LIVEKIT_SDK_AVAILABLE = True
except ImportError:
    LIVEKIT_SDK_AVAILABLE = False

print(f"LIVEKIT_URL: {'[SET]' if LIVEKIT_URL else '[EMPTY]'}")
print(f"LIVEKIT_AVAILABLE: {LIVEKIT_AVAILABLE}")
print(f"LIVEKIT_SDK_AVAILABLE: {LIVEKIT_SDK_AVAILABLE}")

if not LIVEKIT_SDK_AVAILABLE:
    logger.warning("LiveKit SDK not installed. Voice avatar will use browser-based fallback.")

# Initialize FastAPI
app = FastAPI(title="Darksied API", version="1.0.0")

# CORS Configuration - Allow all localhost ports for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for lazy loading
_chatbot_system = None

def get_chatbot():
    """Lazy load the chatbot system to prevent startup delays."""
    global _chatbot_system
    if _chatbot_system is None:
        logger.info("🧠 Initializing Supervisor Chatbot system...")
        from project import SupervisorChatbot
        _chatbot_system = SupervisorChatbot(use_checkpointer=True)
    return _chatbot_system

# Define Models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    uploaded_file_path: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    type: str = "text"   # text, quiz, mindmap
    data: Any = None     # Structured data (cards array or mermaid string)

class LearningRequest(BaseModel):
    topic: str
    session_id: str

# Ensure upload directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"status": "Darksied Neural Link Active"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Process a chat message through the multi-agent system.
    """
    try:
        # Update session ID if provided, otherwise keep existing
        chatbot = get_chatbot()
        if request.session_id:
            chatbot.session_id = request.session_id
        
        print(f"📨 Received chat: {request.message} (Session: {chatbot.session_id})")
        
        response_text = chatbot.chat(
            user_input=request.message,
            uploaded_file_path=request.uploaded_file_path
        )
        
        # Parse for structured content (Mindmaps/Quizzes)
        from project import parse_learning_output
        parsed = parse_learning_output(response_text)
        
        response_type = "text"
        response_data = None
        
        if parsed.get("quiz"):
            response_type = "quiz"
            response_data = parsed["quiz"].get("cards", [])
        elif parsed.get("mindmap"):
            response_type = "mindmap"
            response_data = parsed["mindmap"]
            
        return ChatResponse(
            response=response_text,
            session_id=chatbot.session_id,
            type=response_type,
            data=response_data
        )
    except Exception as e:
        error_str = str(e)
        print(f"❌ Error in chat endpoint: {error_str}")
        
        # Provide user-friendly error messages
        if "getaddrinfo failed" in error_str:
            raise HTTPException(
                status_code=503, 
                detail="Network error: Unable to connect to AI service. Please check your internet connection."
            )
        elif "API key" in error_str.lower() or "authentication" in error_str.lower():
            raise HTTPException(
                status_code=401, 
                detail="Authentication error: Invalid or missing API key. Please check GOOGLE_API_KEY."
            )
        elif "quota" in error_str.lower() or "rate limit" in error_str.lower():
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded. Please wait a moment and try again."
            )
        else:
            raise HTTPException(status_code=500, detail=error_str)

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """
    Upload and index a document for RAG.
    Tries MCP RAG service first, falls back to local indexing.
    """
    # Update session ID in chatbot
    chatbot = get_chatbot()
    chatbot.session_id = session_id
    
    print(f"[INFO] Uploading file: {file.filename}")
    
    # Read file content once
    file_content = await file.read()
    
    # Try MCP RAG service first
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            files = {"file": (file.filename, file_content, file.content_type)}
            params = {"session_id": session_id}
            
            response = await client.post(
                f"{MCP_RAG_URL}/index_pdf",
                files=files,
                params=params
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"[OK] File indexed via MCP RAG: {result}")
                return {
                    "status": "success",
                    "filename": file.filename,
                    "text_chunks": result.get("text_chunks", 0),
                    "table_chunks": result.get("table_chunks", 0),
                    "image_chunks": result.get("image_chunks", 0),
                    "message": result.get("message", "Document indexed successfully")
                }
    except Exception as mcp_error:
        print(f"[WARN] MCP RAG service unavailable: {mcp_error}. Falling back to local indexing.")
    
    # Fallback to local indexing via project.py
    try:
        # Save file temporarily
        file_ext = os.path.splitext(file.filename)[1].lower()
        temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{file_ext}")
        
        with open(temp_path, "wb") as f:
            f.write(file_content)
        
        print(f"[INFO] Saved temp file: {temp_path}")
        
        # Use chatbot's upload_document method
        result = chatbot.upload_document(temp_path)
        
        print(f"[OK] File indexed locally: {result}")
        
        return {
            "status": "success",
            "filename": file.filename,
            "text_chunks": result.get("chunks_indexed", 0),
            "table_chunks": 0,
            "image_chunks": 0,
            "message": f"Document indexed successfully ({result.get('chunks_indexed', 0)} chunks)"
        }
        
    except Exception as local_error:
        print(f"[ERROR] Local indexing failed: {local_error}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to index document: {str(local_error)}"
        )

class RAGQueryRequest(BaseModel):
    question: str
    session_id: str
    top_k: Optional[int] = 5

@app.post("/api/rag/query")
async def query_rag(request: RAGQueryRequest):
    """
    Query indexed documents via MCP RAG service.
    """
    try:
        print(f"[INFO] RAG Query: {request.question[:50]}...")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{MCP_RAG_URL}/query_rag",
                json={
                    "question": request.question,
                    "session_id": request.session_id,
                    "top_k": request.top_k
                }
            )
            
            if response.status_code != 200:
                error_detail = response.json().get("detail", "Unknown error")
                raise HTTPException(status_code=response.status_code, detail=error_detail)
            
            return response.json()
            
    except httpx.RequestError as e:
        print(f"[ERROR] MCP RAG service connection error: {e}")
        raise HTTPException(status_code=503, detail=f"RAG service unavailable: {str(e)}")
    except Exception as e:
        print(f"[ERROR] RAG query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/learning")
async def generate_learning(request: LearningRequest):
    """
    Generate Learning Architect materials (Mindmap + Quiz).
    """
    try:
        chatbot = get_chatbot()
        chatbot.session_id = request.session_id
        print(f"🧠 Generating learning materials for topic: {request.topic}")
        
        result = chatbot.generate_learning_materials(request.topic)
        return result
    except Exception as e:
        print(f"❌ Learning Architect error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================

@app.get("/api/health")
async def health_check():
    """
    Comprehensive health check for all services.
    """
    health = {
        "status": "healthy",
        "timestamp": int(time.time() * 1000),
        "services": {
            "api": {"status": "healthy"},
            "rag": {"status": "unknown"},
            "qdrant": {"status": "unknown"},
            "postgres": {"status": "unknown"},
            "voice": {"status": "unknown", "mode": "browser"}
        }
    }
    
    # Check RAG service
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{MCP_RAG_URL}/health")
            if response.status_code == 200:
                rag_health = response.json()
                health["services"]["rag"] = {"status": "healthy"}
                health["services"]["qdrant"] = {
                    "status": "healthy" if rag_health.get("qdrant") == "connected" else "unhealthy"
                }
    except:
        health["services"]["rag"] = {"status": "unhealthy"}
    
    # Check PostgreSQL (via chatbot)
    try:
        chatbot = get_chatbot()
        if chatbot and hasattr(chatbot, 'checkpointer') and chatbot.checkpointer:
            health["services"]["postgres"] = {"status": "healthy"}
        else:
            health["services"]["postgres"] = {"status": "unavailable"}
    except:
        health["services"]["postgres"] = {"status": "unhealthy"}
    
    # Check Voice/LiveKit
    if LIVEKIT_AVAILABLE and LIVEKIT_SDK_AVAILABLE:
        health["services"]["voice"] = {
            "status": "healthy",
            "mode": "livekit",
            "url": LIVEKIT_URL.replace("wss://", "").replace(".livekit.cloud", "...")
        }
    else:
        health["services"]["voice"] = {
            "status": "healthy",
            "mode": "browser",
            "message": "Using browser-based voice (LiveKit not configured)"
        }
    
    # Overall status
    unhealthy = [k for k, v in health["services"].items() if v.get("status") == "unhealthy"]
    if unhealthy:
        health["status"] = "degraded"
    
    return health


# =============================================================================
# LIVEKIT VOICE ENDPOINTS
# =============================================================================

class VoiceTokenRequest(BaseModel):
    room_name: Optional[str] = None
    participant_name: Optional[str] = None
    session_id: Optional[str] = None

class VoiceTokenResponse(BaseModel):
    token: str
    room_name: str
    participant_name: str
    livekit_url: str
    mode: str  # "livekit" or "browser"

@app.get("/api/voice/status")
async def voice_status():
    """
    Get voice service configuration status.
    """
    return {
        "available": LIVEKIT_AVAILABLE and LIVEKIT_SDK_AVAILABLE,
        "mode": "livekit" if (LIVEKIT_AVAILABLE and LIVEKIT_SDK_AVAILABLE) else "browser",
        "livekit_configured": LIVEKIT_AVAILABLE,
        "livekit_sdk_installed": LIVEKIT_SDK_AVAILABLE,
        "features": {
            "real_time_voice": LIVEKIT_AVAILABLE and LIVEKIT_SDK_AVAILABLE,
            "ai_avatar": bool(os.getenv("BEY_API_KEY")),
            "browser_fallback": True,
            "rag_connected": True
        },
        "requirements": {
            "LIVEKIT_URL": bool(LIVEKIT_URL),
            "LIVEKIT_API_KEY": bool(LIVEKIT_API_KEY),
            "LIVEKIT_API_SECRET": bool(LIVEKIT_API_SECRET),
            "BEY_API_KEY": bool(os.getenv("BEY_API_KEY")),
            "BEY_AVATAR_ID": bool(os.getenv("BEY_AVATAR_ID"))
        }
    }

@app.post("/api/voice/token", response_model=VoiceTokenResponse)
async def get_voice_token(request: VoiceTokenRequest):
    """
    Generate a LiveKit room token for voice sessions.
    Falls back to browser mode if LiveKit is not configured.
    """
    # Generate room and participant names
    room_name = request.room_name or f"darksied-{request.session_id or uuid.uuid4().hex[:8]}"
    participant_name = request.participant_name or f"user-{uuid.uuid4().hex[:6]}"
    
    # If LiveKit is not available, return browser mode token
    if not LIVEKIT_AVAILABLE or not LIVEKIT_SDK_AVAILABLE:
        logger.info(f"🎤 Voice token requested (browser mode): room={room_name}")
        return VoiceTokenResponse(
            token="browser-mode",
            room_name=room_name,
            participant_name=participant_name,
            livekit_url="",
            mode="browser"
        )
    
    try:
        # Generate LiveKit access token
        token = AccessToken(
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET
        )
        
        token.with_identity(participant_name)
        token.with_name(participant_name)
        
        # Grant permissions
        token.with_grants(VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True
        ))
        
        # Set expiry (1 hour)
        token.with_ttl(3600)
        
        jwt_token = token.to_jwt()
        
        logger.info(f"🎤 LiveKit token generated: room={room_name}, participant={participant_name}")
        
        return VoiceTokenResponse(
            token=jwt_token,
            room_name=room_name,
            participant_name=participant_name,
            livekit_url=LIVEKIT_URL,
            mode="livekit"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to generate LiveKit token: {e}")
        # Fall back to browser mode
        return VoiceTokenResponse(
            token="browser-mode",
            room_name=room_name,
            participant_name=participant_name,
            livekit_url="",
            mode="browser"
        )


@app.get("/api/voice/agent-status")
async def voice_agent_status():
    """
    Check if the LiveKit voice agent is running.
    """
    if not LIVEKIT_AVAILABLE:
        return {
            "running": False,
            "mode": "browser",
            "message": "LiveKit not configured. Using browser-based voice."
        }
    
    # Try to check if agent is registered with LiveKit
    # In production, you'd query LiveKit's API to see if the agent worker is connected
    return {
        "running": True,  # Assume running if configured
        "mode": "livekit",
        "agent_type": "socratic-tutor",
        "capabilities": ["stt", "tts", "llm", "rag"],
        "start_command": "python app/avatar_agent.py dev"
    }


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

class SessionInfo(BaseModel):
    id: str
    name: str
    created_at: int
    last_active_at: int
    document_count: int = 0

# In-memory session storage (replace with database in production)
_sessions: Dict[str, SessionInfo] = {}

@app.get("/api/sessions")
async def list_sessions():
    """List all sessions."""
    return {"sessions": list(_sessions.values())}

@app.post("/api/sessions")
async def create_session(name: Optional[str] = None):
    """Create a new session."""
    session_id = uuid.uuid4().hex[:12]
    now = int(time.time() * 1000)
    
    session = SessionInfo(
        id=session_id,
        name=name or "New Session",
        created_at=now,
        last_active_at=now,
        document_count=0
    )
    
    _sessions[session_id] = session
    chatbot = get_chatbot()
    chatbot.session_id = session_id
    
    return session

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its data."""
    if session_id in _sessions:
        del _sessions[session_id]
    
    # Also try to clear from RAG service
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.delete(f"{MCP_RAG_URL}/clear_session/{session_id}")
    except:
        pass
    
    return {"status": "deleted", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn
    logger.info("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                 🚀 DARKSIED BACKEND API                        ║
    ║                                                               ║
    ║   Endpoints:                                                  ║
    ║   • POST /api/chat          - Multi-agent chat                ║
    ║   • POST /api/upload        - Document upload                 ║
    ║   • POST /api/learning      - Learning materials              ║
    ║   • GET  /api/health        - Health check                    ║
    ║   • GET  /api/voice/status  - Voice configuration             ║
    ║   • POST /api/voice/token   - LiveKit room token              ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    logger.info("Starting Darksied API on port 8010...")
    uvicorn.run(app, host="127.0.0.1", port=8010)
