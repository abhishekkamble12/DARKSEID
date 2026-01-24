from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import uuid
import httpx
from typing import Optional, Dict, Any
from project import SupervisorChatbot, parse_learning_output

# MCP RAG Service URL
MCP_RAG_URL = os.getenv("MCP_RAG_URL", "http://localhost:8001")

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

# Initialize Chatbot System
# We set use_checkpointer=True to try using Postgres, but it handles failure gracefully.
chatbot_system = SupervisorChatbot(use_checkpointer=True)

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
        if request.session_id:
            chatbot_system.session_id = request.session_id
        
        print(f"📨 Received chat: {request.message} (Session: {chatbot_system.session_id})")
        
        response_text = chatbot_system.chat(
            user_input=request.message,
            uploaded_file_path=request.uploaded_file_path
        )
        
        # Parse for structured content (Mindmaps/Quizzes)
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
            session_id=chatbot_system.session_id,
            type=response_type,
            data=response_data
        )
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """
    Upload and index a document for RAG via MCP RAG service.
    """
    try:
        # Update session ID in chatbot
        chatbot_system.session_id = session_id
        
        print(f"[INFO] Uploading file: {file.filename} to MCP RAG service")
        
        # Forward file to MCP RAG service
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Read file content
            file_content = await file.read()
            
            # Send to MCP RAG service
            files = {"file": (file.filename, file_content, file.content_type)}
            params = {"session_id": session_id}
            
            response = await client.post(
                f"{MCP_RAG_URL}/index_pdf",
                files=files,
                params=params
            )
            
            if response.status_code != 200:
                error_detail = response.json().get("detail", "Unknown error")
                raise HTTPException(status_code=response.status_code, detail=error_detail)
            
            result = response.json()
            
        print(f"[OK] File indexed: {result}")
        
        return {
            "status": "success",
            "filename": file.filename,
            "text_chunks": result.get("text_chunks", 0),
            "table_chunks": result.get("table_chunks", 0),
            "image_chunks": result.get("image_chunks", 0),
            "message": result.get("message", "Document indexed successfully")
        }
    except httpx.RequestError as e:
        print(f"[ERROR] MCP RAG service connection error: {e}")
        raise HTTPException(status_code=503, detail=f"RAG service unavailable: {str(e)}")
    except Exception as e:
        print(f"[ERROR] Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        chatbot_system.session_id = request.session_id
        print(f"🧠 Generating learning materials for topic: {request.topic}")
        
        result = chatbot_system.generate_learning_materials(request.topic)
        return result
    except Exception as e:
        print(f"❌ Learning Architect error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Darksied Backend API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
