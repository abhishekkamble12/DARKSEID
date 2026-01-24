"""
Multi-modal RAG API Service with Qdrant
========================================
This FastAPI service provides RAG (Retrieval Augmented Generation) capabilities
for processing PDFs, images, and tables using Qdrant as the vector store.

Endpoints:
- POST /index_pdf: Index a PDF file (extracts text, tables, images)
- POST /query_rag: Query the indexed documents

Usage:
    uvicorn Rag:app --host 0.0.0.0 --port 8000
"""

import os
import uuid
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

# Document processing - using PyPDF2 (no poppler required)
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents")

# Initialize FastAPI
app = FastAPI(
    title="Multi-modal RAG API",
    description="RAG service with Qdrant for document Q&A",
    version="2.0.0"
)

# =============================================================================
# Initialize Models and Vector Store
# =============================================================================

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Qdrant client
qdrant_client = None
vector_store = None


def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client."""
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return qdrant_client


def initialize_collection():
    """Initialize Qdrant collection if it doesn't exist."""
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    
    if QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        print(f"[OK] Created Qdrant collection: {QDRANT_COLLECTION}")
    
    return client


def get_vector_store() -> QdrantVectorStore:
    """Get the Qdrant vector store."""
    global vector_store
    if vector_store is None:
        client = initialize_collection()
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embedding=embeddings,
        )
    return vector_store


# =============================================================================
# Helper Functions
# =============================================================================

def generate_summaries(elements: list, is_table: bool = False) -> List[str]:
    """Generate summaries for text or table elements."""
    prompt_text = "Summarize this {type}: {element}. Respond only with the summary."
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()
    
    summaries = []
    for el in elements:
        try:
            content = el.metadata.text_as_html if is_table else el.text
            element_type = "table" if is_table else "text"
            summary = chain.invoke({"type": element_type, "element": content})
            summaries.append(summary)
        except Exception as e:
            summaries.append(str(el))
    
    return summaries


def generate_image_summaries(images_b64: list) -> List[str]:
    """Generate summaries for images using vision capabilities."""
    summaries = []
    
    for img_b64 in images_b64:
        try:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Describe this image from a document in detail. Focus on technical content, data, or diagrams."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            )
            response = llm.invoke([message])
            summaries.append(response.content if isinstance(response.content, str) else str(response.content))
        except Exception as e:
            summaries.append(f"[Image - could not process: {str(e)}]")
    
    return summaries


# =============================================================================
# API Models
# =============================================================================

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    answer: str
    relevant_texts: List[str]
    sources: List[str]
    image_count: int


class IndexResponse(BaseModel):
    status: str
    message: str
    text_chunks: int
    table_chunks: int
    image_chunks: int


# =============================================================================
# Startup Event
# =============================================================================

@app.on_event("startup")
async def startup():
    """Initialize connections on startup."""
    try:
        initialize_collection()
        print(f"[OK] Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    except Exception as e:
        print(f"[WARN] Qdrant connection failed: {e}")


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        return {
            "status": "healthy",
            "qdrant": "connected",
            "collections": len(collections.collections)
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# =============================================================================
# Endpoints
# =============================================================================

@app.post("/index_pdf", response_model=IndexResponse, tags=["Tools"])
async def index_pdf(
    file: UploadFile = File(...),
    session_id: str = "default"
):
    """
    Index a PDF by extracting text using PyPDF2.
    Stores embeddings in Qdrant for later retrieval.
    """
    temp_path = None
    try:
        # Save temp file
        temp_path = f"temp_{uuid.uuid4()}_{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Extract text using PyPDF2
        reader = PdfReader(temp_path)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n\n"

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF. The PDF might be scanned/image-based.")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(full_text)

        vs = get_vector_store()
        text_count = 0

        # Index text chunks
        if chunks:
            docs = [
                Document(
                    page_content=chunk,
                    metadata={
                        "session_id": session_id,
                        "source_file": file.filename,
                        "type": "text",
                        "chunk_index": i
                    }
                )
                for i, chunk in enumerate(chunks)
            ]
            vs.add_documents(docs)
            text_count = len(docs)

        # Cleanup
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return IndexResponse(
            status="success",
            message=f"Indexed {file.filename} successfully ({text_count} chunks)",
            text_chunks=text_count,
            table_chunks=0,
            image_chunks=0
        )

    except HTTPException:
        raise
    except Exception as e:
        # Cleanup on error
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query_rag", response_model=QueryResponse, tags=["Tools"])
async def query_rag(request: QueryRequest):
    """
    Query the indexed documents using RAG.
    Returns an answer based on the relevant context from stored documents.
    """
    try:
        vs = get_vector_store()
        
        # Search with session filter using proper Qdrant filter format
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.session_id",
                    match=MatchValue(value=request.session_id)
                )
            ]
        )
        docs = vs.similarity_search(
            request.question,
            k=request.top_k,
            filter=qdrant_filter
        )

        if not docs:
            return QueryResponse(
                answer="No relevant documents found. Please upload and index documents first.",
                relevant_texts=[],
                sources=[],
                image_count=0
            )

        # Prepare context
        context_parts = []
        sources = set()
        image_count = 0
        
        for doc in docs:
            context_parts.append(doc.page_content)
            sources.add(doc.metadata.get("source_file", "Unknown"))
            if doc.metadata.get("type") == "image":
                image_count += 1

        context = "\n\n".join(context_parts)

        # Generate answer
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful document Q&A assistant. 
Answer the question based ONLY on the provided context.
If the answer is not in the context, say "I couldn't find this information in the documents."
Cite the source documents when relevant.

Context:
{context}

Sources: {sources}"""),
            ("human", "{question}")
        ])

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "sources": ", ".join(sources),
            "question": request.question
        })

        return QueryResponse(
            answer=answer,
            relevant_texts=context_parts[:3],
            sources=list(sources),
            image_count=image_count
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear_session/{session_id}", tags=["Admin"])
async def clear_session(session_id: str):
    """Clear all documents for a specific session."""
    try:
        client = get_qdrant_client()
        # Delete points with matching session_id
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector={
                "filter": {
                    "must": [
                        {"key": "session_id", "match": {"value": session_id}}
                    ]
                }
            }
        )
        return {"status": "success", "message": f"Cleared session: {session_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
