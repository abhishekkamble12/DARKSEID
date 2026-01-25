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
from langchain_core.documents import Document

# Qdrant imports
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams

# Document processing
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "hybrid_rag_docs")

app = FastAPI(title="Hybrid Multi-modal RAG API", version="3.0.0")

# =============================================================================
# Global Instances
# =============================================================================

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

class QdrantManager:
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            cls._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        return cls._client

    @classmethod
    def init_collection(cls):
        client = cls.get_client()
        if not client.collection_exists(QDRANT_COLLECTION):
            # HYBRID CONFIGURATION: Dense (768) + Sparse
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                sparse_vectors_config={
                    "text-sparse": SparseVectorParams() # Required for BM25/Hybrid
                }
            )
            print(f"Initialized Hybrid Collection: {QDRANT_COLLECTION}")

def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=QdrantManager.get_client(),
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
        # This enables Hybrid search within the LangChain wrapper
        retrieval_mode=QdrantVectorStore.HYBRID 
    )

# =============================================================================
# API Models
# =============================================================================

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    mode: str = "hybrid"

# =============================================================================
# Endpoints
# =============================================================================

@app.on_event("startup")
async def startup():
    QdrantManager.init_collection()

@app.post("/index_pdf", tags=["Ingestion"])
async def index_pdf(file: UploadFile = File(...), session_id: str = "default"):
    temp_path = f"temp_{uuid.uuid4()}_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Logic for PDF Extraction
        reader = PdfReader(temp_path)
        full_text = "\n\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        
        if not full_text:
            raise HTTPException(400, "No text found in PDF")

        # Splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_text(full_text)

        # Batch indexing
        docs = [
            Document(
                page_content=chunk,
                metadata={"session_id": session_id, "source": file.filename}
            ) for chunk in chunks
        ]
        
        vs = get_vector_store()
        vs.add_documents(docs)

        return {"status": "indexed", "chunks": len(docs)}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/query_rag", response_model=QueryResponse, tags=["Search"])
async def query_rag(request: QueryRequest):
    vs = get_vector_store()
    
    # Robust Hybrid Search with Metadata Filtering
    # Note: Filter structure depends on whether you stored session_id inside 'metadata' dict
    search_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.session_id", 
                match=models.MatchValue(value=request.session_id)
            )
        ]
    )

    # Hybrid Search (Combines Vector + BM25)
    docs = vs.similarity_search(
        request.question,
        k=request.top_k,
        filter=search_filter
    )

    if not docs:
        return QueryResponse(answer="No context found.", sources=[])

    context = "\n\n".join([d.page_content for d in docs])
    sources = list(set([d.metadata.get("source", "unknown") for d in docs]))

    prompt = ChatPromptTemplate.from_template("""
    You are an expert analyst. Answer based ONLY on the context below.
    Context: {context}
    Question: {question}
    """)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": request.question})

    return QueryResponse(answer=answer, sources=sources)