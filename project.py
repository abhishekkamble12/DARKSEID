"""
Multi-Agent System with Supervisor + RAG
=========================================
This module implements a LangGraph-based multi-agent system where a Supervisor Agent
routes user queries to specialized agents:

1. Supervisor Agent - Routes queries to the appropriate agent
2. Research Agent - Uses Tavily search for web research
3. Examiner Agent - Generates MCQ questions on topics
4. Chat Agent - Handles general conversation
5. RAG Agent - Handles document Q&A from uploaded PDFs/files

Infrastructure:
- Qdrant: Vector database for document embeddings
- PostgreSQL: Checkpointer for conversation memory persistence

Usage:
    python project.py
"""

import os
import sys
import uuid
import tempfile
from typing import TypedDict, Annotated, Literal, List, Optional
from dotenv import load_dotenv
from pathlib import Path

# Fix Windows console encoding for emoji characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# torch and transformers imported lazily inside get_learning_architect_llm
import json
import re

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)

# Load environment variables
load_dotenv()

# =============================================================================
# OPIK (COMET) TRACING & EVALUATION
# =============================================================================
# Load Opik-related environment variables from .env
OPIK_API_KEY = os.getenv("OPIK_API_KEY")
OPIK_WORKSPACE = os.getenv("OPIK_WORKSPACE")
OPIK_PROJECT_NAME = os.getenv("OPIK_PROJECT_NAME")

# Opik initialization disabled to prevent startup hangs
opik_tracer = None
Hallucination = None
track = lambda fn: fn

# =============================================================================
# CONFIGURATION
# =============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/agent_db")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")

# HuggingFace Configuration for Learning Architect
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")  # Optional: for gated models

# Validate required keys
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables!")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    max_retries=2,
)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# =============================================================================
# HUGGINGFACE MODEL FOR LEARNING ARCHITECT
# =============================================================================

def get_learning_architect_llm():
    """
    Initialize the HuggingFace model for Learning Architect (Mindmap + Quiz generation).
    Uses Qwen/Qwen2.5-7B-Instruct by default.
    """
    try:
        # Lazy imports for Learning Architect
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        from langchain_huggingface import HuggingFacePipeline
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🧠 Loading Learning Architect model on {device}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL_ID,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        
        # Load model with appropriate settings
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            token=HF_TOKEN,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Create pipeline
        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Wrap in LangChain
        hf_llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
        print(f"✅ Learning Architect model loaded: {HF_MODEL_ID}")
        return hf_llm
        
    except Exception as e:
        print(f"⚠️ Could not load HuggingFace model: {e}")
        print("   Falling back to Gemini for Learning Architect.")
        return None

# Lazy initialization - will be loaded when needed
_learning_architect_llm = None

def get_or_init_learning_llm():
    """Get or initialize the Learning Architect LLM (lazy loading)."""
    global _learning_architect_llm
    if _learning_architect_llm is None:
        _learning_architect_llm = get_learning_architect_llm()
    return _learning_architect_llm

# =============================================================================
# QDRANT VECTOR STORE SETUP
# =============================================================================

def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client instance."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def initialize_qdrant_collection():
    """Initialize Qdrant collection if it doesn't exist."""
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    
    if QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        print(f"✅ Created Qdrant collection: {QDRANT_COLLECTION}")
    else:
        print(f"✅ Using existing Qdrant collection: {QDRANT_COLLECTION}")
    
    return client


def get_vector_store() -> QdrantVectorStore:
    """Get the Qdrant vector store instance."""
    client = get_qdrant_client()
    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )


# =============================================================================
# POSTGRESQL CHECKPOINTER SETUP
# =============================================================================

def get_checkpointer():
    """Get PostgreSQL checkpointer for conversation memory."""
    # Create a sync connection with autocommit for setup
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    return checkpointer


# =============================================================================
# SHARED CONTEXT MANAGER - NEVER RUN OUT OF CONTEXT
# =============================================================================

class SharedContextManager:
    """
    Manages shared context across all agents with smart summarization.
    Ensures context never runs out by:
    1. Summarizing old conversations
    2. Extracting key facts from documents
    3. Maintaining rolling window of recent messages
    4. Providing unified context to all agents
    """
    
    # Class-level storage for session contexts
    _sessions: dict = {}
    
    # Configuration
    MAX_RECENT_MESSAGES = 10  # Keep last N messages in full
    MAX_CONTEXT_TOKENS = 8000  # Approximate token limit for context
    SUMMARY_THRESHOLD = 15  # Summarize when messages exceed this
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        
        # Initialize session storage if not exists
        if session_id not in SharedContextManager._sessions:
            SharedContextManager._sessions[session_id] = {
                "document_summaries": {},  # {filename: summary}
                "document_key_facts": [],  # List of key facts
                "conversation_summary": "",  # Summary of older conversation
                "recent_messages": [],  # Recent full messages
                "uploaded_files": [],  # List of uploaded file names
                "extracted_topics": [],  # Topics extracted from documents
                "agent_insights": {},  # Insights from each agent
                "total_messages_processed": 0,
            }
        
        self._data = SharedContextManager._sessions[session_id]
    
    @property
    def document_summaries(self) -> dict:
        return self._data["document_summaries"]
    
    @property
    def key_facts(self) -> List[str]:
        return self._data["document_key_facts"]
    
    @property
    def uploaded_files(self) -> List[str]:
        return self._data["uploaded_files"]
    
    @property
    def conversation_summary(self) -> str:
        return self._data["conversation_summary"]
    
    @property
    def topics(self) -> List[str]:
        return self._data["extracted_topics"]
    
    def add_document(self, filename: str, summary: str, key_facts: List[str], topics: List[str] = None):
        """Add a document's context to shared memory."""
        self._data["document_summaries"][filename] = summary
        self._data["document_key_facts"].extend(key_facts)
        self._data["uploaded_files"].append(filename)
        if topics:
            self._data["extracted_topics"].extend(topics)
        # Deduplicate
        self._data["document_key_facts"] = list(set(self._data["document_key_facts"]))[:50]  # Keep top 50 facts
        self._data["extracted_topics"] = list(set(self._data["extracted_topics"]))[:20]
        print(f"📚 Context updated: {filename} added with {len(key_facts)} facts")
    
    def add_message(self, role: str, content: str):
        """Add a message and manage context size."""
        self._data["recent_messages"].append({"role": role, "content": content})
        self._data["total_messages_processed"] += 1
        
        # Check if we need to summarize
        if len(self._data["recent_messages"]) > self.SUMMARY_THRESHOLD:
            self._summarize_old_messages()
    
    def add_agent_insight(self, agent_name: str, insight: str):
        """Store insights from agent processing."""
        if agent_name not in self._data["agent_insights"]:
            self._data["agent_insights"][agent_name] = []
        self._data["agent_insights"][agent_name].append(insight)
        # Keep last 5 insights per agent
        self._data["agent_insights"][agent_name] = self._data["agent_insights"][agent_name][-5:]
    
    def _summarize_old_messages(self):
        """Summarize older messages to prevent context overflow."""
        if len(self._data["recent_messages"]) <= self.MAX_RECENT_MESSAGES:
            return
        
        # Messages to summarize
        to_summarize = self._data["recent_messages"][:-self.MAX_RECENT_MESSAGES]
        self._data["recent_messages"] = self._data["recent_messages"][-self.MAX_RECENT_MESSAGES:]
        
        # Create summary using LLM
        try:
            summary_prompt = f"""Summarize this conversation concisely, keeping key information:

Previous Summary: {self._data['conversation_summary'][:500] if self._data['conversation_summary'] else 'None'}

New Messages:
{chr(10).join([f"{m['role']}: {m['content'][:200]}" for m in to_summarize])}

Provide a brief summary (max 300 words) capturing:
1. Main topics discussed
2. Key questions asked
3. Important answers given
4. Any decisions or conclusions"""

            response = llm.invoke([HumanMessage(content=summary_prompt)])
            self._data["conversation_summary"] = response.content[:1000]
            print(f"📝 Context summarized: {len(to_summarize)} messages compressed")
        except Exception as e:
            # Fallback: simple concatenation
            self._data["conversation_summary"] += f"\n[Earlier: {len(to_summarize)} messages about various topics]"
    
    def get_unified_context(self, include_docs: bool = True, include_history: bool = True) -> str:
        """
        Get unified context string for any agent.
        This is the KEY method - provides same context to all agents.
        """
        context_parts = []
        
        # Document context
        if include_docs and self._data["uploaded_files"]:
            context_parts.append("=== UPLOADED DOCUMENTS ===")
            context_parts.append(f"Files: {', '.join(self._data['uploaded_files'])}")
            
            if self._data["document_summaries"]:
                context_parts.append("\nDocument Summaries:")
                for filename, summary in self._data["document_summaries"].items():
                    context_parts.append(f"• {filename}: {summary[:300]}...")
            
            if self._data["document_key_facts"]:
                context_parts.append(f"\nKey Facts ({len(self._data['document_key_facts'])} total):")
                for fact in self._data["document_key_facts"][:10]:
                    context_parts.append(f"• {fact}")
            
            if self._data["extracted_topics"]:
                context_parts.append(f"\nTopics: {', '.join(self._data['extracted_topics'][:10])}")
        
        # Conversation history
        if include_history:
            if self._data["conversation_summary"]:
                context_parts.append("\n=== CONVERSATION HISTORY ===")
                context_parts.append(f"Summary: {self._data['conversation_summary'][:500]}")
            
            if self._data["recent_messages"]:
                context_parts.append("\nRecent Exchange:")
                for msg in self._data["recent_messages"][-5:]:
                    context_parts.append(f"{msg['role'].upper()}: {msg['content'][:150]}...")
        
        # Agent insights
        if self._data["agent_insights"]:
            context_parts.append("\n=== AGENT INSIGHTS ===")
            for agent, insights in self._data["agent_insights"].items():
                if insights:
                    context_parts.append(f"{agent}: {insights[-1][:100]}")
        
        return "\n".join(context_parts)
    
    def get_document_context_for_rag(self) -> str:
        """Get document-specific context for RAG queries."""
        if not self._data["uploaded_files"]:
            return "No documents uploaded yet."
        
        parts = [f"Available documents: {', '.join(self._data['uploaded_files'])}"]
        
        for filename, summary in self._data["document_summaries"].items():
            parts.append(f"\n{filename}:\n{summary}")
        
        if self._data["document_key_facts"]:
            parts.append(f"\nKey Facts:\n" + "\n".join([f"• {f}" for f in self._data["document_key_facts"][:15]]))
        
        return "\n".join(parts)
    
    def has_documents(self) -> bool:
        """Check if any documents are uploaded."""
        return len(self._data["uploaded_files"]) > 0
    
    def clear(self):
        """Clear all context for this session."""
        SharedContextManager._sessions[self.session_id] = {
            "document_summaries": {},
            "document_key_facts": [],
            "conversation_summary": "",
            "recent_messages": [],
            "uploaded_files": [],
            "extracted_topics": [],
            "agent_insights": {},
            "total_messages_processed": 0,
        }
        self._data = SharedContextManager._sessions[self.session_id]


def extract_document_context(file_path: str, session_id: str) -> dict:
    """
    Extract summary and key facts from a document for shared context.
    Called after document indexing.
    """
    try:
        # Load document
        documents = load_document(file_path)
        full_text = "\n".join([doc.page_content for doc in documents])[:8000]
        filename = Path(file_path).name
        
        # Use LLM to extract summary and key facts
        extraction_prompt = f"""Analyze this document and extract:

DOCUMENT: {filename}
CONTENT:
{full_text}

Provide your response in this exact JSON format:
{{
    "summary": "2-3 sentence summary of the document",
    "key_facts": ["fact 1", "fact 2", "fact 3", "fact 4", "fact 5"],
    "topics": ["topic1", "topic2", "topic3"],
    "document_type": "type of document (e.g., research paper, manual, report)"
}}"""

        response = llm.invoke([HumanMessage(content=extraction_prompt)])
        
        # Parse JSON response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {
                    "summary": response.content[:300],
                    "key_facts": [],
                    "topics": [],
                    "document_type": "unknown"
                }
        except json.JSONDecodeError:
            data = {
                "summary": response.content[:300],
                "key_facts": [],
                "topics": [],
                "document_type": "unknown"
            }
        
        # Update shared context
        ctx_manager = SharedContextManager(session_id)
        ctx_manager.add_document(
            filename=filename,
            summary=data.get("summary", ""),
            key_facts=data.get("key_facts", []),
            topics=data.get("topics", [])
        )
        
        return {
            "status": "success",
            "filename": filename,
            "summary": data.get("summary", ""),
            "facts_extracted": len(data.get("key_facts", [])),
            "topics": data.get("topics", [])
        }
        
    except Exception as e:
        print(f"⚠️ Context extraction failed: {e}")
        return {"status": "error", "message": str(e)}


# =============================================================================
# STATE DEFINITIONS - ENHANCED WITH SHARED CONTEXT
# =============================================================================

class SupervisorState(TypedDict):
    """
    Enhanced state for the supervisor multi-agent system.
    Supports shared context and multi-agent task delegation.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str  # Current agent to handle query
    pending_agents: List[str]  # Additional agents to run after current
    final_response: str  # The final response to return
    uploaded_file_path: Optional[str]  # Path to uploaded file (if any)
    session_id: str  # Session ID for checkpointing
    shared_context: str  # Unified context string for all agents
    agent_results: dict  # Results from each agent {agent_name: result}
    task_type: str  # Type of task: "single", "multi", "chain"


# =============================================================================
# DOCUMENT PROCESSING
# =============================================================================

SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".csv": CSVLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".docx": UnstructuredWordDocumentLoader,
}


def load_document(file_path: str) -> List[Document]:
    """Load a document based on its extension."""
    ext = Path(file_path).suffix.lower()
    
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(SUPPORTED_EXTENSIONS.keys())}")
    
    loader_class = SUPPORTED_EXTENSIONS[ext]
    loader = loader_class(file_path)
    return loader.load()


def index_document(file_path: str, session_id: str = "default") -> dict:
    """
    Index a document into the Qdrant vector store.
    
    Args:
        file_path: Path to the document file
        session_id: Session ID to associate with the document
        
    Returns:
        dict with status and chunk count
    """
    # Load document
    documents = load_document(file_path)
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    # Add metadata
    for chunk in chunks:
        chunk.metadata["session_id"] = session_id
        chunk.metadata["source_file"] = Path(file_path).name
        chunk.metadata["doc_id"] = str(uuid.uuid4())
    
    # Add to vector store
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    
    return {
        "status": "success",
        "chunks_indexed": len(chunks),
        "file_name": Path(file_path).name
    }


def query_documents(question: str, session_id: str = "default", k: int = 5) -> List[Document]:
    """
    Query the vector store for relevant documents.
    
    Args:
        question: The user's question
        session_id: Session ID to filter documents
        k: Number of results to return
        
    Returns:
        List of relevant documents
    """
    vector_store = get_vector_store()
    
    # Search with session filter using proper Qdrant filter format
    qdrant_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.session_id",
                match=MatchValue(value=session_id)
            )
        ]
    )
    
    results = vector_store.similarity_search(
        question,
        k=k,
        filter=qdrant_filter
    )
    
    return results


# =============================================================================
# TOOLS DEFINITION
# =============================================================================

@tool
def find_popular_leetcode_problems(topic: str) -> str:
    """
    Uses Tavily to find curated lists of popular LeetCode problems for a specific topic.
    Useful for gathering problem names like 'Two Sum', '3Sum', etc.
    """
    if not TAVILY_API_KEY:
        return "Tavily API key not configured."
    search = TavilySearchResults(max_results=5, search_depth="advanced", include_raw_content=True)
    query = f"Top 50 LeetCode interview questions for {topic} list"
    return search.invoke(query)


@tool
def get_dsa_explanation(concept: str) -> str:
    """
    Searches GeeksforGeeks and NeetCode for explanations of a DSA concept.
    """
    if not TAVILY_API_KEY:
        return "Tavily API key not configured."
    search = TavilySearchResults(max_results=3, search_depth="advanced")
    query = f"{concept} explanation site:geeksforgeeks.org OR site:neetcode.io"
    return search.invoke(query)


@tool
def tavily_search(query: str) -> str:
    """
    General web search tool using Tavily for any query.
    """
    if not TAVILY_API_KEY:
        return "Tavily API key not configured."
    search = TavilySearchResults(max_results=3)
    return search.invoke(query)


@tool
def search_uploaded_documents(question: str, session_id: str = "default") -> str:
    """
    Search through uploaded documents to find relevant information.
    Use this when the user asks questions about their uploaded PDFs or documents.
    
    Args:
        question: The question to search for
        session_id: The session ID (defaults to 'default')
    
    Returns:
        Relevant context from the documents
    """
    try:
        docs = query_documents(question, session_id, k=5)
        if not docs:
            return "No relevant documents found. Please upload a document first."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "Unknown")
            context_parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
        
        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        return f"Error searching documents: {str(e)}"


# Tools lists
research_tools = [find_popular_leetcode_problems, get_dsa_explanation, tavily_search]
rag_tools = [search_uploaded_documents]
all_tools = research_tools + rag_tools

research_tool_node = ToolNode(research_tools)
llm_with_tools = llm.bind_tools(research_tools)
llm_with_rag_tools = llm.bind_tools(rag_tools)


# =============================================================================
# AGENT DEFINITIONS
# =============================================================================

# Agent names
RESEARCH_AGENT = "research_agent"
EXAMINER_AGENT = "examiner_agent"
CHAT_AGENT = "chat_agent"
RAG_AGENT = "rag_agent"
LEARNING_ARCHITECT_AGENT = "learning_architect_agent"
AGENT_LIST = [RESEARCH_AGENT, EXAMINER_AGENT, CHAT_AGENT, RAG_AGENT, LEARNING_ARCHITECT_AGENT]

# =============================================================================
# LEARNING ARCHITECT SYSTEM PROMPT
# =============================================================================

LEARNING_ARCHITECT_SYSTEM_PROMPT = """You are the "Darksied Learning Architect." Your goal is to transform complex context into structured educational artifacts.

### RULES:
1. ALWAYS output in TWO formats: A structured JSON for Quiz Cards and a Mermaid.js code block for a Mindmap.
2. GROUNDING: Use ONLY the provided context from the RAG retrieval. If the context is missing, ask for a document upload.
3. QUALITY: Quizzes must focus on conceptual "Why" and "How," not just "What."

### OUTPUT STRUCTURE:

[PART 1: MINDMAP]
Generate a Mermaid.js mindmap syntax that visualizes the hierarchy of the topic.
Format (ALWAYS use (("double parentheses and quotes")) for node labels):
```mermaid
mindmap
  root(("Central Topic"))
    Subtopic 1(("Subtopic 1"))
      Detail A(("Detail A"))
      Detail B(("Detail B"))
    Subtopic 2(("Subtopic 2"))
      Detail C(("Detail C"))
```

[PART 2: QUIZ CARDS]
Generate a JSON array of 5 quiz card objects:
```json
{
  "cards": [
    {
      "question": "...",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "answer": "...",
      "explanation": "...",
      "tts_text": "Question: [question text]. Think carefully about the options."
    }
  ]
}
```

### GUIDELINES:
- Mindmap should capture the main concepts and their relationships
- Quiz questions should test understanding, not just memorization
- Include a mix of difficulty levels in the quiz
- Explanations should help reinforce learning
- Use clear, educational language"""


def supervisor_node(state: SupervisorState) -> dict:
    """
    Enhanced Supervisor Agent with multi-agent routing and shared context.
    
    Can route to:
    - Single agent for simple queries
    - Multiple agents for complex queries (e.g., RAG + Mindmap)
    - Chain of agents for sequential processing
    
    All agents receive the same shared context.
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    session_id = state.get("session_id", "default")
    has_uploaded_file = bool(state.get("uploaded_file_path"))
    
    # Load shared context manager
    ctx_manager = SharedContextManager(session_id)
    has_documents = ctx_manager.has_documents()
    
    # Add user message to context manager
    ctx_manager.add_message("user", last_message)
    
    # Get unified context for all agents
    shared_context = ctx_manager.get_unified_context()
    
    # Enhanced supervisor prompt with multi-agent support
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent Supervisor Agent that routes queries to specialized agents.
You can assign ONE or MULTIPLE agents based on query complexity.

=== SHARED CONTEXT (Available to ALL agents) ===
{shared_context}

=== AVAILABLE AGENTS ===
1. **research_agent** - Web searches, LeetCode, DSA, current events, news, technical research
2. **examiner_agent** - Quiz/MCQ generation on general topics
3. **chat_agent** - General conversation, coding help, explanations, math
4. **rag_agent** - Questions about uploaded documents, summarization, Q&A
5. **learning_architect_agent** - Mindmaps and quiz cards FROM documents

=== MULTI-AGENT ROUTING ===
For complex queries, you can assign MULTIPLE agents separated by commas.
Examples:
- "Create a mindmap from the PDF and quiz me on it" → "rag_agent,learning_architect_agent"
- "Summarize the document and search for related topics" → "rag_agent,research_agent"
- "Help me understand this PDF" → "rag_agent" (single agent sufficient)

=== CURRENT STATE ===
Documents uploaded: {has_documents}
Files: {uploaded_files}

=== INSTRUCTIONS ===
Analyze the query and respond with agent name(s) ONLY.
For multiple agents, separate with commas (no spaces).
Single: "chat_agent"
Multiple: "rag_agent,learning_architect_agent"
"""),
        ("human", "{query}")
    ])
    
    # Build context for prompt
    uploaded_files = ", ".join(ctx_manager.uploaded_files) if ctx_manager.uploaded_files else "None"
    
    chain = supervisor_prompt | llm
    response = chain.invoke({
        "query": last_message, 
        "shared_context": shared_context[:2000] if shared_context else "No prior context.",
        "has_documents": "Yes" if has_documents else "No",
        "uploaded_files": uploaded_files
    })
    
    # Parse response for single or multiple agents
    response_text = response.content
    if isinstance(response_text, list):
        response_text = str(response_text[0]) if response_text else "chat_agent"
    
    response_text = response_text.strip().lower().replace(" ", "")
    
    # Check for multi-agent routing
    if "," in response_text:
        agents = [a.strip() for a in response_text.split(",")]
        agents = [a for a in agents if a in AGENT_LIST]  # Validate
        
        if len(agents) >= 2:
            primary_agent = agents[0]
            pending_agents = agents[1:]
            print(f"🎯 Supervisor multi-route: {primary_agent} → then {pending_agents}")
            
            return {
                "next_agent": primary_agent,
                "pending_agents": pending_agents,
                "shared_context": shared_context,
                "task_type": "multi",
                "agent_results": {}
            }
    
    # Single agent routing
    selected_agent = response_text.split(",")[0].strip()
    if selected_agent not in AGENT_LIST:
        selected_agent = CHAT_AGENT
    
    print(f"🎯 Supervisor routed to: {selected_agent}")
    
    # Add insight to context
    ctx_manager.add_agent_insight("Supervisor", f"Routed '{last_message[:50]}...' to {selected_agent}")
    
    return {
        "next_agent": selected_agent,
        "pending_agents": [],
        "shared_context": shared_context,
        "task_type": "single",
        "agent_results": {}
    }


def research_agent_node(state: SupervisorState) -> dict:
    """
    Research Agent: Handles web searches and research tasks using Tavily tools.
    ENHANCED: Uses shared context for document-aware research.
    """
    messages = state["messages"]
    session_id = state.get("session_id", "default")
    shared_context = state.get("shared_context", "")
    
    # Get context manager
    ctx_manager = SharedContextManager(session_id)
    
    # Build context-aware system message
    doc_context = ""
    if ctx_manager.has_documents():
        doc_context = f"""

=== DOCUMENT CONTEXT ===
The user has uploaded these documents: {', '.join(ctx_manager.uploaded_files)}
Topics from documents: {', '.join(ctx_manager.topics[:5]) if ctx_manager.topics else 'Not extracted'}

When searching, consider:
- Information related to the uploaded documents
- Topics that expand on document content
- Research that could help understand the documents better"""

    system_msg = SystemMessage(content=f"""You are a Research Agent with access to web search tools.
Your job is to find accurate, up-to-date information for the user.
Use the available tools to search the web when needed.
Always cite your sources when providing information from searches.
{doc_context}

If the user's query relates to uploaded documents, search for complementary information.""")
    
    response = llm_with_tools.invoke([system_msg] + list(messages))
    
    if response.tool_calls:
        tool_messages = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            for t in research_tools:
                if t.name == tool_name:
                    result = t.invoke(tool_args)
                    tool_messages.append(f"[{tool_name}]: {result}")
                    break
        
        follow_up = llm.invoke([
            system_msg,
            *messages,
            AIMessage(content=f"Tool results:\n" + "\n".join(tool_messages)),
            HumanMessage(content="Based on these search results, provide a helpful response.")
        ])
        final_content = follow_up.content
    else:
        final_content = response.content
    
    if isinstance(final_content, list):
        final_content = final_content[0].get("text", str(final_content)) if final_content else ""
    
    # Share insights with context manager
    ctx_manager.add_agent_insight("Research", f"Searched: {messages[-1].content[:50] if messages else 'unknown'}...")
    ctx_manager.add_message("assistant", final_content[:300])
    
    # Store result for multi-agent workflows
    agent_results = state.get("agent_results", {})
    agent_results["research_agent"] = {"response": final_content}
    
    return {
        "messages": [AIMessage(content=final_content)],
        "final_response": final_content,
        "agent_results": agent_results
    }


def examiner_agent_node(state: SupervisorState) -> dict:
    """
    Examiner Agent: Generates MCQ questions based on the topic.
    ENHANCED: Can generate questions from uploaded document content.
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages else "general knowledge"
    session_id = state.get("session_id", "default")
    shared_context = state.get("shared_context", "")
    agent_results = state.get("agent_results", {})
    
    # Get context manager
    ctx_manager = SharedContextManager(session_id)
    
    # Build document context if available
    doc_context = ""
    if ctx_manager.has_documents():
        doc_context = f"""
=== DOCUMENT CONTEXT FOR QUIZ GENERATION ===
Uploaded files: {', '.join(ctx_manager.uploaded_files)}

Key facts to base questions on:
{chr(10).join(['• ' + f for f in ctx_manager.key_facts[:10]])}

Topics: {', '.join(ctx_manager.topics[:5]) if ctx_manager.topics else 'General'}

If the user wants questions from documents, use these facts to create relevant MCQs.
"""

    # Check if RAG agent provided context
    rag_context = ""
    if "rag_agent" in agent_results:
        rag_context = f"\n=== CONTEXT FROM RAG AGENT ===\n{agent_results['rag_agent'].get('response', '')[:1500]}"

    prompt = ChatPromptTemplate.from_template(
        """You are an Expert Examiner Agent.
        
Based on the user's request, generate Multiple Choice Questions (MCQs).
{doc_context}
{rag_context}

User Request: {query}

Rules:
1. Generate 5-10 relevant MCQ questions based on the topic or document content.
2. Each question should have 4 options (A, B, C, D).
3. Mark the correct answer clearly.
4. Vary the difficulty level.
5. Format nicely with proper spacing.
6. If document content is available, prioritize questions from that content.

Generate the questions now:"""
    )
    
    chain = prompt | llm
    response = chain.invoke({
        "query": last_message,
        "doc_context": doc_context,
        "rag_context": rag_context
    })
    
    content = response.content
    if isinstance(content, list):
        content = content[0].get("text", str(content)) if content else ""
    
    # Share insights
    ctx_manager.add_agent_insight("Examiner", f"Generated MCQs on: {last_message[:50]}...")
    ctx_manager.add_message("assistant", content[:300])
    
    # Store result
    agent_results["examiner_agent"] = {"response": content}
    
    return {
        "messages": [AIMessage(content=content)],
        "final_response": content,
        "agent_results": agent_results
    }


def chat_agent_node(state: SupervisorState) -> dict:
    """
    Chat Agent: Handles general conversation and simple queries.
    ENHANCED: Uses shared context for document-aware conversations.
    """
    messages = state["messages"]
    session_id = state.get("session_id", "default")
    shared_context = state.get("shared_context", "")
    
    # Get context manager for document awareness
    ctx_manager = SharedContextManager(session_id)
    
    # Build context-aware system message
    doc_awareness = ""
    if ctx_manager.has_documents():
        doc_awareness = f"""

=== DOCUMENT CONTEXT ===
You have access to these uploaded documents: {', '.join(ctx_manager.uploaded_files)}

Key facts from documents:
{chr(10).join(['• ' + f for f in ctx_manager.key_facts[:5]])}

If the user asks about these documents, you can reference this information or suggest they use the RAG agent for detailed queries."""

    system_msg = SystemMessage(content=f"""You are a friendly and helpful AI assistant named Darksied.
You can help with:
- General conversation and greetings
- Answering questions from your knowledge
- Explaining concepts
- Helping with coding problems
- Math and logic problems
- Creative writing
{doc_awareness}

=== CONVERSATION CONTEXT ===
{ctx_manager.conversation_summary if ctx_manager.conversation_summary else "This is a new conversation."}

Be conversational, helpful, and concise. Reference uploaded documents when relevant.""")
    
    response = llm.invoke([system_msg] + list(messages))
    
    content = response.content
    if isinstance(content, list):
        content = content[0].get("text", str(content)) if content else ""
    
    # Update context
    ctx_manager.add_message("assistant", content)
    
    return {
        "messages": [AIMessage(content=content)],
        "final_response": content
    }


# =============================================================================
# RAG EVALUATION HELPER (OPIK HALLUCINATION METRIC)
# =============================================================================

def evaluate_rag(query: str, context: str, response: str):
    """
    Evaluate RAG response for hallucination using Opik's Hallucination metric.
    Decorated with @track for Opik tracing visibility.
    
    Args:
        query: The user's original question
        context: The retrieved document context
        response: The generated RAG response
        
    Returns:
        Hallucination score (float) or None if evaluation unavailable
    """
    if Hallucination is None:
        print("⚠️ Opik Hallucination metric not available.")
        return None
    
    try:
        metric = Hallucination()
        # Opik Hallucination metric expects: input, context, output
        score_result = metric.score(
            input=query,
            context=[context],  # Context should be a list
            output=response
        )
        hallucination_score = score_result.value if hasattr(score_result, 'value') else score_result
        print(f"🧾 Opik Hallucination Score: {hallucination_score}")
        return hallucination_score
    except Exception as e:
        print(f"⚠️ Could not compute hallucination metric: {e}")
        return None

# Apply @track decorator if available
if track is not None and track != (lambda fn: fn):
    evaluate_rag = track(evaluate_rag)


def rag_agent_node(state: SupervisorState) -> dict:
    """
    RAG Agent: Handles questions about uploaded documents using retrieval-augmented generation.
    ENHANCED: Uses shared context and contributes insights for other agents.
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    session_id = state.get("session_id", "default")
    shared_context = state.get("shared_context", "")
    
    # Get shared context manager
    ctx_manager = SharedContextManager(session_id)
    
    # Search for relevant documents
    try:
        docs = query_documents(last_message, session_id, k=5)
        
        if not docs:
            # Check if we have shared context with document info
            if ctx_manager.has_documents():
                content = f"""I found document context in our shared memory:

{ctx_manager.get_document_context_for_rag()}

However, I couldn't find specific chunks matching your query. Try rephrasing or ask a more general question about the documents."""
            else:
                content = "I don't have any documents indexed for your session. Please upload a PDF or document first using the upload feature."
        else:
            # Build context from retrieved documents
            context_parts = []
            sources = set()
            for doc in docs:
                context_parts.append(doc.page_content)
                sources.add(doc.metadata.get("source_file", "Unknown"))
            
            context = "\n\n".join(context_parts)
            
            # Enhanced RAG prompt with shared context
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a Document Q&A Agent with access to shared knowledge.

=== SHARED CONTEXT (from other agents) ===
{shared_context}

=== RETRIEVED DOCUMENT CHUNKS ===
{context}

=== SOURCES ===
{sources}

=== INSTRUCTIONS ===
1. Answer based PRIMARILY on the retrieved document chunks
2. Use shared context for additional understanding
3. Always cite which document the information comes from
4. If information isn't in the documents, say so clearly
5. Provide comprehensive answers that could help other agents understand the content"""),
                ("human", "{question}")
            ])
            
            chain = rag_prompt | llm
            response = chain.invoke({
                "context": context,
                "shared_context": shared_context[:1500] if shared_context else "No prior shared context.",
                "sources": ", ".join(sources),
                "question": last_message
            })
            
            content = response.content
            if isinstance(content, list):
                content = content[0].get("text", str(content)) if content else ""
            
            # Share insights with other agents via context manager
            ctx_manager.add_agent_insight("RAG", f"Retrieved from {', '.join(sources)}: {content[:200]}...")
            ctx_manager.add_message("assistant", content)
            
            # === OPIK RAG EVALUATION ===
            try:
                rag_score = evaluate_rag(last_message, context, content)
                if rag_score is not None:
                    print(f"📊 RAG Quality - Hallucination Score: {rag_score}")
            except Exception as eval_err:
                print(f"⚠️ RAG evaluation skipped: {eval_err}")
                
    except Exception as e:
        content = f"Error querying documents: {str(e)}. Please make sure documents are uploaded and Qdrant is running."
    
    # Store result for multi-agent workflows
    agent_results = state.get("agent_results", {})
    agent_results["rag_agent"] = {"response": content, "sources": list(sources) if 'sources' in dir() else []}
    
    return {
        "messages": [AIMessage(content=content)],
        "final_response": content,
        "agent_results": agent_results
    }


def learning_architect_agent_node(state: SupervisorState) -> dict:
    """
    Learning Architect Agent: Generates educational content (Mindmaps + Quiz Cards).
    ENHANCED: Uses shared context from RAG agent and other sources.
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    session_id = state.get("session_id", "default")
    shared_context = state.get("shared_context", "")
    agent_results = state.get("agent_results", {})
    
    # Get shared context manager
    ctx_manager = SharedContextManager(session_id)
    
    try:
        # First, check if RAG agent already retrieved context (multi-agent workflow)
        rag_context = ""
        if "rag_agent" in agent_results:
            rag_result = agent_results["rag_agent"]
            rag_context = rag_result.get("response", "")
            print(f"📚 Learning Architect using RAG context: {len(rag_context)} chars")
        
        # Retrieve relevant documents for context
        docs = query_documents(last_message, session_id, k=8)
        
        # Build context from multiple sources
        context_parts = []
        sources = set()
        
        # Add RAG agent insights if available
        if rag_context:
            context_parts.append(f"=== FROM RAG AGENT ===\n{rag_context[:2000]}")
        
        # Add shared context summaries
        if ctx_manager.has_documents():
            doc_context = ctx_manager.get_document_context_for_rag()
            context_parts.append(f"=== DOCUMENT SUMMARIES ===\n{doc_context[:1500]}")
            sources.update(ctx_manager.uploaded_files)
        
        # Add retrieved chunks
        if docs:
            chunk_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
            context_parts.append(f"=== RETRIEVED DOCUMENT CHUNKS ===\n{chunk_text}")
            for doc in docs:
                sources.add(doc.metadata.get("source_file", "Unknown"))
        
        if not context_parts:
            content = """📚 **Learning Architect Ready!**

I need document context to generate educational materials. Please:
1. Upload a document first using `upload <path>`
2. Then ask me to create a mindmap or quiz from it

Example commands:
- "Create a mindmap from the uploaded document"
- "Generate quiz cards from the PDF"
- "Help me learn the concepts in this document"
"""
            return {
                "messages": [AIMessage(content=content)],
                "final_response": content
            }
        
        context_text = "\n\n".join(context_parts)
        
        # Enhanced generation prompt with all context
        generation_prompt = f"""You are the Darksied Learning Architect. Transform the provided context into educational materials.

=== SHARED CONTEXT FROM ALL AGENTS ===
{shared_context[:1000] if shared_context else 'No shared context.'}

=== COMBINED DOCUMENT CONTEXT ===
{context_text[:6000]}

=== USER REQUEST ===
{last_message}

=== YOUR TASK ===
Generate BOTH a Mermaid.js mindmap AND a JSON quiz based on ALL the context above.
Use insights from the RAG agent and document summaries to create comprehensive materials.

{LEARNING_ARCHITECT_SYSTEM_PROMPT}

Now generate the educational content:"""

        # Try to use HuggingFace model, fall back to Gemini if unavailable
        learning_llm = get_or_init_learning_llm()
        
        if learning_llm is not None:
            try:
                response = learning_llm.invoke(generation_prompt)
                content = response if isinstance(response, str) else str(response)
            except Exception as hf_error:
                print(f"⚠️ HuggingFace model error: {hf_error}, falling back to Gemini")
                content = _generate_with_gemini_fallback(context_text, last_message, sources)
        else:
            content = _generate_with_gemini_fallback(context_text, last_message, sources)
        
        # Add source attribution
        source_list = ", ".join(sources) if sources else "shared context"
        content = f"📚 **Learning Materials Generated from: {source_list}**\n\n{content}"
        
        # Share insights with context manager
        ctx_manager.add_agent_insight("LearningArchitect", f"Generated mindmap/quiz from {len(sources)} sources")
        ctx_manager.add_message("assistant", content[:500])
        
    except Exception as e:
        content = f"""❌ Error generating learning materials: {str(e)}

**Troubleshooting:**
1. Make sure you've uploaded a document first
2. Ensure Qdrant is running
3. Check that the HuggingFace model is properly configured

You can set `HF_TOKEN` environment variable if using gated models."""

    # Store result for multi-agent workflows
    agent_results["learning_architect_agent"] = {"response": content, "sources": list(sources) if sources else []}

    return {
        "messages": [AIMessage(content=content)],
        "final_response": content,
        "agent_results": agent_results
    }


def _generate_with_gemini_fallback(context_text: str, user_request: str, sources: set) -> str:
    """
    Fallback function to generate educational content using Gemini if HuggingFace is unavailable.
    """
    fallback_prompt = ChatPromptTemplate.from_messages([
        ("system", LEARNING_ARCHITECT_SYSTEM_PROMPT),
        ("human", """Using the following context from uploaded documents, generate educational materials.

CONTEXT:
{context}

SOURCES: {sources}

USER REQUEST: {request}

Generate a comprehensive Mermaid mindmap first, then a 5-card JSON quiz.""")
    ])
    
    chain = fallback_prompt | llm
    response = chain.invoke({
        "context": context_text,
        "sources": ", ".join(sources),
        "request": user_request
    })
    
    content = response.content
    if isinstance(content, list):
        content = content[0].get("text", str(content)) if content else ""
    
    return content


def sanitize_mermaid_mindmap(mermaid_code: str) -> str:
    """
    Sanitize Mermaid mindmap code to fix parsing errors.
    1. Replaces nested parentheses in node labels with brackets.
    2. Ensures labels are quoted if they contain special characters.
    3. Handles flowchart/graph diagrams if mistakenly returned as mindmap.
    """
    if not mermaid_code:
        return mermaid_code
    
    # Trim and find the first line
    lines = mermaid_code.strip().split('\n')
    if not lines:
        return mermaid_code
        
    diagram_type = "mindmap"
    if "graph" in lines[0] or "flowchart" in lines[0]:
        diagram_type = "flowchart"
    
    sanitized_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
            
        # Keep diagram declaration
        if stripped in ['mindmap', 'graph TD', 'graph LR', 'flowchart TD', 'flowchart LR'] or stripped.startswith(('graph ', 'flowchart ')):
            sanitized_lines.append(line)
            continue

        if diagram_type == "mindmap":
            # Extract indentation
            indent_match = re.search(r'^(\s*)', line)
            indent = indent_match.group(1) if indent_match else ""
            
            # Check if it has a label part with parentheses
            if '(' in stripped:
                name_part = stripped[:stripped.find('(')].strip()
                label_part = stripped[stripped.find('('):].strip()
                
                # Sanitize the label part
                # Remove outermost parens (could be ( ), (( )), etc.)
                inner_label = label_part
                p_count = 0
                while inner_label.startswith('(') and inner_label.endswith(')'):
                    inner_label = inner_label[1:-1]
                    p_count += 1
                
                # Replace any remaining parens inside with brackets
                inner_label = inner_label.replace('(', '[').replace(')', ']')
                # Remove double quotes if already present to avoid triple quotes
                inner_label = inner_label.replace('"', "'")
                
                # Reconstruct with quotes and double parens (safest for mindmap)
                new_line = f'{indent}{name_part if name_part else ""} (("{inner_label}"))'
                sanitized_lines.append(new_line)
            else:
                # No parens, just a label. Wrap in quotes and double parens.
                # Extract indentation again to be safe
                indent_match = re.search(r'^(\s*)', line)
                indent = indent_match.group(1) if indent_match else ""
                label = stripped.replace('"', "'")
                sanitized_lines.append(f'{indent}(("{label}"))')
        else:
            # Flowchart/Graph sanitization - escape parentheses inside node labels
            # Patterns: A[label], A(label), A{label}, A((label)), A>label], etc.
            # We need to quote labels that contain parentheses
            
            def escape_flowchart_label(match):
                """Escape parentheses inside flowchart node labels by quoting them."""
                prefix = match.group(1)  # e.g., "A[" or "B("
                content = match.group(2)  # label content
                suffix = match.group(3)   # e.g., "]" or ")"
                
                # If content has unquoted parentheses, wrap in quotes
                if '(' in content or ')' in content:
                    # Remove existing quotes if any to avoid double-quoting
                    content = content.strip('"').strip("'")
                    return f'{prefix}"{content}"{suffix}'
                return match.group(0)
            
            # Match node definitions with various shapes
            # [text], (text), {text}, ((text)), >text], etc.
            sanitized_line = re.sub(
                r'(\w+\[)([^\]]+)(\])',  # Square brackets [text]
                escape_flowchart_label,
                line
            )
            sanitized_line = re.sub(
                r'(\w+\()([^)]+)(\)(?!\)))',  # Single parentheses (text) not ((
                escape_flowchart_label,
                sanitized_line
            )
            sanitized_line = re.sub(
                r'(\w+\{\{?)([^}]+)(\}?\})',  # Curly braces {text} or {{text}}
                escape_flowchart_label,
                sanitized_line
            )
            
            sanitized_lines.append(sanitized_line)
            
    return '\n'.join(sanitized_lines)


def parse_learning_output(raw_output: str) -> dict:
    """
    Parse the Learning Architect output to extract mindmap and quiz separately.
    
    Returns:
        dict with 'mindmap' (str) and 'quiz' (dict) keys
    """
    result = {
        "mindmap": None,
        "quiz": None,
        "raw": raw_output
    }
    
    # Extract Mermaid mindmap
    mermaid_pattern = r'```mermaid\s*([\s\S]*?)```'
    mermaid_match = re.search(mermaid_pattern, raw_output)
    if mermaid_match:
        mindmap_code = mermaid_match.group(1).strip()
        # Sanitize the mindmap code to fix parsing errors
        result["mindmap"] = sanitize_mermaid_mindmap(mindmap_code)
    
    # Extract JSON quiz
    json_pattern = r'```json\s*([\s\S]*?)```'
    json_match = re.search(json_pattern, raw_output)
    if json_match:
        try:
            result["quiz"] = json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            # Try to find just the cards array
            cards_pattern = r'\{\s*"cards"\s*:\s*\[([\s\S]*?)\]\s*\}'
            cards_match = re.search(cards_pattern, raw_output)
            if cards_match:
                try:
                    result["quiz"] = json.loads(f'{{"cards": [{cards_match.group(1)}]}}')
                except json.JSONDecodeError:
                    pass
    
    return result


# =============================================================================
# ROUTING FUNCTION
# =============================================================================

def route_to_agent(state: SupervisorState) -> Literal["research_agent", "examiner_agent", "chat_agent", "rag_agent", "learning_architect_agent"]:
    """Routes to the appropriate agent based on supervisor's decision."""
    return state["next_agent"]


# =============================================================================
# BUILD THE GRAPH
# =============================================================================

def multi_agent_router(state: SupervisorState) -> str:
    """
    Routes to the next pending agent or ends the workflow.
    Enables multi-agent task delegation.
    """
    pending = state.get("pending_agents", [])
    
    if pending and len(pending) > 0:
        next_agent = pending[0]
        print(f"🔄 Multi-agent chain: routing to {next_agent}")
        return next_agent
    
    return END


def process_pending_agents(state: SupervisorState) -> dict:
    """Process and update pending agents list."""
    pending = state.get("pending_agents", [])
    
    if pending and len(pending) > 0:
        # Remove the first pending agent (it will be executed next)
        new_pending = pending[1:] if len(pending) > 1 else []
        next_agent = pending[0]
        
        return {
            "next_agent": next_agent,
            "pending_agents": new_pending,
        }
    
    return {"pending_agents": []}


def create_supervisor_graph(checkpointer=None):
    """
    Creates and compiles the supervisor multi-agent graph.
    ENHANCED: Supports multi-agent task delegation and shared context.
    """
    
    workflow = StateGraph(SupervisorState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node(RESEARCH_AGENT, research_agent_node)
    workflow.add_node(EXAMINER_AGENT, examiner_agent_node)
    workflow.add_node(CHAT_AGENT, chat_agent_node)
    workflow.add_node(RAG_AGENT, rag_agent_node)
    workflow.add_node(LEARNING_ARCHITECT_AGENT, learning_architect_agent_node)
    workflow.add_node("multi_agent_handler", process_pending_agents)
    
    # Add edges
    workflow.add_edge(START, "supervisor")
    
    # Conditional routing from supervisor to agents
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            RESEARCH_AGENT: RESEARCH_AGENT,
            EXAMINER_AGENT: EXAMINER_AGENT,
            CHAT_AGENT: CHAT_AGENT,
            RAG_AGENT: RAG_AGENT,
            LEARNING_ARCHITECT_AGENT: LEARNING_ARCHITECT_AGENT,
        }
    )
    
    # Multi-agent chain: after each agent, check for pending agents
    for agent in AGENT_LIST:
        workflow.add_conditional_edges(
            agent,
            multi_agent_router,
            {
                RESEARCH_AGENT: RESEARCH_AGENT,
                EXAMINER_AGENT: EXAMINER_AGENT,
                CHAT_AGENT: CHAT_AGENT,
                RAG_AGENT: RAG_AGENT,
                LEARNING_ARCHITECT_AGENT: LEARNING_ARCHITECT_AGENT,
                END: END,
            }
        )
    
    # Compile with checkpointer if provided
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class SupervisorChatbot:
    """Main chatbot class with supervisor agent, RAG, and memory."""
    
    def __init__(self, use_checkpointer: bool = True):
        """
        Initialize the chatbot.
        
        Args:
            use_checkpointer: Whether to use PostgreSQL for conversation memory
        """
        self.session_id = str(uuid.uuid4())
        self.checkpointer = None
        
        # Try to initialize infrastructure
        try:
            initialize_qdrant_collection()
            print("✅ Qdrant vector store connected!")
        except Exception as e:
            print(f"⚠️ Qdrant not available: {e}")
            print("   RAG features will be limited.")
        
        if use_checkpointer:
            try:
                self.checkpointer = get_checkpointer()
                print("✅ PostgreSQL checkpointer connected!")
            except Exception as e:
                print(f"⚠️ PostgreSQL not available: {e}")
                print("   Conversation memory will not persist.")
                self.checkpointer = None
        
        self.graph = create_supervisor_graph(self.checkpointer)
        print("✅ Supervisor Multi-Agent System initialized!")
        print(f"   Available agents: {', '.join(AGENT_LIST)}")
        print(f"   Session ID: {self.session_id}")
    
    def upload_document(self, file_path: str) -> dict:
        """
        Upload and index a document for RAG.
        ENHANCED: Also extracts context for shared memory across all agents.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            dict with status, chunk count, and context extraction results
        """
        try:
            # Step 1: Index for vector search (RAG)
            result = index_document(file_path, self.session_id)
            print(f"📄 Indexed document: {result['file_name']} ({result['chunks_indexed']} chunks)")
            
            # Step 2: Extract context for shared memory (ALL agents can use this)
            try:
                context_result = extract_document_context(file_path, self.session_id)
                if context_result.get("status") == "success":
                    print(f"🧠 Context extracted: {context_result.get('facts_extracted', 0)} facts, topics: {context_result.get('topics', [])}")
                    result["context_extracted"] = True
                    result["summary"] = context_result.get("summary", "")
                    result["topics"] = context_result.get("topics", [])
                    
                    # Notify about shared context availability
                    print(f"✅ Document context now available to ALL agents (Voice, Mindmap, RAG, Chat)")
            except Exception as ctx_err:
                print(f"⚠️ Context extraction partial: {ctx_err}")
                result["context_extracted"] = False
            
            return result
        except Exception as e:
            error_msg = f"Error indexing document: {str(e)}"
            print(f"❌ {error_msg}")
            return {"status": "error", "message": error_msg}
    
    def get_shared_context(self) -> str:
        """Get the current shared context for this session."""
        ctx_manager = SharedContextManager(self.session_id)
        return ctx_manager.get_unified_context()
    
    def clear_context(self):
        """Clear all shared context for this session."""
        ctx_manager = SharedContextManager(self.session_id)
        ctx_manager.clear()
        print("🧹 Shared context cleared.")
    
    def chat(self, user_input: str, uploaded_file_path: str = None) -> str:
        """
        Process a user message and return the response.
        ENHANCED: Uses shared context and supports multi-agent workflows.
        
        Args:
            user_input: The user's message
            uploaded_file_path: Optional path to an uploaded file
            
        Returns:
            The assistant's response
        """
        # If a file is uploaded, index it first
        if uploaded_file_path:
            self.upload_document(uploaded_file_path)
        
        # Get current shared context
        ctx_manager = SharedContextManager(self.session_id)
        shared_context = ctx_manager.get_unified_context()
        
        # Build config with thread_id for checkpointing
        config = {"configurable": {"thread_id": self.session_id}}
        
        # === CRITICAL: Attach Opik tracer for full LangGraph journey tracing ===
        if opik_tracer is not None:
            config["callbacks"] = config.get("callbacks", []) + [opik_tracer]
        
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "next_agent": "",
                "pending_agents": [],
                "final_response": "",
                "uploaded_file_path": uploaded_file_path,
                "session_id": self.session_id,
                "shared_context": shared_context,
                "agent_results": {},
                "task_type": "single",
            },
            config=config
        )
        return result.get("final_response", "I couldn't process that request.")
    
    def run_interactive(self):
        """Run an interactive chat session."""
        print("\n" + "="*60)
        print("🤖 SUPERVISOR MULTI-AGENT CHATBOT WITH RAG")
        print("="*60)
        print("Commands:")
        print("  'quit' or 'exit' - End the session")
        print("  'upload <path>' - Upload a document for RAG")
        print("  'help' - Show help message")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == "help":
                    self._print_help()
                    continue
                
                # Handle file upload command
                if user_input.lower().startswith("upload "):
                    file_path = user_input[7:].strip()
                    if os.path.exists(file_path):
                        result = self.upload_document(file_path)
                        if result["status"] == "success":
                            print(f"✅ Document uploaded and indexed: {result['chunks_indexed']} chunks\n")
                        else:
                            print(f"❌ {result.get('message', 'Upload failed')}\n")
                    else:
                        print(f"❌ File not found: {file_path}\n")
                    continue
                
                print("\n🔄 Processing...\n")
                response = self.chat(user_input)
                print(f"🤖 Assistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")
    
    def _print_help(self):
        """Print help information."""
        print("""
📚 HELP - Available Commands:
─────────────────────────────────────────────────────
• Type any message to chat
• 'upload <path>' - Upload a PDF/document for Q&A
• 'quit' or 'exit' - End the session
• 'help' - Show this message

🎯 Agent Routing Examples:
─────────────────────────────────────────────────────
• "What's the weather in NYC?" → Research Agent
• "Find LeetCode problems on arrays" → Research Agent  
• "Generate quiz questions on Python" → Examiner Agent
• "Create MCQs about machine learning" → Examiner Agent
• "Hi, how are you?" → Chat Agent
• "Explain recursion to me" → Chat Agent

📄 RAG (Document Q&A) Examples:
─────────────────────────────────────────────────────
1. First upload: upload /path/to/document.pdf
2. Then ask: "What is the main topic of the document?"
3. Or: "Summarize chapter 2 from the PDF"
4. Or: "What does the paper say about neural networks?"

🧠 Learning Architect (Mindmap + Quiz) Examples:
─────────────────────────────────────────────────────
1. First upload: upload /path/to/document.pdf
2. Then ask:
   • "Create a mindmap from this document"
   • "Generate quiz cards from the uploaded PDF"
   • "Help me learn the concepts in this file"
   • "Make study materials from the document"
   • "Visualize the key topics as a mindmap"

The Learning Architect generates:
  📊 Mermaid.js Mindmaps - Visual concept hierarchies
  📝 Quiz Cards (JSON) - 5 MCQ questions with explanations

Supported file types: PDF, TXT, MD, CSV, DOC, DOCX
""")
    
    def generate_learning_materials(self, topic_query: str = None) -> dict:
        """
        Programmatically generate learning materials from uploaded documents.
        
        Args:
            topic_query: Optional specific topic to focus on
            
        Returns:
            dict with 'mindmap', 'quiz', and 'raw' keys
        """
        if topic_query is None:
            topic_query = "Generate a comprehensive mindmap and quiz from the uploaded documents"
        
        # Force routing to learning architect
        response = self.chat(f"Create mindmap and quiz cards: {topic_query}")
        
        # Parse the output
        return parse_learning_output(response)
    
    def get_quiz_only(self, topic: str = None) -> dict:
        """
        Generate only quiz cards from uploaded documents.
        
        Args:
            topic: Optional specific topic to focus on
            
        Returns:
            dict with quiz cards or None if parsing failed
        """
        query = f"Generate quiz cards about {topic}" if topic else "Generate quiz cards from the document"
        response = self.chat(query)
        result = parse_learning_output(response)
        return result.get("quiz")
    
    def get_mindmap_only(self, topic: str = None) -> str:
        """
        Generate only a mindmap from uploaded documents.
        
        Args:
            topic: Optional specific topic to focus on
            
        Returns:
            Mermaid.js mindmap string or None if parsing failed
        """
        query = f"Create a mindmap about {topic}" if topic else "Create a mindmap from the document"
        response = self.chat(query)
        result = parse_learning_output(response)
        return result.get("mindmap")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    use_docker = "--docker" in sys.argv
    
    if use_docker:
        # Running in Docker - wait for services to be ready
        import time
        print("🐳 Running in Docker mode...")
        print("⏳ Waiting for services to be ready...")
        time.sleep(5)  # Give services time to start
    
    # Create and run the chatbot
    chatbot = SupervisorChatbot(use_checkpointer=True)
    
    # Run some test queries (skip if in Docker mode)
    if not use_docker:
        print("\n" + "="*60)
        print("📋 RUNNING TEST QUERIES")
        print("="*60 + "\n")
        
        test_queries = [
            "Hi! What can you do?",
            "Generate 3 MCQ questions about Python basics",
        ]
        
        for query in test_queries:
            print(f"📝 Query: {query}")
            print("-" * 40)
            response = chatbot.chat(query)
            print(f"💬 Response: {response[:500]}..." if len(response) > 500 else f"💬 Response: {response}")
            print("\n")
        
        print("\n" + "="*60)
        print("🧠 LEARNING ARCHITECT READY")
        print("="*60)
        print("Upload a document and try commands like:")
        print("  • 'Create a mindmap from the document'")
        print("  • 'Generate quiz cards from the PDF'")
        print("  • 'Help me learn the concepts in this file'")
        print("="*60 + "\n")
    
    # Start interactive mode
    print("\n" + "="*60)
    print("Starting Interactive Mode...")
    print("="*60)
    chatbot.run_interactive()
