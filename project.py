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
# STATE DEFINITIONS
# =============================================================================

class SupervisorState(TypedDict):
    """State for the supervisor multi-agent system."""
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str  # Which agent should handle this query
    final_response: str  # The final response to return
    uploaded_file_path: Optional[str]  # Path to uploaded file (if any)
    session_id: str  # Session ID for checkpointing


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
Format:
```mermaid
mindmap
  root((Central Topic))
    Subtopic 1
      Detail A
      Detail B
    Subtopic 2
      Detail C
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
    Supervisor Agent: Analyzes the user query and decides which agent should handle it.
    
    Routes to:
    - research_agent: For web searches, LeetCode problems, DSA explanations
    - examiner_agent: For generating quiz/MCQ questions (without document context)
    - chat_agent: For general conversation, greetings, simple questions
    - rag_agent: For questions about uploaded documents/PDFs
    - learning_architect_agent: For generating mindmaps and quizzes FROM uploaded documents
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    has_uploaded_file = bool(state.get("uploaded_file_path"))
    
    # Create supervisor prompt
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Supervisor Agent that routes user queries to specialized agents.

Available Agents:
1. **research_agent** - Use for: web searches, finding LeetCode problems, DSA explanations, 
   current events, weather, news, technical research, anything requiring internet search.
   
2. **examiner_agent** - Use for: generating quiz questions, MCQs, test questions, 
   practice problems on general topics (NOT from documents).
   
3. **chat_agent** - Use for: general conversation, greetings, simple factual questions 
   that don't need search, explanations from your knowledge, coding help, math.

4. **rag_agent** - Use for: general questions about uploaded documents, PDFs, files.
   Keywords: "document", "pdf", "file", "uploaded", "the paper", "summarize".
   {file_context}

5. **learning_architect_agent** - Use for: generating EDUCATIONAL content FROM documents:
   - Mindmaps from document content
   - Quiz cards from document content  
   - Study materials from uploaded files
   - Learning aids based on documents
   Keywords: "mindmap", "mind map", "quiz from document", "study material", 
   "learning cards", "flashcards from pdf", "visualize the document", "help me learn",
   "create quiz from", "generate mindmap from", "study guide".
   {file_context}

Analyze the user's message and respond with ONLY the agent name (one of: research_agent, examiner_agent, chat_agent, rag_agent, learning_architect_agent).
Do NOT include any other text, just the agent name."""),
        ("human", "{query}")
    ])
    
    file_context = "Note: User HAS uploaded a file." if has_uploaded_file else "Note: No file uploaded yet."
    
    chain = supervisor_prompt | llm
    response = chain.invoke({"query": last_message, "file_context": file_context})
    
    # Extract the agent name from response
    response_text = response.content
    if isinstance(response_text, list):
        response_text = str(response_text[0]) if response_text else "chat_agent"
    
    # Clean and validate the response
    selected_agent = response_text.strip().lower().replace(" ", "_")
    
    # Default to chat_agent if invalid
    if selected_agent not in AGENT_LIST:
        selected_agent = CHAT_AGENT
    
    print(f"🎯 Supervisor routed to: {selected_agent}")
    
    return {"next_agent": selected_agent}


def research_agent_node(state: SupervisorState) -> dict:
    """
    Research Agent: Handles web searches and research tasks using Tavily tools.
    """
    messages = state["messages"]
    
    system_msg = SystemMessage(content="""You are a Research Agent with access to web search tools.
Your job is to find accurate, up-to-date information for the user.
Use the available tools to search the web when needed.
Always cite your sources when providing information from searches.""")
    
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
    
    return {
        "messages": [AIMessage(content=final_content)],
        "final_response": final_content
    }


def examiner_agent_node(state: SupervisorState) -> dict:
    """
    Examiner Agent: Generates MCQ questions based on the topic.
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages else "general knowledge"
    
    prompt = ChatPromptTemplate.from_template(
        """You are an Expert Examiner Agent.
        
Based on the user's request, generate Multiple Choice Questions (MCQs).

User Request: {query}

Rules:
1. Generate 5-10 relevant MCQ questions based on the topic mentioned.
2. Each question should have 4 options (A, B, C, D).
3. Mark the correct answer clearly.
4. Vary the difficulty level.
5. Format nicely with proper spacing.

Generate the questions now:"""
    )
    
    chain = prompt | llm
    response = chain.invoke({"query": last_message})
    
    content = response.content
    if isinstance(content, list):
        content = content[0].get("text", str(content)) if content else ""
    
    return {
        "messages": [AIMessage(content=content)],
        "final_response": content
    }


def chat_agent_node(state: SupervisorState) -> dict:
    """
    Chat Agent: Handles general conversation and simple queries.
    """
    messages = state["messages"]
    
    system_msg = SystemMessage(content="""You are a friendly and helpful AI assistant.
You can help with:
- General conversation and greetings
- Answering questions from your knowledge
- Explaining concepts
- Helping with coding problems
- Math and logic problems
- Creative writing

Be conversational, helpful, and concise.""")
    
    response = llm.invoke([system_msg] + list(messages))
    
    content = response.content
    if isinstance(content, list):
        content = content[0].get("text", str(content)) if content else ""
    
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
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    session_id = state.get("session_id", "default")
    
    # Search for relevant documents
    try:
        docs = query_documents(last_message, session_id, k=5)
        
        if not docs:
            content = "I don't have any documents indexed for your session. Please upload a PDF or document first using the upload feature."
        else:
            # Build context from retrieved documents
            context_parts = []
            sources = set()
            for doc in docs:
                context_parts.append(doc.page_content)
                sources.add(doc.metadata.get("source_file", "Unknown"))
            
            context = "\n\n".join(context_parts)
            
            # Create RAG prompt
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a Document Q&A Agent. Answer questions based ONLY on the provided context.
If the answer is not in the context, say "I couldn't find this information in the uploaded documents."
Always cite which document the information comes from.

Context from uploaded documents:
{context}

Sources: {sources}"""),
                ("human", "{question}")
            ])
            
            chain = rag_prompt | llm
            response = chain.invoke({
                "context": context,
                "sources": ", ".join(sources),
                "question": last_message
            })
            
            content = response.content
            if isinstance(content, list):
                content = content[0].get("text", str(content)) if content else ""
            
            # === OPIK RAG EVALUATION ===
            # Evaluate hallucination score for the RAG response (non-blocking)
            try:
                rag_score = evaluate_rag(last_message, context, content)
                if rag_score is not None:
                    print(f"📊 RAG Quality - Hallucination Score: {rag_score}")
            except Exception as eval_err:
                print(f"⚠️ RAG evaluation skipped: {eval_err}")
                
    except Exception as e:
        content = f"Error querying documents: {str(e)}. Please make sure documents are uploaded and Qdrant is running."
    
    return {
        "messages": [AIMessage(content=content)],
        "final_response": content
    }


def learning_architect_agent_node(state: SupervisorState) -> dict:
    """
    Learning Architect Agent: Generates educational content (Mindmaps + Quiz Cards) 
    from uploaded documents using HuggingFace Qwen model.
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    session_id = state.get("session_id", "default")
    
    try:
        # Retrieve relevant documents for context
        docs = query_documents(last_message, session_id, k=8)  # Get more context for learning
        
        if not docs:
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
        
        # Build rich context from retrieved documents
        context_parts = []
        sources = set()
        for doc in docs:
            context_parts.append(doc.page_content)
            sources.add(doc.metadata.get("source_file", "Unknown"))
        
        context_text = "\n\n---\n\n".join(context_parts)
        
        # Create the educational content generation prompt
        generation_prompt = f"""You are the Darksied Learning Architect. Transform the following document context into educational materials.

### CONTEXT FROM DOCUMENTS:
{context_text}

### USER REQUEST:
{last_message}

### YOUR TASK:
Generate BOTH a Mermaid.js mindmap AND a JSON quiz based on the context above.

{LEARNING_ARCHITECT_SYSTEM_PROMPT}

Now generate the educational content:"""

        # Try to use HuggingFace model, fall back to Gemini if unavailable
        learning_llm = get_or_init_learning_llm()
        
        if learning_llm is not None:
            # Use HuggingFace Qwen model
            try:
                response = learning_llm.invoke(generation_prompt)
                content = response if isinstance(response, str) else str(response)
            except Exception as hf_error:
                print(f"⚠️ HuggingFace model error: {hf_error}, falling back to Gemini")
                content = _generate_with_gemini_fallback(context_text, last_message, sources)
        else:
            # Fallback to Gemini
            content = _generate_with_gemini_fallback(context_text, last_message, sources)
        
        # Add source attribution
        source_list = ", ".join(sources)
        content = f"📚 **Learning Materials Generated from: {source_list}**\n\n{content}"
        
    except Exception as e:
        content = f"""❌ Error generating learning materials: {str(e)}

**Troubleshooting:**
1. Make sure you've uploaded a document first
2. Ensure Qdrant is running
3. Check that the HuggingFace model is properly configured

You can set `HF_TOKEN` environment variable if using gated models."""

    return {
        "messages": [AIMessage(content=content)],
        "final_response": content
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
        result["mindmap"] = mermaid_match.group(1).strip()
    
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

def create_supervisor_graph(checkpointer=None):
    """Creates and compiles the supervisor multi-agent graph."""
    
    workflow = StateGraph(SupervisorState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node(RESEARCH_AGENT, research_agent_node)
    workflow.add_node(EXAMINER_AGENT, examiner_agent_node)
    workflow.add_node(CHAT_AGENT, chat_agent_node)
    workflow.add_node(RAG_AGENT, rag_agent_node)
    workflow.add_node(LEARNING_ARCHITECT_AGENT, learning_architect_agent_node)
    
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
    
    # All agents end after processing
    workflow.add_edge(RESEARCH_AGENT, END)
    workflow.add_edge(EXAMINER_AGENT, END)
    workflow.add_edge(CHAT_AGENT, END)
    workflow.add_edge(RAG_AGENT, END)
    workflow.add_edge(LEARNING_ARCHITECT_AGENT, END)
    
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
        
        Args:
            file_path: Path to the document file
            
        Returns:
            dict with status and chunk count
        """
        try:
            result = index_document(file_path, self.session_id)
            print(f"📄 Indexed document: {result['file_name']} ({result['chunks_indexed']} chunks)")
            return result
        except Exception as e:
            error_msg = f"Error indexing document: {str(e)}"
            print(f"❌ {error_msg}")
            return {"status": "error", "message": error_msg}
    
    def chat(self, user_input: str, uploaded_file_path: str = None) -> str:
        """
        Process a user message and return the response.
        
        Args:
            user_input: The user's message
            uploaded_file_path: Optional path to an uploaded file
            
        Returns:
            The assistant's response
        """
        # If a file is uploaded, index it first
        if uploaded_file_path:
            self.upload_document(uploaded_file_path)
        
        # Build config with thread_id for checkpointing
        config = {"configurable": {"thread_id": self.session_id}}
        
        # === CRITICAL: Attach Opik tracer for full LangGraph journey tracing ===
        if opik_tracer is not None:
            config["callbacks"] = config.get("callbacks", []) + [opik_tracer]
        
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "next_agent": "",
                "final_response": "",
                "uploaded_file_path": uploaded_file_path,
                "session_id": self.session_id,
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
