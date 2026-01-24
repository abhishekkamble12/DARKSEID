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
import uuid
import tempfile
from typing import TypedDict, Annotated, Literal, List, Optional
from dotenv import load_dotenv
from pathlib import Path

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

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
# CONFIGURATION
# =============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/agent_db")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")

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
    
    # Search with session filter
    results = vector_store.similarity_search(
        question,
        k=k,
        filter={"session_id": session_id}
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
AGENT_LIST = [RESEARCH_AGENT, EXAMINER_AGENT, CHAT_AGENT, RAG_AGENT]


def supervisor_node(state: SupervisorState) -> dict:
    """
    Supervisor Agent: Analyzes the user query and decides which agent should handle it.
    
    Routes to:
    - research_agent: For web searches, LeetCode problems, DSA explanations
    - examiner_agent: For generating quiz/MCQ questions
    - chat_agent: For general conversation, greetings, simple questions
    - rag_agent: For questions about uploaded documents/PDFs
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
   practice problems, assessments on any topic.
   
3. **chat_agent** - Use for: general conversation, greetings, simple factual questions 
   that don't need search, explanations from your knowledge, coding help, math.

4. **rag_agent** - Use for: questions about uploaded documents, PDFs, files.
   Keywords that indicate this: "document", "pdf", "file", "uploaded", "the paper", 
   "this document", "what does the file say", "summarize the document", "in the pdf".
   {file_context}

Analyze the user's message and respond with ONLY the agent name (one of: research_agent, examiner_agent, chat_agent, rag_agent).
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
                
    except Exception as e:
        content = f"Error querying documents: {str(e)}. Please make sure documents are uploaded and Qdrant is running."
    
    return {
        "messages": [AIMessage(content=content)],
        "final_response": content
    }


# =============================================================================
# ROUTING FUNCTION
# =============================================================================

def route_to_agent(state: SupervisorState) -> Literal["research_agent", "examiner_agent", "chat_agent", "rag_agent"]:
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
        }
    )
    
    # All agents end after processing
    workflow.add_edge(RESEARCH_AGENT, END)
    workflow.add_edge(EXAMINER_AGENT, END)
    workflow.add_edge(CHAT_AGENT, END)
    workflow.add_edge(RAG_AGENT, END)
    
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
        
        config = {"configurable": {"thread_id": self.session_id}}
        
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

Supported file types: PDF, TXT, MD, CSV, DOC, DOCX
""")


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
    
    # Start interactive mode
    print("\n" + "="*60)
    print("Starting Interactive Mode...")
    print("="*60)
    chatbot.run_interactive()
