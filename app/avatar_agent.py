"""
Darksied Voice Avatar Agent
===========================
Real-time Socratic Tutor with LiveKit Voice Pipeline and Beyond Presence Avatar.

This module creates an interactive voice-based learning assistant that:
- Uses Google Gemini 1.5 Flash for STT, LLM, and TTS
- Connects to the Darksied RAG system for knowledge retrieval
- Employs Socratic questioning methodology
- Optionally displays a Beyond Presence avatar

Usage:
    python app/avatar_agent.py dev
"""

import logging
import os
import asyncio
from typing import Optional
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import google

# Try to import Beyond Presence plugin (optional)
try:
    from livekit.plugins import bey
    BEY_AVAILABLE = True
except ImportError:
    BEY_AVAILABLE = False
    logging.warning("Beyond Presence plugin not available. Running in voice-only mode.")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("darksied-avatar")

# =============================================================================
# CONFIGURATION
# =============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
BEY_API_KEY = os.getenv("BEY_API_KEY")
BEY_AVATAR_ID = os.getenv("BEY_AVATAR_ID")

# Qdrant configuration for RAG connection
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")

# =============================================================================
# SOCRATIC TUTOR SYSTEM PROMPT
# =============================================================================

SOCRATIC_TUTOR_PROMPT = """You are the Darksied Socratic Tutor, an AI avatar that helps students learn through guided questioning.

### YOUR CORE PHILOSOPHY:
You believe that true understanding comes from self-discovery, not lectures. Your role is to guide students to find answers themselves.

### YOUR BEHAVIOR:
1. **ASK, DON'T TELL**: Instead of explaining concepts, ask probing questions that lead the student to the answer.
2. **SHORT RESPONSES**: Keep your responses under 2-3 sentences. Long lectures lose attention.
3. **CHALLENGE ASSUMPTIONS**: If a student gives a surface-level answer, dig deeper with "Why do you think that?" or "What would happen if...?"
4. **GRACEFUL CORRECTION**: If the student makes a mistake, DO NOT correct them immediately. Instead, say: "Interesting thought. But consider [Concept X] - does that change your answer?"
5. **USE THE KNOWLEDGE BASE**: Before explaining complex topics, use the 'consult_knowledge_base' tool to ground your questions in the uploaded material.
6. **CELEBRATE DISCOVERY**: When a student reaches the correct understanding, acknowledge it: "Exactly! You've grasped the key insight."

### YOUR PERSONALITY:
- Warm but intellectually rigorous
- Patient with struggling students
- Genuinely curious about the student's thought process
- Encouraging but not patronizing

### EXAMPLE INTERACTION:
Student: "What is backpropagation?"
BAD Response: "Backpropagation is an algorithm used to train neural networks by computing gradients..."
GOOD Response: "Before I explain, tell me: what do you think a neural network needs to learn from its mistakes?"

### STARTING THE SESSION:
Begin by asking what topic the student wants to review from their uploaded documents."""

# =============================================================================
# RAG BRIDGE - STUDY TOOLS
# =============================================================================

class StudyTools(llm.FunctionContext):
    """
    Tools that allow the Voice Avatar to access the Darksied RAG Knowledge Base.
    
    In production, this would connect to the existing Qdrant vector store
    and query documents indexed by the main Darksied system.
    """
    
    def __init__(self, session_id: str = "voice_session"):
        super().__init__()
        self.session_id = session_id
        self._qdrant_client = None
    
    async def _get_qdrant_client(self):
        """Lazy initialization of Qdrant client."""
        if self._qdrant_client is None:
            try:
                from qdrant_client import QdrantClient
                self._qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
                logger.info(f"✅ Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
            except Exception as e:
                logger.warning(f"⚠️ Could not connect to Qdrant: {e}")
                self._qdrant_client = None
        return self._qdrant_client
    
    async def _query_qdrant(self, topic: str, k: int = 3) -> Optional[str]:
        """
        Query the Qdrant vector store for relevant documents.
        
        Args:
            topic: The topic to search for
            k: Number of results to return
            
        Returns:
            Concatenated relevant text or None if unavailable
        """
        try:
            # Try to use the existing RAG infrastructure
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from langchain_qdrant import QdrantVectorStore
            
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
            
            client = await self._get_qdrant_client()
            if client is None:
                return None
            
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=QDRANT_COLLECTION,
                embedding=embeddings,
            )
            
            # Search with session filter
            results = vector_store.similarity_search(
                topic,
                k=k,
                filter={"session_id": self.session_id}
            )
            
            if results:
                context_parts = [doc.page_content for doc in results]
                return "\n\n".join(context_parts)
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ RAG query failed: {e}")
            return None
    
    @llm.ai_callable(
        description="Consult the uploaded study material to verify a concept, get context, or find specific information from the student's documents."
    )
    async def consult_knowledge_base(self, topic: str) -> str:
        """
        Query the Darksied knowledge base for information about a topic.
        
        Args:
            topic: The concept or topic to look up in the uploaded documents
            
        Returns:
            Relevant context from the documents or a helpful message
        """
        logger.info(f"🔍 Avatar consulting knowledge base for: {topic}")
        
        # Try real RAG lookup first
        rag_result = await self._query_qdrant(topic)
        
        if rag_result:
            logger.info(f"✅ Found RAG context for: {topic}")
            return f"""Found relevant information in the uploaded documents:

{rag_result}

Use this context to formulate your Socratic questions."""
        
        # Fallback: Mock response for demonstration/hackathon
        logger.info(f"📚 Using mock response for: {topic} (RAG unavailable or no results)")
        
        mock_responses = {
            "default": f"""Accessing Darksied Knowledge Graph...
            
Context found for '{topic}':
- This concept appears in the student's uploaded materials
- Key aspects involve understanding the fundamental principles
- Related concepts may include prerequisite knowledge

Use this context to ask probing questions about {topic}. 
Remember: Guide them to discover the answer, don't lecture.""",
        }
        
        return mock_responses.get(topic.lower(), mock_responses["default"])
    
    @llm.ai_callable(
        description="Check the student's answer against the knowledge base to see if it's correct or needs refinement."
    )
    async def verify_student_answer(self, topic: str, student_answer: str) -> str:
        """
        Verify a student's answer against the knowledge base.
        
        Args:
            topic: The topic being discussed
            student_answer: What the student said/claimed
            
        Returns:
            Guidance on whether the answer is correct and how to respond
        """
        logger.info(f"🎯 Verifying student answer about: {topic}")
        
        # Try real RAG lookup
        rag_result = await self._query_qdrant(topic)
        
        if rag_result:
            return f"""Knowledge base context for verification:

{rag_result}

Student's answer: "{student_answer}"

Compare the student's answer with the context above. 
If correct: Acknowledge and probe deeper.
If partially correct: Ask a follow-up question about the missing piece.
If incorrect: Gently redirect with "What about [X]? How does that fit in?" """
        
        # Fallback mock
        return f"""Analyzing student's response about '{topic}':
Student said: "{student_answer}"

Guidance:
- This requires verification against the uploaded material
- Ask the student to explain their reasoning
- If uncertain, use Socratic questioning: "What led you to that conclusion?"
- Encourage them to refer back to specific parts of their study material"""

    @llm.ai_callable(
        description="Generate a quiz question based on the uploaded study material to test the student's understanding."
    )
    async def generate_quiz_question(self, topic: str, difficulty: str = "medium") -> str:
        """
        Generate a quiz question from the knowledge base.
        
        Args:
            topic: The topic to quiz on
            difficulty: easy, medium, or hard
            
        Returns:
            A Socratic quiz question
        """
        logger.info(f"📝 Generating {difficulty} quiz question for: {topic}")
        
        rag_result = await self._query_qdrant(topic)
        
        if rag_result:
            return f"""Based on the uploaded material about '{topic}':

{rag_result[:500]}...

Generate a {difficulty}-level Socratic question that:
1. Tests conceptual understanding, not memorization
2. Requires the student to explain "why" or "how"
3. Can be answered in 1-2 sentences

Remember: Frame it as a genuine inquiry, not a quiz show question."""
        
        return f"""Generate a {difficulty}-level Socratic question about '{topic}'.
        
Frame it to test understanding, not recall. For example:
- "Why do you think [concept] works this way?"
- "What would happen if we changed [variable]?"
- "How does [concept A] relate to [concept B]?"

Make the student think, don't make them recite."""


# =============================================================================
# VOICE PIPELINE ENTRYPOINT
# =============================================================================

async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the LiveKit Voice Pipeline Agent.
    
    This function:
    1. Connects to the LiveKit room
    2. Optionally starts the Beyond Presence avatar
    3. Initializes the Gemini-powered voice pipeline
    4. Begins the Socratic tutoring session
    """
    logger.info("🚀 Starting Darksied Socratic Avatar...")
    
    # Connect to the LiveKit room (audio only, avatar handles video)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info(f"✅ Connected to room: {ctx.room.name}")
    
    # --- A. Setup Avatar (Beyond Presence) ---
    participant = None
    
    if BEY_AVAILABLE and BEY_AVATAR_ID and BEY_API_KEY:
        try:
            logger.info("🎭 Initializing Beyond Presence Avatar...")
            avatar = bey.AvatarSession(
                avatar_id=BEY_AVATAR_ID,
                api_key=BEY_API_KEY
            )
            await avatar.start(ctx.room)
            participant = avatar
            logger.info("✅ Avatar started successfully!")
        except Exception as e:
            logger.warning(f"⚠️ Could not start avatar: {e}. Running in voice-only mode.")
            participant = None
    else:
        logger.info("🎤 Running in voice-only mode (no avatar configured)")
    
    # --- B. Setup the Socratic Brain (Gemini 1.5 Flash) ---
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=SOCRATIC_TUTOR_PROMPT
    )
    
    # --- C. Initialize Study Tools (RAG Connection) ---
    # Use a session ID that could be passed from the frontend
    session_id = ctx.room.name or "voice_session"
    study_tools = StudyTools(session_id=session_id)
    
    # --- D. The Voice Pipeline (STT -> LLM -> TTS) ---
    # Using All-Google Stack for consistency and low latency
    try:
        agent = VoicePipelineAgent(
            vad=agents.VAD.load(),                    # Voice Activity Detection
            stt=google.STT(
                api_key=GOOGLE_API_KEY,
                model="latest_long",                   # Best for conversational audio
            ),
            llm=google.LLM(
                api_key=GOOGLE_API_KEY,
                model="gemini-1.5-flash",              # Fast, capable model
                temperature=0.7,                       # Some creativity for natural conversation
            ),
            tts=google.TTS(
                api_key=GOOGLE_API_KEY,
                voice="en-US-Neural2-D",               # Natural male voice
                speaking_rate=1.0,                     # Normal speed
            ),
            fnc_ctx=study_tools,                       # Connect to RAG
            chat_ctx=initial_ctx,                      # System prompt
        )
        logger.info("✅ Voice pipeline initialized with Google Gemini stack")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize voice pipeline: {e}")
        raise
    
    # --- E. Start the Agent ---
    agent.start(ctx.room, participant=participant)
    logger.info("🎓 Socratic Tutor is now active!")
    
    # --- F. Initial Greeting ---
    greeting = (
        "Hello, student! I am your Socratic Tutor from Darksied. "
        "My goal is not to lecture you, but to help you discover understanding through questions. "
        "What topic from your uploaded materials would you like to explore today?"
    )
    
    await agent.say(greeting, allow_interruptions=True)
    logger.info("👋 Greeting delivered, awaiting student response...")


# =============================================================================
# WORKER ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Validate required environment variables
    if not GOOGLE_API_KEY:
        logger.error("❌ GOOGLE_API_KEY is required!")
        exit(1)
    
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        logger.warning("⚠️ LiveKit credentials not fully configured. Some features may not work.")
    
    logger.info("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║           🎓 DARKSIED SOCRATIC AVATAR                         ║
    ║                                                               ║
    ║   Voice-based learning assistant powered by:                  ║
    ║   • Google Gemini 1.5 Flash (STT/LLM/TTS)                    ║
    ║   • LiveKit (Real-time communication)                         ║
    ║   • Beyond Presence (Avatar - optional)                       ║
    ║   • Darksied RAG Engine (Knowledge base)                      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Run the LiveKit agent
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
