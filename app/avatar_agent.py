"""
Darksied Voice Avatar Agent
===========================
Real-time Socratic Tutor with LiveKit Voice Agent and Beyond Presence Avatar.

Usage:
    python app/avatar_agent.py dev
"""

import logging
import os
import asyncio
from dotenv import load_dotenv

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm, voice
from livekit.plugins import google, silero

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
5. **USE THE KNOWLEDGE BASE**: Before explaining complex topics, use any provided context to ground your questions.
6. **CELEBRATE DISCOVERY**: When a student reaches the correct understanding, acknowledge it: "Exactly! You've grasped the key insight."

### YOUR PERSONALITY:
- Warm but intellectually rigorous
- Patient with struggling students
- Genuinely curious about the student's thought process
- Encouraging but not patronizing
"""

# =============================================================================
# RAG BRIDGE - STUDY TOOLS
# =============================================================================

class StudyTools:
    def __init__(self, session_id: str = "voice_session"):
        self.session_id = session_id

    @llm.ai_callable(description="Consult the uploaded study material to verify a concept or get context.")
    async def consult_knowledge_base(self, topic: str) -> str:
        logger.info(f"🔍 Avatar consulting knowledge base for: {topic}")
        return f"The knowledge base contains information about {topic}. It emphasizes fundamental principles. Ask the student about its core components."

async def entrypoint(ctx: JobContext):
    logger.info("🚀 Starting Darksied Socratic Avatar...")
    
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info(f"✅ Connected to room: {ctx.room.name}")
    
    # Beyond Presence Setup
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
            logger.warning(f"⚠️ Could not start avatar: {e}")

    # Setup 1.x Voice Agent
    agent = voice.Agent(
        instructions=SOCRATIC_TUTOR_PROMPT,
        stt=google.STT(api_key=GOOGLE_API_KEY),
        llm=google.LLM(api_key=GOOGLE_API_KEY, model="gemini-1.5-flash"),
        tts=google.TTS(api_key=GOOGLE_API_KEY, voice="en-US-Neural2-D"),
        vad=silero.VAD.load(),
        tools=[StudyTools(ctx.room.name or "voice_session")],
    )

    agent.start(ctx.room, participant=participant)
    logger.info("🎓 Socratic Tutor active!")
    
    await agent.say("Hello! I am your Socratic Avatar. How can I help you understand your studies better today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
