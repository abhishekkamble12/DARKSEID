# 🤖 Darksied - Supervisor Multi-Agent System with RAG

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-0.2+-green.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg" alt="Qdrant">
  <img src="https://img.shields.io/badge/PostgreSQL-15-blue.svg" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED.svg" alt="Docker">
  <img src="https://img.shields.io/badge/🤗%20HuggingFace-Qwen2.5-yellow.svg" alt="HuggingFace">
  <img src="https://img.shields.io/badge/🎙️%20LiveKit-Voice%20AI-ff6b6b.svg" alt="LiveKit">
</p>

A production-ready, LangGraph-based **multi-agent system** featuring an intelligent Supervisor Agent that routes user queries to specialized agents. Includes **RAG (Retrieval Augmented Generation)** for document Q&A, **Learning Architect** for generating mindmaps & quizzes from documents, **Voice Avatar Mode** with real-time Socratic tutoring, persistent conversation memory with PostgreSQL, and vector storage with Qdrant.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
  - [Docker Setup (Recommended)](#docker-setup-recommended)
  - [Local Development Setup](#local-development-setup)
- [Configuration](#-configuration)
- [Usage](#-usage)
  - [Interactive Mode](#interactive-mode)
  - [Document Upload (RAG)](#document-upload-rag)
  - [Learning Architect (Mindmaps & Quizzes)](#learning-architect-mindmaps--quizzes)
  - [Voice Avatar Mode](#-voice-avatar-mode-livekit-integration)
  - [API Mode](#api-mode)
- [Agent Details](#-agent-details)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Docker Services](#-docker-services)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### 🎯 Intelligent Query Routing
- **Supervisor Agent** automatically analyzes user queries and routes them to the most appropriate specialized agent
- Zero configuration needed - just ask your question naturally

### 🔍 Research Capabilities
- Web search using Tavily API
- LeetCode problem discovery
- DSA (Data Structures & Algorithms) explanations from GeeksforGeeks and NeetCode

### 📚 Document Q&A (RAG)
- Upload PDFs, Word documents, text files, and more
- Automatic text extraction, chunking, and embedding
- Multi-modal support: text, tables, and images from documents
- Session-based document management

### 📝 Quiz Generation
- Generate MCQ questions on any topic
- Customizable difficulty levels
- Perfect for learning and assessment

### 🧠 Learning Architect (NEW!)
- **Mindmap Generation** - Visual Mermaid.js diagrams from document content
- **Quiz Cards** - Structured JSON quiz cards with explanations
- **Powered by HuggingFace** - Uses Qwen/Qwen2.5-7B-Instruct model
- **TTS-Ready** - Quiz cards include text-to-speech formatted text
- **Smart Fallback** - Automatically falls back to Gemini if HF unavailable

### 🎙️ Voice Avatar Mode (NEW!)
- **Real-time Voice Interaction** - Talk to your AI tutor using WebRTC via LiveKit
- **Socratic Teaching Method** - The avatar asks probing questions instead of lecturing
- **All-Google Stack** - Google Gemini 1.5 Flash for STT, LLM, and TTS
- **Visual Avatar** - Optional Beyond Presence integration for visual representation
- **RAG-Connected** - Voice queries are grounded in your uploaded documents

### 💬 General Chat
- Natural conversation capabilities
- Coding help and explanations
- Math and logic problem solving

### 🧠 Persistent Memory
- Conversation history stored in PostgreSQL
- Resume conversations across sessions
- Thread-based chat management

### 🐳 Production Ready
- Fully containerized with Docker Compose
- Scalable microservices architecture
- Health checks and monitoring

---

## 🏗️ Architecture

### Text Mode (Supervisor Multi-Agent)
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│                    (Interactive CLI / API / Web App)                        │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SUPERVISOR AGENT                                  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  • Analyzes user intent using LLM                                   │   │
│   │  • Routes to appropriate specialized agent                          │   │
│   │  • Handles context and session management                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
    ┌───────────┬─────────────────┼─────────────────┬───────────┐
    │           │                 │                 │           │
    ▼           ▼                 ▼                 ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────────────┐ ┌─────────┐ ┌─────────────────┐
│RESEARCH │ │EXAMINER │ │      CHAT       │ │   RAG   │ │    LEARNING     │
│ AGENT   │ │ AGENT   │ │     AGENT       │ │  AGENT  │ │   ARCHITECT     │
│         │ │         │ │                 │ │         │ │                 │
│• Search │ │• MCQ Gen│ │• General Q&A    │ │• Doc Q&A│ │• Mindmap Gen    │
│• DSA    │ │• Quiz   │ │• Coding Help    │ │• PDF    │ │• Quiz Cards     │
│• News   │ │• Test   │ │• Math/Logic     │ │• Summary│ │• Study Material │
└────┬────┘ └─────────┘ └─────────────────┘ └────┬────┘ └───────┬─────────┘
     │                                           │              │
     ▼                                           │              ▼
┌─────────┐                                      │      ┌───────────────┐
│ TAVILY  │                                      │      │  HUGGINGFACE  │
│Search   │                                      │      │ Qwen2.5-7B    │
└─────────┘                                      │      └───────────────┘
                                                 │
                                                 ▼
                                         ┌───────────────┐
                                         │    QDRANT     │
                                         │ Vector Store  │
                                         └───────────────┘

                    ┌─────────────────────────────────┐
                    │         POSTGRESQL              │
                    │   (Conversation Checkpointer)   │
                    │                                 │
                    │  • Thread management            │
                    │  • Message history              │
                    │  • Session persistence          │
                    └─────────────────────────────────┘
```

### Voice Mode (Socratic Avatar)
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER (Voice)                                    │
│                         🎤 Microphone → Speaker 🔊                          │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ WebRTC
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LIVEKIT SERVER                                     │
│                      (Real-time Audio Transport)                            │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VOICE AVATAR AGENT                                   │
│                                                                             │
│   ┌──────────┐    ┌──────────────────┐    ┌──────────┐                     │
│   │   STT    │ →  │  GEMINI 1.5 FLASH │ →  │   TTS    │                     │
│   │  Google  │    │  (Socratic Brain) │    │  Google  │                     │
│   └──────────┘    └────────┬─────────┘    └──────────┘                     │
│                            │                                                │
│                            ▼                                                │
│                   ┌─────────────────┐                                       │
│                   │  STUDY TOOLS    │ ←── RAG Bridge to Qdrant             │
│                   └────────┬────────┘                                       │
│                            │                                                │
│   Optional:       ┌────────▼────────┐                                       │
│                   │ BEYOND PRESENCE │ → Visual Avatar Stream                │
│                   └─────────────────┘                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

#### Text Mode Flow
```
1. User Input (Text)
       │
       ▼
2. Supervisor Analysis ──────────────────────────────┐
       │                                              │
       ▼                                              ▼
3. Agent Selection                            PostgreSQL
   ├── research_agent ──────────► Tavily API  (Save State)
   ├── examiner_agent ──────────► LLM Generation
   ├── chat_agent ──────────────► LLM Response
   ├── rag_agent ───────────────► Qdrant Search
   └── learning_architect_agent ─► Qdrant + HuggingFace
       │                            │
       │                            ├── Mindmap (Mermaid.js)
       │                            └── Quiz Cards (JSON)
       │
       ▼
4. Response Generation
       │
       ▼
5. User Output (Text)
```

#### Voice Mode Flow
```
1. User Speech (Microphone)
       │
       ▼
2. LiveKit WebRTC Transport
       │
       ▼
3. Google STT (Speech-to-Text)
       │
       ▼
4. Gemini 1.5 Flash ─────────────► StudyTools
       │                              │
       │                              ▼
       │                          Qdrant RAG
       │                              │
       ◄─────────────────────────────┘
       │
       ▼
5. Google TTS (Text-to-Speech)
       │
       ▼
6. User Hears Response (Speaker)
```

---

## 📋 Prerequisites

### Required
- **Python 3.11+**
- **Docker & Docker Compose** (for containerized setup)
- **Google API Key** - [Get it here](https://makersuite.google.com/app/apikey)

### Optional (for full features)
- **Tavily API Key** - [Get it here](https://tavily.com/) (for Research Agent)
- **8GB+ RAM** recommended for PDF processing

---

## 🚀 Installation

### Docker Setup (Recommended)

This is the easiest way to get started with all services configured automatically.

#### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd Darksied
```

#### Step 2: Create Environment File

Create a `.env` file in the `Darksied` directory:

```bash
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional (for Research Agent)
TAVILY_API_KEY=your_tavily_api_key_here

# Optional (for RAG API service)
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

#### Step 3: Start Services

```bash
# Start all core services (Agent + PostgreSQL + Qdrant)
docker-compose up -d

# Check if services are running
docker-compose ps

# View logs
docker-compose logs -f agent
```

#### Step 4: Run the Application

```bash
# Interactive mode
docker-compose exec -it agent python project.py

# Or run with Docker flag (waits for services)
docker-compose exec -it agent python project.py --docker
```

#### Optional: Start RAG API Service

```bash
# Start with RAG API microservice
docker-compose --profile rag up -d
```

---

### Local Development Setup

For development without Docker:

#### Step 1: Install Python Dependencies

```bash
cd Darksied
pip install -r app/requirements.txt
```

#### Step 2: Start Infrastructure Services

```bash
# Start Qdrant
docker run -d -p 6333:6333 -p 6334:6334 \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant

# Start PostgreSQL
docker run -d -p 5432:5432 \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=agent_db \
  -v pg_data:/var/lib/postgresql/data \
  postgres:15
```

#### Step 3: Set Environment Variables

```bash
# Linux/Mac
export GOOGLE_API_KEY=your_key
export TAVILY_API_KEY=your_key
export DATABASE_URL=postgresql://user:password@localhost:5432/agent_db
export QDRANT_HOST=localhost
export QDRANT_PORT=6333

# Windows (PowerShell)
$env:GOOGLE_API_KEY="your_key"
$env:TAVILY_API_KEY="your_key"
$env:DATABASE_URL="postgresql://user:password@localhost:5432/agent_db"
$env:QDRANT_HOST="localhost"
$env:QDRANT_PORT="6333"
```

#### Step 4: Run the Application

```bash
python app/project.py
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | ✅ Yes | - | Google AI API key for Gemini |
| `TAVILY_API_KEY` | ⚠️ For Research | - | Tavily API for web search |
| `DATABASE_URL` | ⚠️ For Memory | `postgresql://user:password@localhost:5432/agent_db` | PostgreSQL connection string |
| `QDRANT_HOST` | ⚠️ For RAG | `localhost` | Qdrant server hostname |
| `QDRANT_PORT` | ⚠️ For RAG | `6333` | Qdrant server port |
| `QDRANT_COLLECTION` | No | `documents` | Qdrant collection name |
| `HF_MODEL_ID` | No | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model for Learning Architect |
| `HF_TOKEN` | No | - | HuggingFace token (for gated models) |
| `LIVEKIT_URL` | ⚠️ For Voice | - | LiveKit server WebSocket URL |
| `LIVEKIT_API_KEY` | ⚠️ For Voice | - | LiveKit API key |
| `LIVEKIT_API_SECRET` | ⚠️ For Voice | - | LiveKit API secret |
| `BEY_API_KEY` | No | - | Beyond Presence API key (for avatar) |
| `BEY_AVATAR_ID` | No | - | Beyond Presence avatar ID |
| `OPENAI_API_KEY` | No | - | OpenAI API (for RAG service) |
| `GROQ_API_KEY` | No | - | Groq API (for RAG service) |

### Model Configuration

The default model is `gemini-2.0-flash`. To change it, modify in `project.py`:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Change model here
    google_api_key=GOOGLE_API_KEY,
    temperature=0,  # Adjust creativity (0-1)
    max_retries=2,
)
```

---

## 📖 Usage

### Interactive Mode

Start the chatbot in interactive mode:

```bash
python app/project.py
```

You'll see:
```
✅ Qdrant vector store connected!
✅ PostgreSQL checkpointer connected!
🧠 Loading Learning Architect model on cuda...
✅ Learning Architect model loaded: Qwen/Qwen2.5-7B-Instruct
✅ Supervisor Multi-Agent System initialized!
   Available agents: research_agent, examiner_agent, chat_agent, rag_agent, learning_architect_agent
   Session ID: abc123-def456-...

============================================================
🤖 SUPERVISOR MULTI-AGENT CHATBOT WITH RAG
============================================================
Commands:
  'quit' or 'exit' - End the session
  'upload <path>' - Upload a document for RAG
  'help' - Show help message
============================================================

You: _
```

### Example Conversations

#### General Chat
```
You: Hi! What can you do?
🎯 Supervisor routed to: chat_agent
🤖 Assistant: Hello! I'm a multi-agent AI assistant. I can help you with:
   - Research: Web searches, finding LeetCode problems, DSA explanations
   - Quizzes: Generate MCQ questions on any topic
   - Documents: Answer questions about your uploaded PDFs
   - General: Coding help, explanations, math problems
```

#### Research Query
```
You: What are the top LeetCode problems for dynamic programming?
🎯 Supervisor routed to: research_agent
🤖 Assistant: Based on my search, here are the top DP problems:
   1. Climbing Stairs (Easy)
   2. House Robber (Medium)
   3. Coin Change (Medium)
   ...
```

#### Quiz Generation
```
You: Generate 5 MCQ questions about Python decorators
🎯 Supervisor routed to: examiner_agent
🤖 Assistant: 
   **Question 1:** What symbol is used to apply a decorator?
   A) #
   B) @
   C) $
   D) &
   **Correct Answer: B**
   ...
```

### Document Upload (RAG)

#### Upload a Document
```
You: upload C:\Documents\research_paper.pdf
📄 Indexed document: research_paper.pdf (42 chunks)
✅ Document uploaded and indexed: 42 chunks
```

#### Ask Questions About the Document
```
You: What is the main conclusion of the paper?
🎯 Supervisor routed to: rag_agent
🤖 Assistant: Based on the uploaded document "research_paper.pdf", 
   the main conclusion is that...

You: Summarize chapter 3
🎯 Supervisor routed to: rag_agent
🤖 Assistant: Chapter 3 discusses...
```

#### Supported File Types
| Extension | Type | Notes |
|-----------|------|-------|
| `.pdf` | PDF Documents | Full support with images/tables |
| `.txt` | Plain Text | UTF-8 encoding |
| `.md` | Markdown | Preserves formatting |
| `.csv` | CSV Data | Parsed as structured data |
| `.doc` | Word 97-2003 | Legacy Word format |
| `.docx` | Word Document | Modern Word format |

### Learning Architect (Mindmaps & Quizzes)

The Learning Architect agent transforms your uploaded documents into educational materials using HuggingFace's Qwen2.5-7B-Instruct model.

#### Generate Learning Materials
```
You: upload C:\Documents\machine_learning_basics.pdf
📄 Indexed document: machine_learning_basics.pdf (58 chunks)
✅ Document uploaded and indexed: 58 chunks

You: Create a mindmap from this document
🎯 Supervisor routed to: learning_architect_agent
📚 Learning Materials Generated from: machine_learning_basics.pdf

[PART 1: MINDMAP]
```mermaid
mindmap
  root((Machine Learning))
    Supervised Learning
      Classification
        Decision Trees
        Neural Networks
      Regression
        Linear Regression
        Polynomial Regression
    Unsupervised Learning
      Clustering
        K-Means
        Hierarchical
      Dimensionality Reduction
        PCA
        t-SNE
    Reinforcement Learning
      Q-Learning
      Policy Gradient
```

[PART 2: QUIZ CARDS]
```json
{
  "cards": [
    {
      "question": "Why is feature scaling important in machine learning?",
      "options": ["A) It makes the code run faster", "B) It ensures features contribute equally to the model", "C) It reduces the dataset size", "D) It is only needed for neural networks"],
      "answer": "B",
      "explanation": "Feature scaling ensures that all features contribute proportionally to the model's learning process, preventing features with larger ranges from dominating.",
      "tts_text": "Question: Why is feature scaling important in machine learning? Think carefully about the options."
    }
  ]
}
```
```

#### Example Queries for Learning Architect
```
• "Create a mindmap from the uploaded document"
• "Generate quiz cards from the PDF"
• "Help me learn the concepts in this file"
• "Make study materials from the document"
• "Visualize the key topics as a mindmap"
• "Create flashcards from chapter 3"
```

#### Programmatic Usage

```python
from project import SupervisorChatbot

# Initialize
chatbot = SupervisorChatbot()

# Upload document
chatbot.upload_document("path/to/document.pdf")

# Get both mindmap and quiz
materials = chatbot.generate_learning_materials("Neural Networks concepts")
print(materials["mindmap"])  # Mermaid.js syntax
print(materials["quiz"])     # JSON quiz cards

# Get individual components
mindmap = chatbot.get_mindmap_only("Deep Learning")
quiz = chatbot.get_quiz_only("Backpropagation")
```

#### Output Structure

**Mindmap Output (Mermaid.js)**
- Can be rendered with any Mermaid.js compatible viewer
- Hierarchical visualization of concepts
- Suitable for documentation and presentations

**Quiz Cards Output (JSON)**
```json
{
  "cards": [
    {
      "question": "Question text here",
      "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
      "answer": "Correct answer letter",
      "explanation": "Why this is the correct answer",
      "tts_text": "Text-to-speech friendly version"
    }
  ]
}
```

---

## 🎙️ Voice Avatar Mode (LiveKit Integration)

Darksied isn't just a text bot—it features a **Real-Time Socratic Avatar** that can quiz you verbally using natural conversation.

### Voice Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER (Browser/App)                       │
│                     🎤 Microphone → Speaker 🔊                   │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ WebRTC
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       LIVEKIT SERVER                             │
│                   (Real-time Audio Transport)                    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DARKSIED VOICE PIPELINE                       │
│                                                                  │
│  ┌─────────┐    ┌─────────────┐    ┌─────────┐                  │
│  │  STT    │ →  │   GEMINI    │ →  │   TTS   │                  │
│  │ Google  │    │ 1.5 Flash   │    │ Google  │                  │
│  └─────────┘    └──────┬──────┘    └─────────┘                  │
│                        │                                         │
│                        ▼                                         │
│               ┌────────────────┐                                 │
│               │  STUDY TOOLS   │ ← Connected to Qdrant RAG      │
│               │  (RAG Bridge)  │                                 │
│               └────────────────┘                                 │
│                                                                  │
│  Optional: ┌─────────────────┐                                   │
│            │ BEYOND PRESENCE │ → Visual Avatar Stream            │
│            │     Avatar      │                                   │
│            └─────────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Transport** | LiveKit (WebRTC) | Real-time audio/video streaming |
| **Brain** | Google Gemini 1.5 Flash | Conversation understanding & generation |
| **Voice Input** | Google STT | Speech-to-Text transcription |
| **Voice Output** | Google TTS | Natural text-to-speech |
| **Knowledge** | Darksied RAG Engine | Grounded in uploaded documents |
| **Visuals** | Beyond Presence (Optional) | Animated avatar representation |

### Setup Voice Avatar Mode

#### Step 1: Install Voice Dependencies

```bash
cd Darksied
pip install -r requirements-voice.txt
```

#### Step 2: Configure Environment Variables

Add these to your `.env` file:

```bash
# LiveKit Configuration (get from https://cloud.livekit.io/)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# Beyond Presence Avatar (optional, get from https://beyondpresence.ai/)
BEY_API_KEY=your_beyond_presence_api_key
BEY_AVATAR_ID=your_avatar_id
```

#### Step 3: Run the Voice Avatar Agent

```bash
# Development mode (with hot reload)
python app/avatar_agent.py dev

# Production mode
python app/avatar_agent.py start
```

#### Step 4: Connect via LiveKit Playground

1. Go to the [LiveKit Agents Playground](https://agents-playground.livekit.io/)
2. Connect using your LiveKit credentials
3. Start talking to your Socratic Tutor!

### Socratic Teaching Method

The Voice Avatar uses the **Socratic Method** to teach:

```
┌──────────────────────────────────────────────────────────────┐
│                    SOCRATIC DIALOGUE FLOW                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Student: "What is backpropagation?"                         │
│                                                              │
│  ❌ BAD (Lecture Mode):                                      │
│  "Backpropagation is an algorithm used to train neural       │
│   networks by computing gradients through the chain rule..." │
│                                                              │
│  ✅ GOOD (Socratic Mode):                                    │
│  "Before I explain, tell me: what do you think a neural      │
│   network needs to learn from its mistakes?"                 │
│                                                              │
│  Student: "Um... it needs to know how wrong it was?"         │
│                                                              │
│  Avatar: "Exactly! And how might it figure out which         │
│   weights caused the most error? Think about working         │
│   backwards through the network..."                          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Voice Avatar Features

| Feature | Description |
|---------|-------------|
| 🎯 **Probing Questions** | Asks "why" and "how" instead of lecturing |
| 🔍 **RAG-Grounded** | Queries are verified against uploaded documents |
| 🔄 **Graceful Correction** | Never says "wrong"—redirects with hints |
| ⏸️ **Interruptions** | Can be interrupted mid-speech naturally |
| 📚 **Quiz Generation** | Can generate verbal quizzes on the fly |
| 🎭 **Visual Avatar** | Optional animated avatar via Beyond Presence |

### Example Voice Session

```
🤖 Avatar: "Hello student! I am your Socratic Tutor from Darksied. 
           What topic from your uploaded materials would you like to explore?"

👤 Student: "I want to understand machine learning."

🤖 Avatar: *consults RAG* "I see you've uploaded a document about ML fundamentals.
           Let me start with a question: Why do you think machines need to 'learn' 
           rather than just being programmed with rules?"

👤 Student: "Because there are too many rules to write manually?"

🤖 Avatar: "That's one reason! But think deeper—what kind of problems have 
           patterns that are hard for humans to define explicitly?"

👤 Student: "Like... recognizing faces? We can do it but can't explain how?"

🤖 Avatar: "Excellent insight! You've identified a key motivation for ML.
           Now, given that intuition, how do you think a neural network 
           might learn to recognize patterns?"
```

### Voice Mode vs Text Mode

| Aspect | Text Mode | Voice Mode |
|--------|-----------|------------|
| **Interaction** | Type messages | Speak naturally |
| **Response** | Read text | Hear spoken response |
| **Avatar** | None | Optional visual avatar |
| **Best For** | Detailed research, code review | Quick quizzes, verbal practice |
| **Latency** | Instant | ~500ms (real-time) |

### API Mode

The RAG service can also run as a REST API:

```bash
# Start RAG API
docker-compose --profile rag up -d

# Or locally
cd "mcp tools"
uvicorn Rag:app --host 0.0.0.0 --port 8001
```

---

## 🤖 Agent Details

### 1. Supervisor Agent

**Purpose:** Intelligent query routing

**How it works:**
1. Receives user input
2. Analyzes intent using LLM
3. Selects the best agent for the task
4. Routes the query

**Routing Logic (Text Mode):**
```
User Input → LLM Analysis → Agent Selection
   │
   ├── Contains "search", "find", "weather", "news" → Research Agent
   ├── Contains "quiz", "MCQ", "questions", "test" (general) → Examiner Agent
   ├── Contains "document", "PDF", "file", "summarize" → RAG Agent
   ├── Contains "mindmap", "quiz from document", "study material",
   │   "learning cards", "flashcards from pdf", "help me learn" → Learning Architect
   └── Everything else → Chat Agent
```

**Voice Mode (Separate Pipeline):**
```
User Speech → LiveKit → Voice Avatar Agent (Socratic Tutor)
                              │
                              ├── Always uses Socratic questioning method
                              ├── Connects to RAG via StudyTools
                              └── Responds via TTS (voice output)
```

### 2. Research Agent

**Purpose:** Web research and information retrieval

**Tools:**
- `tavily_search`: General web search
- `find_popular_leetcode_problems`: LeetCode problem discovery
- `get_dsa_explanation`: DSA concepts from GFG/NeetCode

**Example Queries:**
- "What's the weather in Tokyo?"
- "Find sorting algorithms explained"
- "Latest news about AI"

### 3. Examiner Agent

**Purpose:** Educational content generation

**Capabilities:**
- Generate MCQ questions
- Create quizzes on any topic
- Vary difficulty levels
- Provide correct answers

**Example Queries:**
- "Create a quiz about machine learning"
- "Generate 10 hard questions about databases"
- "Make practice problems for Python OOP"

### 4. Chat Agent

**Purpose:** General assistance

**Capabilities:**
- Natural conversation
- Code writing and debugging
- Mathematical calculations
- Concept explanations
- Creative writing

**Example Queries:**
- "Explain recursion like I'm 5"
- "Write a Python function for Fibonacci"
- "What's the difference between REST and GraphQL?"

### 5. RAG Agent

**Purpose:** Document-based Q&A

**Workflow:**
```
Document Upload
      │
      ▼
┌─────────────────┐
│  Text Extraction │ (PyPDF, Unstructured)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Chunking     │ (RecursiveTextSplitter)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embedding     │ (Google Embeddings)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Qdrant Store   │ (Vector Database)
└─────────────────┘

User Query
      │
      ▼
┌─────────────────┐
│ Similarity Search│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Context + LLM   │ → Answer
└─────────────────┘
```

### 6. Learning Architect Agent 🧠

**Purpose:** Generate educational materials (mindmaps & quizzes) from documents

**Model:** HuggingFace Qwen/Qwen2.5-7B-Instruct (with Gemini fallback)

**Workflow:**
```
User Request (e.g., "Create mindmap from document")
      │
      ▼
┌─────────────────┐
│   RAG Retrieval │ (Get relevant chunks from Qdrant)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Context Building│ (Combine chunks into rich context)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  HuggingFace    │ (Qwen2.5-7B-Instruct)
│  OR Gemini      │ (Fallback)
└────────┬────────┘
         │
         ├────────────────┬────────────────┐
         ▼                ▼                ▼
   ┌──────────┐    ┌───────────┐    ┌───────────┐
   │ Mindmap  │    │Quiz Cards │    │ Formatted │
   │(Mermaid) │    │  (JSON)   │    │  Output   │
   └──────────┘    └───────────┘    └───────────┘
```

**Capabilities:**
- 📊 **Mindmap Generation** - Mermaid.js syntax for visual concept hierarchies
- 📝 **Quiz Card Creation** - 5 MCQ questions with explanations
- 🔊 **TTS-Ready Output** - Text-to-speech formatted content
- 🎯 **Context-Grounded** - Uses ONLY document content, no hallucination
- ♻️ **Smart Fallback** - Automatically uses Gemini if HuggingFace unavailable

**Example Queries:**
- "Create a mindmap from this document"
- "Generate quiz cards about chapter 2"
- "Help me learn the concepts in this PDF"
- "Make study materials from the uploaded file"

**System Prompt Philosophy:**
The Learning Architect follows the "Darksied Learning Architect" system prompt which emphasizes:
1. **Dual Output** - Always generates BOTH mindmap AND quiz
2. **Context Grounding** - Uses ONLY retrieved document content
3. **Quality Focus** - Quiz questions test "Why" and "How", not just "What"
4. **Structured Format** - Consistent Mermaid.js and JSON output

### 7. Voice Avatar Agent 🎙️

**Purpose:** Real-time voice-based Socratic tutoring

**Model:** Google Gemini 1.5 Flash (STT, LLM, TTS)

**Transport:** LiveKit (WebRTC)

**Workflow:**
```
User Speech (Microphone)
      │
      ▼
┌─────────────────┐
│   LiveKit       │ (WebRTC Transport)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Google STT     │ (Speech-to-Text)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Gemini 1.5     │ (Understanding + Response)
│     Flash       │
└────────┬────────┘
         │
         ├─── Uses StudyTools ──► Qdrant RAG
         │
         ▼
┌─────────────────┐
│   Google TTS    │ (Text-to-Speech)
└────────┬────────┘
         │
         ▼
   User Hears Response
```

**Study Tools (RAG Bridge):**
- `consult_knowledge_base(topic)` - Query uploaded documents
- `verify_student_answer(topic, answer)` - Check student understanding
- `generate_quiz_question(topic, difficulty)` - Create verbal quizzes

**Capabilities:**
- 🎤 **Real-time Voice** - Natural conversational interaction
- 🧠 **Socratic Method** - Asks questions instead of lecturing
- 📚 **RAG-Grounded** - Responses verified against documents
- 🔄 **Graceful Correction** - Redirects without saying "wrong"
- ⏸️ **Interruptible** - Can be stopped mid-response
- 🎭 **Visual Avatar** - Optional Beyond Presence integration

**Example Interaction:**
```
Student: "Explain backpropagation to me"

Avatar (Socratic): "Before I explain, let me ask you: 
                   What do you think a neural network needs 
                   to learn from its mistakes?"
```

**Running the Voice Avatar:**
```bash
# Development mode
python app/avatar_agent.py dev

# Connect via LiveKit Playground
# https://agents-playground.livekit.io/
```

---

## 📁 Project Structure

```
Darksied/
│
├── app/                          # Main application code
│   ├── project.py               # Main entry point (Text agents)
│   ├── avatar_agent.py          # Voice Avatar agent (LiveKit)
│   └── requirements.txt         # Python dependencies
│
├── mcp tools/                    # MCP (Model Context Protocol) tools
│   ├── Rag.py                   # RAG API service
│   └── requirements-rag.txt     # RAG service dependencies
│
├── uploads/                      # Document upload directory
│
├── docker-compose.yml           # Docker orchestration
├── Dockerfile                   # Main app container
├── Dockerfile.rag               # RAG service container
├── init-db.sql                  # PostgreSQL initialization
│
├── requirements-voice.txt       # Voice Avatar dependencies
├── env.example                  # Environment variables template
├── Chatbot.ipynb                # Jupyter notebook for testing
├── README.md                    # This file
└── .env                         # Environment variables (create this)
```

---

## 📚 API Reference

### RAG API Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "qdrant": "connected",
  "collections": 1
}
```

#### Index PDF
```http
POST /index_pdf
Content-Type: multipart/form-data

file: <PDF file>
session_id: "user123" (optional)
```

**Response:**
```json
{
  "status": "success",
  "message": "Indexed document.pdf successfully",
  "text_chunks": 35,
  "table_chunks": 5,
  "image_chunks": 2
}
```

#### Query Documents
```http
POST /query_rag
Content-Type: application/json

{
  "question": "What is the main topic?",
  "session_id": "user123",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "The main topic is...",
  "relevant_texts": ["chunk1", "chunk2", "chunk3"],
  "sources": ["document.pdf"],
  "image_count": 0
}
```

#### Clear Session
```http
DELETE /clear_session/{session_id}
```

**Response:**
```json
{
  "status": "success",
  "message": "Cleared session: user123"
}
```

---

## 🐳 Docker Services

### Service Overview

| Service | Container Name | Port(s) | Description |
|---------|---------------|---------|-------------|
| `agent` | supervisor-agent | 8000 | Main multi-agent application |
| `postgres` | agent-postgres | 5432 | PostgreSQL for checkpointer |
| `qdrant` | agent-qdrant | 6333, 6334 | Vector database for RAG |
| `rag-api` | rag-api | 8001 | Optional RAG microservice |

### Docker Commands

```bash
# Start all services
docker-compose up -d

# Start with RAG API
docker-compose --profile rag up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v

# Rebuild containers
docker-compose build --no-cache

# Shell into container
docker-compose exec agent bash

# Run Python script in container
docker-compose exec agent python project.py
```

### Resource Requirements

| Service | CPU | RAM | Disk | GPU (Optional) |
|---------|-----|-----|------|----------------|
| agent | 1 core | 2GB | 1GB | - |
| agent (with Learning Architect) | 2 cores | 8GB | 15GB | CUDA GPU recommended |
| postgres | 0.5 core | 512MB | 1GB | - |
| qdrant | 1 core | 1GB | Varies | - |
| rag-api | 1 core | 2GB | 1GB | - |

**Without Learning Architect HF Model:** 4 cores, 8GB RAM
**With Learning Architect HF Model (CPU):** 4 cores, 16GB RAM
**With Learning Architect HF Model (GPU):** 4 cores, 8GB RAM + CUDA GPU (8GB+ VRAM)

> **Note:** The Learning Architect uses HuggingFace Qwen2.5-7B-Instruct. If insufficient resources, it automatically falls back to Gemini API.

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "GOOGLE_API_KEY not found"
```bash
# Check if .env file exists
cat .env

# Ensure the key is set
echo "GOOGLE_API_KEY=your_key_here" >> .env
```

#### 2. "Cannot connect to Qdrant"
```bash
# Check if Qdrant is running
docker-compose ps qdrant

# Check Qdrant logs
docker-compose logs qdrant

# Restart Qdrant
docker-compose restart qdrant
```

#### 3. "Cannot connect to PostgreSQL"
```bash
# Check if PostgreSQL is healthy
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Wait for health check
docker-compose exec postgres pg_isready -U user -d agent_db
```

#### 4. "PDF processing failed"
```bash
# Ensure poppler is installed (in Docker, it's automatic)
# For local development:
# Ubuntu/Debian
apt-get install poppler-utils tesseract-ocr

# macOS
brew install poppler tesseract

# Windows - Download from https://github.com/oschwartz10612/poppler-windows/releases
```

#### 5. "Out of memory during PDF processing"
- Large PDFs require significant RAM
- Try processing smaller files
- Increase Docker memory limit:
```yaml
# In docker-compose.yml
services:
  agent:
    deploy:
      resources:
        limits:
          memory: 4G
```

#### 6. "HuggingFace model failed to load" / Learning Architect issues
```bash
# Check if you have enough RAM (16GB+ for CPU inference)
free -h  # Linux/Mac
Get-Process | Sort-Object WorkingSet -Descending  # Windows

# For GPU acceleration, verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If using a gated model, set your HuggingFace token
export HF_TOKEN=your_huggingface_token

# The system will automatically fall back to Gemini if HF fails
# Check logs for: "⚠️ Could not load HuggingFace model... Falling back to Gemini"
```

#### 7. "Learning Architect returns empty mindmap/quiz"
- Ensure you've uploaded a document first
- Check that Qdrant has indexed the document:
```bash
curl http://localhost:6333/collections/documents
```
- Try a more specific query: "Create a mindmap about [specific topic] from the document"

#### 8. Voice Avatar: "Cannot connect to LiveKit"
```bash
# Verify LiveKit credentials in .env
echo $LIVEKIT_URL
echo $LIVEKIT_API_KEY

# Test connection (Python)
python -c "from livekit import api; print('LiveKit SDK OK')"

# Check if running in development mode
python app/avatar_agent.py dev --help
```

#### 9. Voice Avatar: "Google STT/TTS not working"
```bash
# Verify Google API key has Speech-to-Text and Text-to-Speech APIs enabled
# Go to: https://console.cloud.google.com/apis/library

# Test Google credentials
python -c "from livekit.plugins import google; print('Google plugin OK')"
```

#### 10. Voice Avatar: "No audio input/output"
- Check browser permissions for microphone
- Ensure you're using HTTPS or localhost
- Try the LiveKit Agents Playground: https://agents-playground.livekit.io/

### Debug Mode

Enable verbose logging:
```python
# In project.py, add at the top:
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

```bash
# Check all services
docker-compose ps

# Check individual service health
curl http://localhost:6333/health  # Qdrant
curl http://localhost:8001/health  # RAG API (if running)

# PostgreSQL
docker-compose exec postgres pg_isready -U user
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Development Setup

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Make your changes
4. Run tests:
   ```bash
   pytest tests/
   ```
5. Commit your changes:
   ```bash
   git commit -m 'Add amazing feature'
   ```
6. Push to the branch:
   ```bash
   git push origin feature/amazing-feature
   ```
7. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for functions
- Keep functions focused and small

### Adding a New Agent

1. Define the agent function in `project.py`:
   ```python
   def my_new_agent_node(state: SupervisorState) -> dict:
       """My new agent description."""
       messages = state["messages"]
       last_message = messages[-1].content if messages else ""
       
       # Your agent logic here
       content = "Agent response"
       
       return {
           "messages": [AIMessage(content=content)],
           "final_response": content
       }
   ```

2. Add to agent list:
   ```python
   MY_NEW_AGENT = "my_new_agent"
   AGENT_LIST = [RESEARCH_AGENT, EXAMINER_AGENT, CHAT_AGENT, RAG_AGENT, 
                 LEARNING_ARCHITECT_AGENT, MY_NEW_AGENT]
   ```

3. Update supervisor routing prompt in `supervisor_node()`:
   ```python
   # Add description for your agent
   6. **my_new_agent** - Use for: [describe when to route here]
   ```

4. Update the routing Literal type:
   ```python
   def route_to_agent(state) -> Literal[..., "my_new_agent"]:
   ```

5. Add node and edges to graph in `create_supervisor_graph()`:
   ```python
   workflow.add_node(MY_NEW_AGENT, my_new_agent_node)
   # In conditional edges dict:
   MY_NEW_AGENT: MY_NEW_AGENT,
   # Add end edge:
   workflow.add_edge(MY_NEW_AGENT, END)
   ```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - LLM framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent orchestration
- [Qdrant](https://qdrant.tech/) - Vector database
- [Google Gemini](https://ai.google.dev/) - LLM provider
- [Tavily](https://tavily.com/) - Search API
- [HuggingFace](https://huggingface.co/) - Open source ML models
- [Qwen](https://huggingface.co/Qwen) - Qwen2.5-7B-Instruct model
- [LiveKit](https://livekit.io/) - Real-time voice/video infrastructure
- [Beyond Presence](https://beyondpresence.ai/) - AI avatar technology

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions:** [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email:** your-email@example.com

---

<p align="center">
  Made with ❤️ by the Darksied Team
</p>
