# 🤖 Darksied - Supervisor Multi-Agent System with RAG

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-0.2+-green.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg" alt="Qdrant">
  <img src="https://img.shields.io/badge/PostgreSQL-15-blue.svg" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED.svg" alt="Docker">
</p>

A production-ready, LangGraph-based **multi-agent system** featuring an intelligent Supervisor Agent that routes user queries to specialized agents. Includes **RAG (Retrieval Augmented Generation)** for document Q&A, persistent conversation memory with PostgreSQL, and vector storage with Qdrant.

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
        ┌─────────────┬───────────┴───────────┬─────────────┐
        │             │                       │             │
        ▼             ▼                       ▼             ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   RESEARCH    │ │   EXAMINER    │ │     CHAT      │ │      RAG      │
│    AGENT      │ │    AGENT      │ │    AGENT      │ │    AGENT      │
│               │ │               │ │               │ │               │
│ • Web Search  │ │ • MCQ Gen     │ │ • General Q&A │ │ • Doc Search  │
│ • LeetCode    │ │ • Quiz Create │ │ • Coding Help │ │ • PDF Q&A     │
│ • DSA Explain │ │ • Assessment  │ │ • Math/Logic  │ │ • Summarize   │
└───────┬───────┘ └───────────────┘ └───────────────┘ └───────┬───────┘
        │                                                     │
        ▼                                                     ▼
┌───────────────┐                                     ┌───────────────┐
│    TAVILY     │                                     │    QDRANT     │
│   Search API  │                                     │ Vector Store  │
└───────────────┘                                     └───────────────┘

                    ┌─────────────────────────────────┐
                    │         POSTGRESQL              │
                    │   (Conversation Checkpointer)   │
                    │                                 │
                    │  • Thread management            │
                    │  • Message history              │
                    │  • Session persistence          │
                    └─────────────────────────────────┘
```

### Data Flow

```
1. User Input
       │
       ▼
2. Supervisor Analysis ──────────────────────────────┐
       │                                              │
       ▼                                              ▼
3. Agent Selection                            PostgreSQL
   ├── research_agent ───► Tavily API         (Save State)
   ├── examiner_agent ───► LLM Generation
   ├── chat_agent ───────► LLM Response
   └── rag_agent ────────► Qdrant Search
       │
       ▼
4. Response Generation
       │
       ▼
5. User Output
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
✅ Supervisor Multi-Agent System initialized!
   Available agents: research_agent, examiner_agent, chat_agent, rag_agent
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

**Routing Logic:**
```
User Input → LLM Analysis → Agent Selection
   │
   ├── Contains "search", "find", "weather", "news" → Research Agent
   ├── Contains "quiz", "MCQ", "questions", "test" → Examiner Agent
   ├── Contains "document", "PDF", "file", "uploaded" → RAG Agent
   └── Everything else → Chat Agent
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

---

## 📁 Project Structure

```
Darksied/
│
├── app/                          # Main application code
│   ├── project.py               # Main entry point
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

| Service | CPU | RAM | Disk |
|---------|-----|-----|------|
| agent | 1 core | 2GB | 1GB |
| postgres | 0.5 core | 512MB | 1GB |
| qdrant | 1 core | 1GB | Varies |
| rag-api | 1 core | 2GB | 1GB |

**Total Recommended:** 4 cores, 8GB RAM

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
       # Implementation
       return {"messages": [...], "final_response": "..."}
   ```

2. Add to agent list:
   ```python
   MY_NEW_AGENT = "my_new_agent"
   AGENT_LIST = [..., MY_NEW_AGENT]
   ```

3. Update supervisor routing prompt

4. Add node and edges to graph

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

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions:** [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email:** your-email@example.com

---

<p align="center">
  Made with ❤️ by the Darksied Team
</p>
