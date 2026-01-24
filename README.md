# Supervisor Multi-Agent System with RAG

A LangGraph-based multi-agent system featuring a Supervisor Agent that routes queries to specialized agents, including a RAG (Retrieval Augmented Generation) agent for document Q&A.

## 🏗️ Architecture

```
                         ┌─────────────────┐
                         │   User Query    │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │   SUPERVISOR    │
                         │     AGENT       │
                         └────────┬────────┘
                                  │
        ┌───────────┬─────────────┼─────────────┬───────────┐
        │           │             │             │           │
┌───────▼───────┐   │   ┌─────────▼─────────┐   │   ┌───────▼───────┐
│   RESEARCH    │   │   │     EXAMINER      │   │   │     CHAT      │
│    AGENT      │   │   │      AGENT        │   │   │     AGENT     │
│  (Tavily)     │   │   │   (MCQ Gen)       │   │   │   (General)   │
└───────────────┘   │   └───────────────────┘   │   └───────────────┘
                    │                           │
            ┌───────▼───────┐                   │
            │      RAG      │                   │
            │     AGENT     │◄──────────────────┘
            │  (Documents)  │
            └───────┬───────┘
                    │
            ┌───────▼───────┐
            │    QDRANT     │
            │ Vector Store  │
            └───────────────┘
```

## 🚀 Features

| Agent | Description | Tools |
|-------|-------------|-------|
| **Supervisor** | Routes queries to appropriate agents | LLM-based routing |
| **Research** | Web searches, LeetCode problems, DSA explanations | Tavily Search |
| **Examiner** | Generates MCQ/quiz questions | LLM |
| **Chat** | General conversation, coding help | LLM |
| **RAG** | Document Q&A from uploaded PDFs | Qdrant Vector Store |

## 📦 Infrastructure

- **Qdrant**: Vector database for document embeddings (RAG)
- **PostgreSQL**: Checkpointer for conversation memory persistence
- **Docker Compose**: Orchestrates all services

## 🛠️ Setup

### Prerequisites

- Docker & Docker Compose
- API Keys:
  - Google API Key (Gemini)
  - Tavily API Key (optional, for research)

### 1. Clone and Configure

```bash
cd Darksied

# Create .env file with your API keys
cat > .env << EOF
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
EOF
```

### 2. Start Services

```bash
# Start all services (Agent + PostgreSQL + Qdrant)
docker-compose up -d

# Or start with RAG API service
docker-compose --profile rag up -d
```

### 3. Access the Application

```bash
# Interactive mode
docker-compose exec agent python project.py

# Or run locally (without Docker)
python project.py
```

## 📄 Using RAG (Document Q&A)

### Upload Documents

```bash
# In interactive mode
You: upload /path/to/document.pdf
✅ Document uploaded and indexed: 42 chunks

# Then ask questions
You: What is the main topic of the document?
🤖 Assistant: Based on the uploaded document...
```

### Supported File Types

- PDF (`.pdf`)
- Text (`.txt`)
- Markdown (`.md`)
- CSV (`.csv`)
- Word Documents (`.doc`, `.docx`)

## 🐳 Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `agent` | 8000 | Main application |
| `postgres` | 5432 | Checkpointer (conversation memory) |
| `qdrant` | 6333/6334 | Vector database (RAG) |
| `rag-api` | 8001 | Optional RAG microservice |

## 📝 API Endpoints (RAG Service)

If running the RAG API separately:

```bash
# Index a PDF
curl -X POST "http://localhost:8001/index_pdf" \
  -F "file=@document.pdf" \
  -F "session_id=user123"

# Query documents
curl -X POST "http://localhost:8001/query_rag" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "session_id": "user123"}'
```

## 🎯 Example Queries

| Query | Agent |
|-------|-------|
| "Hi, how are you?" | Chat |
| "Explain recursion" | Chat |
| "What's the weather in NYC?" | Research |
| "Find LeetCode problems on graphs" | Research |
| "Generate MCQs about Python" | Examiner |
| "What does the document say about X?" | RAG |
| "Summarize chapter 2 from the PDF" | RAG |

## 📁 Project Structure

```
Darksied/
├── app/
│   ├── project.py          # Main application
│   └── requirements.txt    # Python dependencies
├── mcp tools/
│   ├── Rag.py              # RAG API service
│   └── requirements-rag.txt
├── docker-compose.yml      # Docker orchestration
├── Dockerfile              # Main app container
├── Dockerfile.rag          # RAG service container
├── init-db.sql             # PostgreSQL initialization
└── README.md
```

## 🔧 Local Development

```bash
# Install dependencies
pip install -r app/requirements.txt

# Start Qdrant locally
docker run -p 6333:6333 qdrant/qdrant

# Start PostgreSQL locally
docker run -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:15

# Run the application
python project.py
```

## 📊 Monitoring

- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **PostgreSQL**: Connect with any SQL client to `localhost:5432`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📜 License

MIT License
