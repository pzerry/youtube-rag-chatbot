# 🎥 Conversational AI for YouTube Videos

### Context-Aware Question Answering using Retrieval-Augmented Generation (RAG), Semantic Search, and Large Language Models

---

## Overview

Conversational AI for YouTube Videos is an end-to-end Retrieval-Augmented Generation (RAG) application that enables users to interact with YouTube videos through natural language conversations.

The system automatically extracts video transcripts, converts them into semantic embeddings, stores them in a FAISS vector database, and retrieves the most relevant transcript segments to answer user questions. Retrieved context is combined with conversational history and passed to a Large Language Model (LLM) to generate accurate, context-aware, and grounded responses.

The application supports multi-turn conversations, session-based memory, transcript caching, semantic retrieval, and a production-ready API architecture.

Built using FastAPI, LangChain LCEL, FAISS, Sentence Transformers, Groq-hosted Llama 3.3, and Streamlit.

---

## Features

* 🎥 YouTube Transcript Extraction
* 🧠 Retrieval-Augmented Generation (RAG)
* 🔍 Semantic Search using FAISS
* 💬 Multi-Turn Conversational Memory
* ⚡ Transcript Caching for Faster Processing
* 🚀 FastAPI REST API
* 🎨 Interactive Streamlit Frontend
* 🔗 LangChain Expression Language (LCEL)
* 🤖 Groq Llama 3.3 Integration
* 🎯 Context-Aware Answer Generation
* 🚫 Hallucination Reduction through Grounded Retrieval
* 🐳 Dockerized Deployment Support

---

## System Architecture

```text
                        ┌───────────────────┐
                        │   YouTube Video   │
                        └─────────┬─────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │ Transcript Extraction   │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ Recursive Text Splitting│
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ Sentence Embeddings     │
                    │ all-MiniLM-L6-v2        │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ FAISS Vector Store      │
                    └─────────┬───────────────┘
                              │
                              ▼
                        ┌─────────────┐
                        │ Retriever   │
                        └──────┬──────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
      Transcript Context           Conversation History
                │                             │
                └──────────────┬──────────────┘
                               ▼
                    ┌─────────────────────────┐
                    │ Prompt Construction     │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ Groq Llama 3.3 70B      │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ Grounded Response       │
                    └─────────────────────────┘
```

---

## Project Structure

```bash
.
├── app.py                  # Streamlit frontend
├── main.py                 # FastAPI backend
├── rag.py                  # RAG pipeline
├── session_store.py        # Session management
├── models.py              # Request/Response schemas
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── cache/
│   └── faiss_indexes/
│
└── README.md
```

---

## Tech Stack

| Component              | Technology                               |
| ---------------------- | ---------------------------------------- |
| Programming Language   | Python                                   |
| Frontend               | Streamlit                                |
| Backend API            | FastAPI                                  |
| Transcript Extraction  | youtube-transcript-api                   |
| Text Splitting         | LangChain RecursiveCharacterTextSplitter |
| Embeddings             | sentence-transformers/all-MiniLM-L6-v2   |
| Vector Database        | FAISS                                    |
| Retrieval Framework    | LangChain                                |
| Chain Orchestration    | LCEL                                     |
| Large Language Model   | Groq Llama 3.3 70B                       |
| Environment Management | python-dotenv                            |
| Containerization       | Docker                                   |

---

## Workflow

### 1. Transcript Extraction

The transcript is fetched directly from a YouTube video using the YouTube Transcript API.

### 2. Document Chunking

The transcript is divided into overlapping chunks:

* Chunk Size: 1000
* Chunk Overlap: 200

This improves retrieval quality while preserving contextual continuity.

### 3. Embedding Generation

Each chunk is converted into semantic vector embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### 4. Vector Store Creation

Embeddings are indexed and stored in FAISS for efficient similarity search.

### 5. Session Creation

Each processed video creates a unique session.

```python
session_id = str(uuid.uuid4())
```

A session stores:

```python
{
    "session_id": "...",
    "video_id": "...",
    "vectorstore": "...",
    "history": []
}
```

### 6. Semantic Retrieval

User queries are transformed into embeddings and matched against stored vectors to retrieve the most relevant transcript chunks.

### 7. Context Augmentation

Retrieved context and conversation history are combined using LangChain LCEL.

### 8. Answer Generation

The augmented prompt is sent to Groq-hosted Llama 3.3 to generate grounded responses.

### 9. Conversation Memory

Previous user interactions are automatically included in future prompts, enabling natural follow-up questions.

---

## Caching Strategy

To improve performance and reduce repeated processing, FAISS indexes are cached locally.

### First Processing

```text
Cache MISS
```

* Transcript fetched
* Embeddings generated
* FAISS index created

### Subsequent Processing

```text
Cache HIT
```

* Existing FAISS index loaded directly
* No reprocessing required

Cache Location:

```text
cache/faiss_indexes/
```

---

## API Endpoints

### Health Check

```http
GET /
```

### Process Video

```http
POST /process-video
```

Request

```json
{
  "youtube_url": "https://youtube.com/watch?v=VIDEO_ID"
}
```

### Chat

```http
POST /chat
```

Request

```json
{
  "session_id": "your-session-id",
  "question": "What is the main topic of the video?"
}
```

### Get Conversation History

```http
GET /history/{session_id}
```

### Delete Session

```http
DELETE /session/{session_id}
```

### List Active Sessions

```http
GET /sessions
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/conversational-ai-youtube-videos.git

cd conversational-ai-youtube-videos
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## Run Backend

```bash
uvicorn main:app --reload
```

Backend URL:

```text
http://localhost:8000
```

Swagger Documentation:

```text
http://localhost:8000/docs
```

---

## Run Frontend

```bash
streamlit run app.py
```

Frontend URL:

```text
http://localhost:8501
```

---

## Docker Deployment

Build Docker Image:

```bash
docker build -t youtube-conversational-ai .
```

Run Container:

```bash
docker run -p 8000:8000 youtube-conversational-ai
```

Using Docker Compose:

```bash
docker-compose up --build
```

---

## Skills Demonstrated

* Retrieval-Augmented Generation (RAG)
* Conversational AI Systems
* LangChain Expression Language (LCEL)
* Semantic Search
* Vector Databases (FAISS)
* Prompt Engineering
* Session-Based Memory Management
* FastAPI Development
* Streamlit Application Development
* Embedding Models
* Large Language Model Integration
* Docker Containerization
* Production-Ready AI System Design

---

## Future Enhancements

* Multi-Video Knowledge Base
* Persistent Session Storage (Redis/PostgreSQL)
* Hybrid Retrieval (BM25 + Vector Search)
* Source Citations
* Query Rewriting
* Streaming Responses
* User Authentication
* Cloud Deployment

---

## Author

**Pooja Sharma**

Built to explore Retrieval-Augmented Generation (RAG), semantic search, vector databases, conversational AI, and modern Large Language Model orchestration frameworks.
