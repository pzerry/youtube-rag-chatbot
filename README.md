# 🎥 YouTube RAG Chatbot using LangChain, FAISS, HuggingFace & Groq

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline that enables users to ask natural language questions about the content of a YouTube video.

Instead of sending the entire transcript to the LLM, the system retrieves only the most relevant transcript segments using semantic search and injects them into the prompt. This approach improves response quality, reduces context size, minimizes hallucinations, and demonstrates core concepts used in modern AI-powered search and question-answering systems.

---

## Problem Statement

Large Language Models have limited context windows and may hallucinate when asked questions about long videos.

This project addresses that challenge by:

* Extracting video transcripts from YouTube
* Converting transcript chunks into vector embeddings
* Storing embeddings in a FAISS vector database
* Retrieving semantically relevant chunks for a user query
* Providing grounded answers using an LLM

---

## System Architecture

```text
YouTube Video
       │
       ▼
Transcript Extraction
       │
       ▼
Text Chunking
       │
       ▼
Embedding Generation
       │
       ▼
FAISS Vector Store
       │
       ▼
Semantic Retrieval
       │
       ▼
Prompt Augmentation
       │
       ▼
Groq Llama 3.3
       │
       ▼
Generated Answer
```

---

## Technology Stack

| Component              | Technology                               |
| ---------------------- | ---------------------------------------- |
| Transcript Extraction  | youtube-transcript-api                   |
| Text Splitting         | LangChain RecursiveCharacterTextSplitter |
| Embedding Model        | sentence-transformers/all-MiniLM-L6-v2   |
| Vector Database        | FAISS                                    |
| Retrieval Framework    | LangChain Retriever                      |
| LLM                    | Llama 3.3 70B (Groq)                     |
| Environment Management | python-dotenv                            |
| Programming Language   | Python                                   |

---

## Project Workflow

### 1. Transcript Ingestion

The transcript is extracted directly from a YouTube video using the YouTube Transcript API.

```python
transcript_list = ytt.fetch(video_id)
transcript = " ".join([t.text for t in transcript_list])
```

---

### 2. Document Chunking

The transcript is divided into overlapping chunks to preserve semantic continuity.

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

Benefits:

* Prevents information loss at chunk boundaries
* Improves retrieval quality
* Maintains contextual coherence

---

### 3. Embedding Generation

Each chunk is transformed into a dense vector representation using Sentence Transformers.

Model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model captures semantic meaning rather than relying on keyword matching.

---

### 4. Vector Store Creation

Chunk embeddings are stored inside a FAISS index for efficient similarity search.

```python
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)
```

---

### 5. Semantic Retrieval

User queries are automatically converted into embeddings and compared against stored vectors.

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":4}
)
```

Top-K relevant chunks are retrieved based on cosine similarity.

---

### 6. Prompt Augmentation

Retrieved chunks are combined into a context block and injected into a prompt template.

```python
context_text = "".join(
    [doc.page_content for doc in retrieved_docs]
)
```

This ensures the model answers using transcript content only.

---

### 7. Response Generation

The augmented prompt is sent to Groq-hosted Llama 3.3 70B.

```python
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)
```

The generated response is grounded in the retrieved transcript context.

---

## Example Query

### User Question

```text
What is a perceptron?
```

### Retrieval Phase

Relevant transcript chunks discussing perceptrons and neural networks are retrieved from FAISS.

### Generation Phase

The LLM generates an answer exclusively from retrieved context.

---

## Key RAG Concepts Demonstrated

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Dense Vector Embeddings
* Similarity-Based Retrieval
* Vector Databases
* Prompt Engineering
* Context Injection
* Hallucination Reduction
* Large Language Model Integration

---

## Project Structure

```text
youtube-rag-chatbot/
│
├── indexing.py
├── retriever.py
├── augment.py
├── generate.py
├── main.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd youtube-rag-chatbot
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

---

## Future Enhancements

* Multi-video knowledge base
* Persistent FAISS storage
* Hybrid Search (BM25 + Vector Search)
* Conversation Memory
* Streamlit Frontend
* Source Citations
* Query Rewriting
* Reranking Models
* Production Deployment

---

## Skills Demonstrated

* Retrieval-Augmented Generation (RAG)
* LangChain Framework
* FAISS Vector Database
* Embedding Models
* Prompt Engineering
* LLM Orchestration
* Semantic Search
* Python Development
* AI Application Engineering

---

## Author

Pooja Sharma

Built as a hands-on implementation to understand the complete Retrieval-Augmented Generation pipeline from document ingestion to grounded response generation using open-source AI technologies.
