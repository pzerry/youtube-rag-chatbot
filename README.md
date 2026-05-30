# 🎥 YouTube RAG Chatbot using LangChain LCEL, FAISS, Hugging Face & Groq

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that enables users to ask natural language questions about YouTube videos.

The system extracts a video's transcript, converts transcript chunks into vector embeddings, stores them in a FAISS vector database, retrieves semantically relevant chunks, and generates grounded answers using Groq-hosted Llama 3.3.

The application is built using LangChain's Expression Language (LCEL), enabling a clean and composable retrieval-generation pipeline.

---

## Features

* YouTube Transcript Extraction
* Semantic Text Chunking
* Hugging Face Embeddings
* FAISS Vector Search
* Retrieval-Augmented Generation (RAG)
* LangChain Expression Language (LCEL)
* Groq Llama 3.3 Integration
* Hallucination Reduction through Context Grounding

---

## Architecture

```text
YouTube Video
      │
      ▼
Transcript Extraction
      │
      ▼
Recursive Text Splitting
      │
      ▼
Sentence Transformer Embeddings
      │
      ▼
FAISS Vector Store
      │
      ▼
Retriever
      │
      ▼
RunnableParallel
 ┌──────────────┬──────────────┐
 │              │              │
 ▼              ▼              │
Context      Question          │
Formatting   Passthrough       │
 └──────────────┴──────────────┘
                │
                ▼
         Prompt Template
                │
                ▼
         Groq Llama 3.3
                │
                ▼
        StrOutputParser
                │
                ▼
          Final Answer
```

---

## Tech Stack

| Component              | Technology                               |
| ---------------------- | ---------------------------------------- |
| Programming Language   | Python                                   |
| Transcript Extraction  | youtube-transcript-api                   |
| Text Splitting         | LangChain RecursiveCharacterTextSplitter |
| Embeddings             | sentence-transformers/all-MiniLM-L6-v2   |
| Vector Database        | FAISS                                    |
| Retrieval Framework    | LangChain                                |
| LLM                    | Groq Llama 3.3 70B                       |
| Chain Orchestration    | LangChain LCEL                           |
| Environment Management | python-dotenv                            |

---

## Workflow

### 1. Transcript Extraction

The transcript is fetched directly from a YouTube video using the YouTube Transcript API.

### 2. Document Chunking

The transcript is split into overlapping chunks using:

* Chunk Size: 1000
* Chunk Overlap: 200

This improves retrieval quality while preserving context across chunk boundaries.

### 3. Embedding Generation

Each chunk is converted into dense vector embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

These embeddings capture semantic meaning rather than relying on keyword matching.

### 4. Vector Database Creation

Embeddings are stored in a FAISS index to enable efficient similarity search.

### 5. Semantic Retrieval

User queries are automatically converted into embeddings and matched against stored vectors to retrieve the most relevant transcript chunks.

### 6. Context Augmentation

Retrieved chunks are formatted and combined into a context block using LCEL's RunnableLambda.

### 7. Answer Generation

LangChain's RunnableParallel executes:

* Context Retrieval Pipeline
* Question Passthrough Pipeline

The outputs are merged into a prompt template and sent to Groq-hosted Llama 3.3 for answer generation.

### 8. Output Parsing

The model response is parsed into a clean string using StrOutputParser.

---

## Example Query

### Input

```text
What is a perceptron?
```

### Output

```text
A perceptron is the simplest form of an artificial neural network used for binary classification. It consists of input features, associated weights, a bias term, and an activation function.
```

---

## LCEL Chain Design

The retrieval and generation process is implemented using LangChain Expression Language (LCEL):

```python
main_chain = (
    parallel_chain
    | prompt
    | llm
    | parser
)
```

This creates a declarative pipeline where retrieval, prompt augmentation, generation, and output parsing are composed into a single executable chain.

---

## Skills Demonstrated

* Retrieval-Augmented Generation (RAG)
* LangChain Expression Language (LCEL)
* Semantic Search
* Vector Databases
* Prompt Engineering
* LLM Integration
* Embedding Models
* AI Application Development
* Python Software Engineering

---

## Future Enhancements

* Multi-Video Knowledge Base
* Persistent FAISS Storage
* Streamlit User Interface
* Hybrid Search (BM25 + Vector Search)
* Source Citations
* Conversational Memory
* Query Rewriting
* Production Deployment

---

## Author

Pooja Sharma

Built to gain hands-on experience with Retrieval-Augmented Generation, vector databases, semantic retrieval, and modern LLM orchestration frameworks.
