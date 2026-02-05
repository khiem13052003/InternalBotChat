# InternalBotChat – Retrieval-Augmented Generation (RAG) System

## Overview
**InternalBotChat** is an internal Retrieval-Augmented Generation (RAG) system designed to provide accurate, context-aware responses based on proprietary and internal data sources.  
The system follows a modular, service-oriented architecture to ensure scalability, maintainability, and ease of deployment in production environments.

This repository focuses on:
- Secure ingestion of internal documents
- Efficient semantic retrieval via vector databases
- High-quality LLM inference with retrieved context
- Containerized deployment using Docker

---

## High-Level Architecture

User Query
│
▼
[ Retrieval Service ]
│ ├─ Embed query
│ ├─ Vector search (VectorDB)
│ └─ Context reranking (optional)
▼
[ Inference Service ]
│ ├─ Prompt construction
│ └─ LLM generation
▼
Response

---

## Directory Structure

InternalBotChat
├── docker/
│ ├── .env
│ └── docker-compose.yml
│
├── models/
│ ├── llm/ # LLM weights or model artifacts
│ └── embedding/ # Embedding model weights
│
├── services/
│ ├── ingestion/ # Data ingestion & preprocessing
│ ├── retrieval/ # Query embedding & vector search
│ ├── inference/ # LLM inference & prompt orchestration
│ └── vectorDB/ # Vector database service & configuration
│
├── storage/ # Persistent data (documents, indexes, vectors)
│
└── README.md


---

## Component Description

### 1. Ingestion Service (`services/ingestion`)
Responsible for processing and indexing internal data sources.

**Key responsibilities:**
- Document loading (PDF, DOCX, TXT, Markdown, etc.)
- Text cleaning and normalization
- Chunking strategy (fixed, recursive, or semantic)
- Embedding generation
- Writing vectors to the VectorDB

> This service is typically executed as a batch job or triggered by data updates.

---

### 2. VectorDB Service (`services/vectorDB`)
Handles storage and retrieval of vector embeddings.

**Responsibilities:**
- Vector indexing
- Similarity search (cosine / dot / L2)
- Metadata filtering (source, date, department, permissions)
- Persistence and backup

Examples:
- FAISS
- Milvus
- Qdrant
- Weaviate

---

### 3. Retrieval Service (`services/retrieval`)
Acts as the bridge between user queries and stored knowledge.

**Workflow:**
1. Receive user query
2. Generate query embedding
3. Perform vector similarity search
4. (Optional) Rerank retrieved chunks
5. Return top-K relevant contexts

---

### 4. Inference Service (`services/inference`)
Responsible for generating final responses using an LLM.

**Responsibilities:**
- Prompt template management
- Context injection
- Token limit handling
- LLM inference (local or remote)
- Response post-processing

---

### 5. Models (`models/`)
Stores model artifacts required by the system.

- `llm/`: Language model weights and configs
- `embedding/`: Embedding model weights and configs

> Models are kept separate from code to simplify versioning and deployment.

---

### 6. Storage (`storage/`)
Persistent storage for:
- Raw documents
- Processed chunks
- Vector indexes
- Logs and intermediate artifacts

This directory should be mounted as a Docker volume in production.

---

## Configuration

All environment variables are managed via: docker/.env

---

## Deployment

### Requirements
- Docker >= 24.x
- Docker Compose v2
- (Optional) NVIDIA Container Toolkit for GPU inference

