# 🔎 RAG-Powered Knowledge Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Pydantic-v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white" alt="Pydantic">
  <img src="https://img.shields.io/badge/SentenceTransformers-2.7%2B-F7931E?style=for-the-badge" alt="SentenceTransformers">
  <img src="https://img.shields.io/badge/Tests-pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="pytest">
  <img src="https://img.shields.io/badge/Local--First-No%20API%20Key-22C55E?style=for-the-badge" alt="Local First">
</p>

<p align="center">
  A basic local-first Retrieval-Augmented Generation app built with Streamlit.<br>
  Upload text documents, ask questions, and inspect the source chunks used to answer them.
</p>

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [How It Works](#how-it-works)
* [Chunking Strategies](#chunking-strategies)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Running the App](#running-the-app)
* [Configuration](#configuration)
* [Running Tests](#running-tests)
* [Design Decisions](#design-decisions)
* [Limitations](#limitations)
* [Roadmap](#roadmap)

---

## Overview

This project demonstrates the basic functioning of a Retrieval-Augmented Generation (RAG) system in a clean, inspectable codebase.

It implements the core RAG loop:

1. Upload text documents.
2. Split documents into chunks.
3. Convert chunks into embeddings.
4. Retrieve the most relevant chunks for a user question.
5. Build a simple answer from the retrieved context.
6. Show the source chunks used to support the answer.

The goal is **clarity**, not production-scale complexity. This project intentionally avoids advanced retrieval pipelines, agents, reranking, persistent vector databases, and LLM-based generation so that the basic RAG mechanism remains easy to understand.

---

## Features

| Capability                  | Detail                                                                              |
| --------------------------- | ----------------------------------------------------------------------------------- |
| **Document ingestion**      | Upload one or more `.txt` files through the Streamlit UI                            |
| **Basic chunking options**  | Character, Word Boundary, Sentence, and Paragraph strategies                        |
| **Local embeddings**        | Uses `sentence-transformers/all-MiniLM-L6-v2` after first model download            |
| **Semantic retrieval**      | Cosine similarity over normalized embeddings using NumPy                            |
| **Source-grounded answers** | Responses are assembled from retrieved sentences rather than hallucinated           |
| **Source attribution**      | Each answer shows the originating file and retrieved chunks                         |
| **Validated data model**    | Pydantic v2 schemas for documents, chunks, settings, retrieval results, and answers |
| **No API key required**     | Runs locally after dependencies and the embedding model are installed/downloaded    |
| **Test suite**              | pytest tests covering chunking, document loading, and answer generation             |

---

## How It Works

```text
┌─────────────────────────────────────────────────────────────────┐
│                        User uploads .txt files                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Document Loader    │  UTF-8 / Latin-1 decode
                  │   document_loader    │  → UploadedDocument
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │     Text Chunker     │  Character / Word Boundary /
                  │      chunking        │  Sentence / Paragraph strategy
                  └──────────┬───────────┘  → List[TextChunk]
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Embedding Model    │  SentenceTransformers
                  │     embeddings       │  all-MiniLM-L6-v2
                  └──────────┬───────────┘  → Normalized NumPy array
                             │
                             ▼
                  ┌──────────────────────┐
                  │  In-Memory Retriever │  Cosine similarity
                  │      retriever       │  over normalized vectors
                  └──────────┬───────────┘  → List[RetrievedChunk]
                             │
                    User asks a question
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Answer Synthesizer  │  Keyword-overlap sentence
                  │      generator       │  selection; no LLM required
                  └──────────┬───────────┘  → ChatAnswer + SourceReferences
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Streamlit Chat UI  │  Answer, sources, and
                  │        app.py        │  retrieved context
                  └──────────────────────┘
```

---

## Chunking Strategies

Chunking quality affects retrieval quality. This project includes four basic chunking strategies that can be selected from the sidebar.

| Strategy          | How it splits                                 | Best for                                       |
| ----------------- | --------------------------------------------- | ---------------------------------------------- |
| **Character**     | Exact character slices at `chunk_size`        | Fastest; acceptable when text is already clean |
| **Word Boundary** | Snaps each boundary back to the nearest space | General purpose; avoids broken words           |
| **Sentence**      | Groups complete sentences up to `chunk_size`  | Factual Q&A and structured prose               |
| **Paragraph**     | Groups complete paragraphs when possible      | Longer documents, reports, and articles        |

These are still intentionally simple chunking strategies. The project does not attempt semantic chunking, recursive document splitting, metadata-aware chunking, or advanced retrieval optimization.

---

## Project Structure

```text
RAG-Powered-Knowledge-Assistant/
│
├── app.py                   # Streamlit entry point
├── requirements.txt         # Minimal dependencies
├── pytest.ini               # pytest configuration
├── README.md
│
├── src/
│   ├── schemas.py           # Pydantic models and ChunkingStrategy enum
│   ├── document_loader.py   # Uploaded .txt file decoding
│   ├── chunking.py          # Basic chunking strategies
│   ├── embeddings.py        # SentenceTransformers wrapper
│   ├── retriever.py         # In-memory cosine similarity retrieval
│   ├── generator.py         # Source-grounded answer synthesis
│   └── rag_pipeline.py      # Orchestrates ingestion and retrieval
│
└── tests/
    ├── test_chunking.py
    ├── test_document_loader.py
    └── test_generator.py
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/drkianmaleki/RAG-Powered-Knowledge-Assistant.git
cd RAG-Powered-Knowledge-Assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

<details>
<summary>Windows PowerShell</summary>

```powershell
.\venv\Scripts\Activate.ps1
```

</details>

<details>
<summary>macOS / Linux</summary>

```bash
source venv/bin/activate
```

</details>

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The embedding model, `sentence-transformers/all-MiniLM-L6-v2`, is downloaded automatically by SentenceTransformers on first use and cached locally for later runs.

---

## Running the App

```bash
streamlit run app.py
```

Then:

1. Select a chunking strategy in the sidebar.
2. Adjust chunk size, chunk overlap, and number of retrieved chunks if needed.
3. Upload one or more `.txt` files.
4. Click **Index uploaded files**.
5. Ask a question in the chat box.
6. Expand **Retrieved context** to inspect the source chunks used to build the answer.

---

## Configuration

All settings are controlled from the Streamlit sidebar and take effect when **Index uploaded files** is clicked.

| Setting           |            Default |                                     Range | Description                               |
| ----------------- | -----------------: | ----------------------------------------: | ----------------------------------------- |
| Chunking strategy |      Word Boundary |                                         — | How text is split into chunks             |
| Chunk size        |                900 |                       300–2500 characters | Maximum characters per chunk              |
| Chunk overlap     |                180 |                          0–800 characters | Shared context between consecutive chunks |
| Retrieved chunks  |                  4 |                                      1–10 | Number of chunks retrieved per question   |
| Embedding model   | `all-MiniLM-L6-v2` | any compatible SentenceTransformers model | Embedding model identifier                |

Pydantic validates settings when they are created. Invalid combinations, such as overlap greater than or equal to chunk size, are rejected with a clear error message.

---

## Running Tests

```bash
pytest tests -v
```

The test suite covers:

* text chunking behavior
* chunking edge cases
* document loading
* answer synthesis
* source attribution

---

## Design Decisions

### No LLM-based answer generation

The answer synthesizer selects and combines sentences from retrieved chunks based on keyword overlap with the user query. This keeps the project fully inspectable and avoids unsupported generation.

Every word in the answer comes from retrieved document context.

### In-memory retrieval

Uploaded documents are embedded and stored in memory for the current Streamlit session. Cosine similarity is computed as a dot product over normalized embeddings.

This is simple and appropriate for a basic RAG demonstration.

### Pydantic v2 throughout

The main data objects are Pydantic models:

* `UploadedDocument`
* `TextChunk`
* `RetrievedChunk`
* `SourceReference`
* `ChatAnswer`
* `RAGSettings`

This keeps the code more explicit, typed, and maintainable.

### Local-first design

The app does not require an API key. After the embedding model is downloaded once, the app can run locally without external model API calls.

---

## Limitations

This is a basic RAG demonstration, not a production RAG system.

Current limitations:

* Supports `.txt` files only.
* Works best with small to medium documents.
* The index is stored in memory and is not persisted after the session ends.
* The answer generation is extractive and simple.
* Complex reasoning and multi-hop questions may not work well.
* No PDF, DOCX, HTML, or web page support.
* No persistent vector database.
* No reranking.
* No hybrid keyword/vector retrieval.
* No metadata filtering.
* No LLM-based response generation.
* No authentication or deployment infrastructure.

---

## Roadmap

Potential enhancements for a more advanced future version:

* [ ] PDF and DOCX support
* [ ] Persistent vector storage with ChromaDB or FAISS
* [ ] Hybrid retrieval using dense embeddings and sparse keyword search
* [ ] Reranking with a cross-encoder
* [ ] LLM-based answer generation
* [ ] Citation-aware response formatting
* [ ] Retrieval evaluation metrics
* [ ] Docker packaging
* [ ] GitHub Actions CI
* [ ] Deployment
* [ ] Multi-user session isolation
* [ ] Monitoring and observability

---

## Professional Focus

This project demonstrates the basic mechanics of RAG in a clear and inspectable way:

* document upload
* chunking
* embeddings
* semantic retrieval
* source-grounded answering
* Streamlit app development
* Pydantic validation
* simple test coverage

It is intended as a foundational RAG project before building a more advanced retrieval system.
