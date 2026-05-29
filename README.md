# 🔎 RAG-Powered Knowledge Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Pydantic-v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white" alt="Pydantic">
  <img src="https://img.shields.io/badge/SentenceTransformers-2.7%2B-F7931E?style=for-the-badge" alt="SentenceTransformers">
  <img src="https://img.shields.io/badge/Tests-pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="pytest">
  <img src="https://img.shields.io/badge/No%20API%20Key-Local%20First-22C55E?style=for-the-badge" alt="Local First">
</p>

<p align="center">
  A fully local, no-API-key Retrieval-Augmented Generation application built with Streamlit.<br>
  Upload documents, ask questions, and get source-grounded answers — entirely on your machine.
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Chunking Strategies](#chunking-strategies)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Configuration](#configuration)
- [Running Tests](#running-tests)
- [Design Decisions](#design-decisions)
- [Limitations](#limitations)
- [Roadmap](#roadmap)

---

## Overview

This project implements the complete RAG loop — document ingestion, chunking, embedding, semantic retrieval, and answer synthesis — in a clean, inspectable codebase. Every step is explicit and testable. There are no hidden API calls, no cloud services, and no dependencies beyond the packages in `requirements.txt`.

It is designed to be a solid foundation for understanding how RAG systems work before layering on complexity.

---

## Features

| Capability | Detail |
|---|---|
| **Document ingestion** | Upload one or more `.txt` files through the Streamlit UI |
| **Four chunking strategies** | Character, Word Boundary, Sentence, Paragraph — selectable at runtime |
| **Local embeddings** | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) runs entirely offline |
| **Semantic retrieval** | Cosine similarity over normalized embeddings via NumPy |
| **Source-grounded answers** | Responses are assembled from retrieved sentences; no hallucination |
| **Source attribution** | Every answer links back to the originating file and chunk |
| **Validated data model** | Pydantic v2 schemas with model-level constraint validation throughout |
| **Progressive loading** | Sidebar renders immediately; heavy models load with a status indicator |
| **No API key required** | Fully offline after initial model download |
| **Test suite** | 22 pytest tests covering chunking logic, document loading, and answer generation |

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        User uploads .txt files                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Document Loader    │  UTF-8 / Latin-1 decode
                  │   (document_loader)  │  → UploadedDocument
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    Text Chunker      │  Character / Word Boundary /
                  │    (chunking)        │  Sentence / Paragraph strategy
                  └──────────┬───────────┘  → List[TextChunk]
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Embedding Model    │  SentenceTransformers
                  │   (embeddings)       │  all-MiniLM-L6-v2
                  └──────────┬───────────┘  → Normalized NumPy array
                             │
                             ▼
                  ┌──────────────────────┐
                  │  In-Memory Retriever │  Cosine similarity (dot product
                  │  (retriever)         │  on normalized vectors)
                  └──────────┬───────────┘  → List[RetrievedChunk]
                             │
                    User asks a question
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Answer Synthesizer  │  Keyword-overlap sentence
                  │  (generator)         │  selection; no LLM required
                  └──────────┬───────────┘  → ChatAnswer + SourceReferences
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Streamlit Chat UI  │  Answer, sources, and full
                  │   (app.py)           │  retrieved context on request
                  └──────────────────────┘
```

---

## Chunking Strategies

Chunking quality directly affects retrieval quality. Four strategies are available and can be switched in the sidebar without restarting the app.

| Strategy | How it splits | Best for |
|---|---|---|
| **Character** | Exact character slices at `chunk_size` | Fastest; acceptable when text is already clean |
| **Word Boundary** | Snaps each boundary back to the nearest space | General purpose; no broken words *(default)* |
| **Sentence** | Groups complete sentences up to `chunk_size`; oversized sentences fall back to word-boundary | Factual Q&A, structured prose |
| **Paragraph** | Groups complete paragraphs; oversized paragraphs fall back to word-boundary | Long-form documents, reports, articles |

All strategies share the same `chunk_size` and `chunk_overlap` controls. Structure-aware strategies (Sentence, Paragraph) use a lighter normalisation pass that preserves newlines before splitting; Character and Word Boundary use full whitespace collapse.

---

## Project Structure

```
RAG-Streamlit-Knowledge-Assistant/
│
├── app.py                   # Streamlit entry point
├── requirements.txt
├── pytest.ini
│
├── src/
│   ├── schemas.py           # Pydantic models + ChunkingStrategy enum
│   ├── document_loader.py   # UTF-8/Latin-1 file decoding
│   ├── chunking.py          # Four chunking strategies + clean_text
│   ├── embeddings.py        # SentenceTransformers wrapper
│   ├── retriever.py         # In-memory cosine similarity retrieval
│   ├── generator.py         # Keyword-based sentence extraction
│   └── rag_pipeline.py      # Orchestrates ingestion → retrieval
│
└── tests/
    ├── test_chunking.py      # 22 tests — all strategies, edge cases
    ├── test_document_loader.py
    └── test_generator.py
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/drkianmaleki/RAG-Powered-Knowledge-Assistant-local.git
cd RAG-Powered-Knowledge-Assistant-local
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
```

<details>
<summary>Windows (PowerShell)</summary>

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

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

> The embedding model (`all-MiniLM-L6-v2`, ~90 MB) is downloaded automatically by SentenceTransformers on first use and cached locally for all subsequent runs.

---

## Running the App

```bash
streamlit run app.py
```

The sidebar and page title render immediately. A status indicator appears while the embedding model and pipeline dependencies load in the background — this only happens once per session.

**Quickstart**

1. Select a **Chunking strategy** in the sidebar.
2. Adjust **Chunk size**, **Chunk overlap**, and **Retrieved chunks** as needed.
3. Click **Browse files** and upload one or more `.txt` files.
4. Click **Index uploaded files**.
5. Type a question in the chat box.
6. Expand **Retrieved context** to inspect the exact source chunks used to build the answer.

---

## Configuration

All settings are accessible from the sidebar and take effect when you click **Index uploaded files**.

| Setting | Default | Range | Description |
|---|---|---|---|
| Chunking strategy | Word Boundary | — | How text is split into chunks |
| Chunk size | 900 | 300 – 2 500 chars | Maximum characters per chunk |
| Chunk overlap | 180 | 0 – 800 chars | Shared context between consecutive chunks |
| Retrieved chunks (top-k) | 4 | 1 – 10 | Number of chunks retrieved per question |
| Embedding model | `all-MiniLM-L6-v2` | any HF model | SentenceTransformers model identifier |

Pydantic validates all settings on construction — invalid combinations (e.g. overlap ≥ chunk size) are caught immediately with a clear error message.

---

## Running Tests

```bash
pytest tests -v
```

```
tests/test_chunking.py          22 tests  — strategies, validation, edge cases
tests/test_document_loader.py    X tests  — encoding, empty files
tests/test_generator.py          X tests  — answer synthesis, source attribution
```

---

## Design Decisions

**No LLM for answer generation**
The answer synthesiser (`generator.py`) selects and combines sentences from retrieved chunks based on keyword overlap with the query. This keeps the system fully offline, deterministic, and transparent — every word in the answer can be traced directly to a source chunk.

**In-memory retrieval**
Uploaded documents are embedded and stored in a NumPy array for the duration of the session. Cosine similarity is computed as a dot product over L2-normalised vectors — fast, dependency-free, and easy to inspect.

**Pydantic v2 throughout**
Every data object (`UploadedDocument`, `TextChunk`, `RetrievedChunk`, `ChatAnswer`, `RAGSettings`) is a validated Pydantic model. Constraints are enforced at the model level via `@model_validator` — callers cannot construct invalid state.

**Deferred heavy imports**
`sentence-transformers` and the pipeline modules are imported inside a `@st.cache_resource` function, so the Streamlit sidebar and title render immediately while dependencies load in the background.

---

## Limitations

- Works best with plain `.txt` files of small to medium length.
- Does not support PDF, DOCX, HTML, or web pages.
- The in-memory index is not persisted — re-indexing is required after each session.
- Answer quality is bounded by keyword overlap; complex multi-hop reasoning is not supported.
- No reranking, hybrid search, or metadata filtering.

---

## Roadmap

Potential enhancements for a more advanced version of this system:

- [ ] PDF and DOCX document support
- [ ] Persistent vector storage (ChromaDB or FAISS)
- [ ] Hybrid retrieval (dense + sparse / BM25)
- [ ] Reranking (cross-encoder)
- [ ] LLM-based answer generation (Anthropic / OpenAI / local Ollama)
- [ ] Citation-aware response formatting
- [ ] Retrieval evaluation metrics (MRR, NDCG, faithfulness)
- [ ] Docker packaging
- [ ] Multi-user session isolation
- [ ] Monitoring and observability
