"""
pharmadoc/persistence.py

Section 9 - Persistence and scaling helpers
Source notebook cells: [38]

Verbatim conversion: the code below this header is copied directly from
the notebook's cell source (mechanical extraction, not retyped). Only this
docstring and the import lines immediately below are new.
"""

# --- external imports (used by this file's verbatim code) ---
from pathlib import Path
from collections import defaultdict
import faiss
import hashlib
import json
import numpy as np

# --- cross-module imports (this package's own files) ---
from .config import PERSIST_DIR
from .retrieval import normalize_retrieval_text

# ===== NOTEBOOK CELLS [38] (verbatim) =====


#@title CELL 28F — Deduplication, persistence, and scaling helpers

def normalized_item_signature(item):
    text = normalize_retrieval_text(item.get("text_for_embedding", ""))
    return (
        item.get("document_id"),
        item.get("page_start"),
        item.get("content_type"),
        hashlib.sha1(text.encode("utf-8")).hexdigest(),
    )


def deduplicate_content_items(content_items):
    unique = []
    seen = set()
    for item in content_items:
        signature = normalized_item_signature(item)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(item)
    return unique


def build_document_centroids(content_items, embeddings):
    grouped = defaultdict(list)
    for item, vector in zip(content_items, embeddings):
        grouped[item["document_id"]].append(vector)

    centroids = {}
    for document_id, vectors in grouped.items():
        centroid = np.mean(np.vstack(vectors), axis=0)
        norm = np.linalg.norm(centroid)
        if norm:
            centroid = centroid / norm
        centroids[document_id] = centroid.astype("float32")
    return centroids


def save_rag_artifacts(
    directory=PERSIST_DIR,
    index=None,
    content_items=None,
    registry=None,
    embeddings=None,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    index = index if index is not None else globals().get("faiss_index")
    content_items = content_items if content_items is not None else globals().get("rag_content_items", [])
    registry = registry if registry is not None else globals().get("document_registry", {})
    embeddings = embeddings if embeddings is not None else globals().get("chunk_embeddings")

    if index is None:
        raise RuntimeError("No FAISS index is available to save.")

    faiss.write_index(index, str(directory / "index.faiss"))
    with open(directory / "content_items.json", "w", encoding="utf-8") as stream:
        json.dump(content_items, stream, ensure_ascii=False, indent=2, default=str)
    with open(directory / "document_registry.json", "w", encoding="utf-8") as stream:
        json.dump(registry, stream, ensure_ascii=False, indent=2, default=str)
    if embeddings is not None:
        np.save(directory / "embeddings.npy", embeddings)

    return str(directory)


def load_rag_artifacts(directory=PERSIST_DIR):
    directory = Path(directory)
    index_path = directory / "index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"No saved index found in {directory}")

    index = faiss.read_index(str(index_path))
    with open(directory / "content_items.json", encoding="utf-8") as stream:
        content_items = json.load(stream)
    with open(directory / "document_registry.json", encoding="utf-8") as stream:
        registry = json.load(stream)

    embeddings_path = directory / "embeddings.npy"
    embeddings = np.load(embeddings_path) if embeddings_path.exists() else None

    if index.ntotal != len(content_items):
        raise RuntimeError(
            "Saved FAISS index and content metadata are misaligned."
        )

    return {
        "faiss_index": index,
        "rag_content_items": content_items,
        "document_registry": registry,
        "chunk_embeddings": embeddings,
    }


def suggest_scaling_strategy(document_count):
    if document_count < 100:
        return "Single FAISS index with metadata filters is appropriate."
    if document_count < 1000:
        return (
            "Use persistent artifacts, batch embeddings, file fingerprints, "
            "and document-level centroid prefiltering."
        )
    return (
        "Use hierarchical retrieval: document-level routing followed by "
        "chunk retrieval, plus a persistent vector database and background ingestion."
    )

