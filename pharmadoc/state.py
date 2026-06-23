"""
Session state container for PharmaDoc AI's Gradio application.

RAGState is an explicit dataclass that holds all document-processing
session data. Threading state explicitly through Gradio's gr.State()
ensures correct behaviour in multi-user server deployments.
"""
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RAGState:
    file_paths: list = field(default_factory=list)
    document_registry: dict = field(default_factory=dict)
    all_content_items: list = field(default_factory=list)
    chunked_content_items: list = field(default_factory=list)
    all_table_items: list = field(default_factory=list)
    all_ocr_items: list = field(default_factory=list)
    all_plot_items: list = field(default_factory=list)
    rag_content_items: list = field(default_factory=list)
    faiss_index: Optional[Any] = None
    chunk_embeddings: Optional[Any] = None
    document_centroids: dict = field(default_factory=dict)
