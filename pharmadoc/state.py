"""
pharmadoc/state.py

NEW FILE - not from the notebook. Defines RAGState, an explicit container
for the document-processing session state that the original notebook held
as bare module-level globals (see the placeholder values at the end of
notebook CELL 03, preserved verbatim in config.py). Default values below
mirror those originals exactly. This is the user-approved replacement for
the `global` statements that do not survive a notebook -> multi-file
package conversion.
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
