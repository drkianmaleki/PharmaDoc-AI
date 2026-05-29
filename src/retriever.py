from __future__ import annotations

import numpy as np

from src.embeddings import EmbeddingModel
from src.schemas import RetrievedChunk, TextChunk


class InMemoryRetriever:
    """In-memory semantic retriever for uploaded documents.

    This is intentionally simple and appropriate for a Streamlit demo:
    uploaded files stay in the current session and are not persisted.
    """

    def __init__(self, chunks: list[TextChunk], embedding_model: EmbeddingModel):
        if not chunks:
            raise ValueError("At least one text chunk is required.")

        self.chunks = chunks
        self.embedding_model = embedding_model
        self.chunk_embeddings = embedding_model.embed([chunk.text for chunk in chunks])

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = self.embedding_model.embed([query])[0]

        # Embeddings are normalized, so dot product is cosine similarity.
        scores = self.chunk_embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]

        retrieved: list[RetrievedChunk] = []

        for index in top_indices:
            chunk = self.chunks[int(index)]
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    filename=chunk.filename,
                    text=chunk.text,
                    score=float(scores[int(index)]),
                )
            )

        return retrieved
