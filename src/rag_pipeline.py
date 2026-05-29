from __future__ import annotations

from src.chunking import chunk_documents
from src.embeddings import EmbeddingModel
from src.generator import synthesize_answer
from src.retriever import InMemoryRetriever
from src.schemas import ChatAnswer, RAGSettings, UploadedDocument


class StreamlitRAGPipeline:
    """RAG pipeline for uploaded Streamlit text files."""

    def __init__(self, documents: list[UploadedDocument], settings: RAGSettings):
        self.settings = settings
        self.documents = documents
        self.chunks = chunk_documents(
            documents=documents,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            strategy=settings.chunking_strategy,
        )
        self.embedding_model = EmbeddingModel(settings.embedding_model)
        self.retriever = InMemoryRetriever(self.chunks, self.embedding_model)

    def ask(self, query: str) -> ChatAnswer:
        retrieved = self.retriever.retrieve(query=query, top_k=self.settings.top_k)
        return synthesize_answer(query=query, retrieved_chunks=retrieved)
