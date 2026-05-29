from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class ChunkingStrategy(str, Enum):
    """Text chunking strategy used during document indexing."""

    CHARACTER = "character"
    WORD_BOUNDARY = "word_boundary"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"

    @property
    def label(self) -> str:
        return {
            ChunkingStrategy.CHARACTER: "Character — fastest, may split mid-word",
            ChunkingStrategy.WORD_BOUNDARY: "Word Boundary — snaps to word edges (default)",
            ChunkingStrategy.SENTENCE: "Sentence — groups whole sentences, ideal for Q&A",
            ChunkingStrategy.PARAGRAPH: "Paragraph — groups whole paragraphs, ideal for structured docs",
        }[self]


class UploadedDocument(BaseModel):
    """A text document uploaded through the Streamlit interface."""

    filename: str = Field(min_length=1)
    text: str = Field(min_length=1)


class TextChunk(BaseModel):
    """A chunk of text created from an uploaded document."""

    chunk_id: int = Field(ge=0)
    chunk_index: int = Field(ge=0)
    filename: str = Field(min_length=1)
    text: str = Field(min_length=1)


class RetrievedChunk(BaseModel):
    """A retrieved chunk with similarity score."""

    chunk_id: int = Field(ge=0)
    filename: str = Field(min_length=1)
    text: str = Field(min_length=1)
    score: float = Field(ge=0.0)


class SourceReference(BaseModel):
    """Summary of sources used in an answer."""

    filename: str = Field(min_length=1)
    retrieved_chunks: int = Field(ge=1)


class ChatAnswer(BaseModel):
    """Final chatbot response."""

    answer: str = Field(min_length=1)
    sources: list[SourceReference] = Field(default_factory=list)
    retrieved_context: list[RetrievedChunk] = Field(default_factory=list)


class RAGSettings(BaseModel):
    """Validated RAG settings controlled by the app sidebar."""

    chunk_size: int = Field(default=900, ge=200, le=3000)
    chunk_overlap: int = Field(default=180, ge=0, le=1000)
    top_k: int = Field(default=4, ge=1, le=10)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.WORD_BOUNDARY

    @model_validator(mode="after")
    def _check_overlap(self) -> RAGSettings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self
