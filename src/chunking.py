from __future__ import annotations

import re

from src.schemas import ChunkingStrategy, TextChunk, UploadedDocument

# Sentence boundary: a sentence-ending punctuation mark followed by whitespace.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
# Two or more consecutive newlines mark a paragraph boundary.
_PARAGRAPH_SEP_RE = re.compile(r"\n\s*\n")
# ASCII control characters except tab (\x09) and newline (\x0a).
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Collapse all whitespace to single spaces and strip control characters."""
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_preserve_structure(text: str) -> str:
    """Light normalisation that keeps newlines so structure-aware strategies
    can detect sentence and paragraph boundaries."""
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)       # collapse horizontal whitespace only
    text = re.sub(r"\n{3,}", "\n\n", text)     # cap consecutive newlines at two
    return text.strip()


# ---------------------------------------------------------------------------
# Internal splitting helpers
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_SEP_RE.split(text) if p.strip()]


def _chunks_from_units(
    units: list[str],
    chunk_size: int,
    overlap: int,
    joiner: str,
) -> list[str]:
    """Group text units (sentences or paragraphs) into overlapping chunks.

    Units that individually exceed *chunk_size* should be pre-split by the
    caller so every element in *units* fits within the budget.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for unit in units:
        unit_len = len(unit)
        join_cost = len(joiner) if current else 0

        if current and current_size + join_cost + unit_len > chunk_size:
            chunks.append(joiner.join(current))

            # Retain the most recent units that fit within the overlap window.
            overlap_units: list[str] = []
            overlap_size = 0
            for prev in reversed(current):
                cost = len(prev) + (len(joiner) if overlap_units else 0)
                if overlap_size + cost > overlap:
                    break
                overlap_units.insert(0, prev)
                overlap_size += cost

            current = overlap_units
            current_size = overlap_size

        current.append(unit)
        current_size += (len(joiner) if len(current) > 1 else 0) + unit_len

    if current:
        chunks.append(joiner.join(current))

    return chunks


# ---------------------------------------------------------------------------
# Core chunking strategies
# ---------------------------------------------------------------------------


def _chunk_character(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _chunk_word_boundary(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Like character chunking but snaps each boundary to the nearest space
    so words are never split across chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _chunk_sentence(text: str, chunk_size: int, overlap: int) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    # Sentences longer than chunk_size are subdivided by word boundary.
    units: list[str] = []
    for sentence in sentences:
        if len(sentence) > chunk_size:
            units.extend(_chunk_word_boundary(sentence, chunk_size, overlap))
        else:
            units.append(sentence)
    return _chunks_from_units(units, chunk_size, overlap, joiner=" ")


def _chunk_paragraph(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []
    # Paragraphs larger than chunk_size are subdivided by word boundary.
    units: list[str] = []
    for para in paragraphs:
        if len(para) > chunk_size:
            units.extend(_chunk_word_boundary(para, chunk_size, overlap))
        else:
            units.append(para)
    return _chunks_from_units(units, chunk_size, overlap, joiner="\n\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_STRATEGY_DISPATCH = {
    ChunkingStrategy.CHARACTER: _chunk_character,
    ChunkingStrategy.WORD_BOUNDARY: _chunk_word_boundary,
    ChunkingStrategy.SENTENCE: _chunk_sentence,
    ChunkingStrategy.PARAGRAPH: _chunk_paragraph,
}


def chunk_text(
    text: str,
    chunk_size: int = 900,
    overlap: int = 180,
    strategy: ChunkingStrategy = ChunkingStrategy.WORD_BOUNDARY,
) -> list[str]:
    """Split *text* into overlapping chunks using *strategy*.

    Parameters
    ----------
    text:
        Raw document text.
    chunk_size:
        Maximum number of characters per chunk.
    overlap:
        Number of characters (or approximate unit-based equivalent) shared
        between consecutive chunks to preserve local context.
    strategy:
        ``CHARACTER``     — exact character slicing; fastest but may split mid-word.
        ``WORD_BOUNDARY`` — snaps boundaries to the nearest space; no broken words.
        ``SENTENCE``      — groups complete sentences; best for factual / Q&A content.
        ``PARAGRAPH``     — groups complete paragraphs; best for structured documents.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    structure_aware = strategy in (ChunkingStrategy.SENTENCE, ChunkingStrategy.PARAGRAPH)
    preprocessed = _clean_preserve_structure(text) if structure_aware else clean_text(text)

    if not preprocessed:
        return []

    fn = _STRATEGY_DISPATCH[strategy]
    return fn(preprocessed, chunk_size, overlap)


def chunk_documents(
    documents: list[UploadedDocument],
    chunk_size: int = 900,
    overlap: int = 180,
    strategy: ChunkingStrategy = ChunkingStrategy.WORD_BOUNDARY,
) -> list[TextChunk]:
    """Create validated :class:`TextChunk` objects from a list of uploaded documents.

    Each chunk carries a document-scoped *chunk_index* (its position within the
    source file) in addition to a session-scoped *chunk_id*.
    """
    chunks: list[TextChunk] = []
    global_id = 0

    for document in documents:
        doc_chunks = chunk_text(
            document.text,
            chunk_size=chunk_size,
            overlap=overlap,
            strategy=strategy,
        )
        for chunk_index, text in enumerate(doc_chunks):
            chunks.append(
                TextChunk(
                    chunk_id=global_id,
                    chunk_index=chunk_index,
                    filename=document.filename,
                    text=text,
                )
            )
            global_id += 1

    return chunks
