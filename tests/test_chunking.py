from __future__ import annotations

import pytest

from src.chunking import chunk_documents, chunk_text, clean_text
from src.schemas import ChunkingStrategy, UploadedDocument


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------


def test_clean_text_collapses_whitespace():
    assert clean_text("hello   \t world\n") == "hello world"


def test_clean_text_removes_null_bytes():
    assert "\x00" not in clean_text("hel\x00lo")


def test_clean_text_removes_control_characters():
    # Control characters are replaced with a space, then collapsed — so words
    # that were only separated by a control char gain a single space between them.
    assert clean_text("hel\x01lo\x1fworld") == "hel lo world"
    assert "\x01" not in clean_text("hel\x01lo")
    assert "\x1f" not in clean_text("foo\x1fbar")


def test_clean_text_empty_input():
    assert clean_text("   ") == ""


# ---------------------------------------------------------------------------
# chunk_text — validation
# ---------------------------------------------------------------------------


def test_chunk_text_raises_on_non_positive_chunk_size():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("hello", chunk_size=0)


def test_chunk_text_raises_on_negative_overlap():
    with pytest.raises(ValueError, match="overlap cannot be negative"):
        chunk_text("hello", chunk_size=100, overlap=-1)


def test_chunk_text_raises_when_overlap_exceeds_chunk_size():
    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        chunk_text("hello", chunk_size=100, overlap=100)


def test_chunk_text_empty_returns_empty():
    assert chunk_text("   ", chunk_size=100, overlap=10) == []


def test_chunk_text_short_text_returns_single_chunk():
    result = chunk_text("short text", chunk_size=500, overlap=50)
    assert result == ["short text"]


# ---------------------------------------------------------------------------
# CHARACTER strategy
# ---------------------------------------------------------------------------


def test_character_produces_multiple_chunks():
    chunks = chunk_text("A" * 1200, chunk_size=500, overlap=100, strategy=ChunkingStrategy.CHARACTER)
    assert len(chunks) >= 2
    assert all(len(c) <= 500 for c in chunks)


def test_character_respects_overlap():
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=500, overlap=100, strategy=ChunkingStrategy.CHARACTER)
    assert len(chunks) >= 2
    # The start of the second chunk should overlap with the end of the first.
    assert chunks[0][-50:] in chunks[1]


# ---------------------------------------------------------------------------
# WORD_BOUNDARY strategy
# ---------------------------------------------------------------------------


def test_word_boundary_does_not_split_words():
    text = "one two three four five six seven eight nine ten " * 30
    chunks = chunk_text(text, chunk_size=100, overlap=20, strategy=ChunkingStrategy.WORD_BOUNDARY)
    for chunk in chunks:
        # No chunk should start or end mid-word (i.e. start/end at space boundary).
        assert not chunk[0].isspace()
        assert not chunk[-1].isspace()


def test_word_boundary_fallback_for_no_spaces():
    # Long run of characters with no spaces falls back to character behaviour.
    chunks = chunk_text("X" * 1200, chunk_size=500, overlap=100, strategy=ChunkingStrategy.WORD_BOUNDARY)
    assert len(chunks) >= 2
    assert all(len(c) <= 500 for c in chunks)


# ---------------------------------------------------------------------------
# SENTENCE strategy
# ---------------------------------------------------------------------------


def test_sentence_keeps_whole_sentences():
    text = "First sentence. Second sentence. Third sentence. " * 20
    chunks = chunk_text(text, chunk_size=200, overlap=50, strategy=ChunkingStrategy.SENTENCE)
    assert len(chunks) >= 2
    for chunk in chunks:
        # Every chunk must end at a sentence boundary (period, !, or ?).
        assert chunk.rstrip()[-1] in ".!?"


def test_sentence_handles_no_sentence_endings():
    text = "no sentence endings here just a long run of words " * 30
    chunks = chunk_text(text, chunk_size=200, overlap=40, strategy=ChunkingStrategy.SENTENCE)
    assert chunks
    assert all(len(c) <= 200 for c in chunks)


# ---------------------------------------------------------------------------
# PARAGRAPH strategy
# ---------------------------------------------------------------------------


def test_paragraph_preserves_paragraph_boundaries():
    paragraph_a = "This is the first paragraph with several words."
    paragraph_b = "This is the second paragraph, also with several words."
    text = f"{paragraph_a}\n\n{paragraph_b}"
    # chunk_size large enough to hold each paragraph individually.
    chunks = chunk_text(text, chunk_size=300, overlap=0, strategy=ChunkingStrategy.PARAGRAPH)
    assert len(chunks) == 1  # both fit in one chunk
    assert paragraph_a in chunks[0]
    assert paragraph_b in chunks[0]


def test_paragraph_splits_large_paragraph_by_word_boundary():
    big_paragraph = "word " * 300  # ~1 500 chars
    chunks = chunk_text(big_paragraph, chunk_size=200, overlap=40, strategy=ChunkingStrategy.PARAGRAPH)
    assert len(chunks) >= 2
    assert all(len(c) <= 200 for c in chunks)


# ---------------------------------------------------------------------------
# chunk_documents
# ---------------------------------------------------------------------------


def test_chunk_documents_tracks_filename():
    documents = [UploadedDocument(filename="sample.txt", text="hello world " * 100)]
    chunks = chunk_documents(documents, chunk_size=200, overlap=50)
    assert chunks
    assert all(c.filename == "sample.txt" for c in chunks)


def test_chunk_documents_chunk_index_is_sequential():
    documents = [UploadedDocument(filename="doc.txt", text="sentence one. sentence two. " * 40)]
    chunks = chunk_documents(documents, chunk_size=200, overlap=40)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_documents_global_id_is_unique():
    documents = [
        UploadedDocument(filename="a.txt", text="alpha " * 200),
        UploadedDocument(filename="b.txt", text="beta " * 200),
    ]
    chunks = chunk_documents(documents, chunk_size=200, overlap=40)
    ids = [c.chunk_id for c in chunks]
    assert ids == list(range(len(chunks)))


def test_chunk_documents_chunk_index_resets_per_document():
    documents = [
        UploadedDocument(filename="a.txt", text="word " * 200),
        UploadedDocument(filename="b.txt", text="word " * 200),
    ]
    chunks = chunk_documents(documents, chunk_size=200, overlap=40)
    a_indices = [c.chunk_index for c in chunks if c.filename == "a.txt"]
    b_indices = [c.chunk_index for c in chunks if c.filename == "b.txt"]
    assert a_indices == list(range(len(a_indices)))
    assert b_indices == list(range(len(b_indices)))


def test_chunk_documents_respects_strategy():
    text = "First sentence. Second sentence. Third sentence. " * 20
    documents = [UploadedDocument(filename="test.txt", text=text)]
    chunks = chunk_documents(
        documents, chunk_size=200, overlap=50, strategy=ChunkingStrategy.SENTENCE
    )
    assert chunks
    for chunk in chunks:
        assert chunk.text.rstrip()[-1] in ".!?"
