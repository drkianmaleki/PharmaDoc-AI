from __future__ import annotations

import re
from collections import Counter

from src.schemas import ChatAnswer, RetrievedChunk, SourceReference


def _keywords(text: str) -> set[str]:
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on",
        "is", "are", "was", "were", "what", "who", "when", "where", "why",
        "how", "does", "do", "did", "this", "that", "from", "it", "as", "by",
        "be", "can", "will", "would", "should", "could", "about", "into",
    }
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower())
    return {word for word in words if word not in stopwords and len(word) > 2}


def synthesize_answer(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    max_sentences: int = 5,
) -> ChatAnswer:
    """Create a grounded answer from retrieved chunks.

    This is intentionally not a hallucinating LLM. It extracts and combines
    the most query-relevant sentences from retrieved context and returns sources.
    """

    if not retrieved_chunks:
        return ChatAnswer(
            answer="I could not find relevant information in the uploaded documents.",
            sources=[],
            retrieved_context=[],
        )

    query_terms = _keywords(query)
    candidates: list[tuple[int, float, str]] = []

    for chunk in retrieved_chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            overlap = len(query_terms & _keywords(sentence))
            if overlap > 0:
                candidates.append((overlap, chunk.score, sentence))

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected_sentences = []
        seen = set()

        for _, _, sentence in candidates:
            normalized = sentence.lower()
            if normalized not in seen:
                selected_sentences.append(sentence)
                seen.add(normalized)

            if len(selected_sentences) >= max_sentences:
                break

        answer = " ".join(selected_sentences)
    else:
        answer = retrieved_chunks[0].text[:900].strip()

    source_counts = Counter(chunk.filename for chunk in retrieved_chunks)

    return ChatAnswer(
        answer=answer,
        sources=[
            SourceReference(filename=filename, retrieved_chunks=count)
            for filename, count in source_counts.items()
        ],
        retrieved_context=retrieved_chunks,
    )
