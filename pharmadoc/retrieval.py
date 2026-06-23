"""
Embeddings, FAISS indexing, and hybrid retrieval for PharmaDoc AI.

Loads the sentence-transformer embedding model, builds and queries the FAISS
index, applies metadata filters, and re-ranks results using hybrid
keyword and semantic scoring.
"""

from sentence_transformers import SentenceTransformer
import faiss

from .config import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME, SUPPORTED_CONTENT_TYPES



def load_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print("Embedding model loaded successfully.")
    return model


embedding_model = load_embedding_model()



def build_faiss_index(content_items, embedding_model):
    """Create a normalized inner-product FAISS index."""
    if not content_items:
        raise ValueError("Cannot build a FAISS index because no content items were created.")

    texts = [
        item["text_for_embedding"].strip()
        for item in content_items
        if item.get("text_for_embedding", "").strip()
    ]

    if len(texts) != len(content_items):
        raise ValueError("Every content item must contain non-empty text_for_embedding.")

    print(f"Creating embeddings for {len(texts)} content items...")

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    if index.ntotal != len(content_items):
        raise RuntimeError("FAISS index and metadata list are misaligned.")

    print(f"FAISS index built with {index.ntotal} vectors.")
    return index, embeddings



def retrieve_relevant_chunks(
    question,
    faiss_index,
    content_items,
    embedding_model,
    top_k=DEFAULT_TOP_K,
    document_filter="All documents",
    doc_type_filter="All types",
    content_type_filter=None,
):
    """Retrieve semantically similar items and then apply metadata filters."""
    if faiss_index is None or not content_items:
        return []

    if content_type_filter is None:
        content_type_filter = SUPPORTED_CONTENT_TYPES

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # Search the complete small/medium index so filtering cannot hide a valid result.
    search_k = len(content_items)
    scores, indices = faiss_index.search(question_embedding, search_k)

    retrieved = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        item = content_items[int(idx)]

        if document_filter != "All documents" and item["file"] != document_filter:
            continue
        if doc_type_filter != "All types" and item["doc_type"] != doc_type_filter:
            continue
        if item["content_type"] not in content_type_filter:
            continue

        result = dict(item)
        result["score"] = float(score)
        retrieved.append(result)

        if len(retrieved) >= int(top_k):
            break

    return retrieved



import re
from difflib import SequenceMatcher


RERANK_CONFIG = {
    "candidate_multiplier": 6,
    "minimum_candidates": 20,

    # Lexical bonuses added to the FAISS similarity score
    "exact_phrase_bonus": 0.12,
    "token_overlap_bonus": 0.16,
    "identifier_bonus": 0.10,
    "table_intent_bonus": 0.05,
    "key_value_intent_bonus": 0.03,

    # Deduplication
    "duplicate_similarity_threshold": 0.92,
}


QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do",
    "does", "for", "from", "how", "in", "is", "it", "of",
    "on", "or", "the", "to", "was", "what", "when", "where",
    "which", "who", "with"
}


TABLE_INTENT_TERMS = {
    "amount", "article", "audit", "batch", "compatible",
    "condition", "date", "description", "expiration",
    "field", "limit", "lot", "material", "method",
    "number", "part", "pressure", "result", "specification",
    "status", "temperature", "value", "weight"
}


KEY_VALUE_INTENT_TERMS = {
    "address", "audit", "date", "expiration", "lot",
    "manufacturer", "number", "pressure", "product",
    "reference", "revision", "status", "temperature",
    "weight"
}


def normalize_retrieval_text(text):
    """Normalize text for lexical comparison."""
    text = str(text or "").lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_retrieval_text(text):
    """
    Return meaningful query or document tokens.

    Symbols important to scientific values are retained where possible.
    """
    normalized = normalize_retrieval_text(text)

    tokens = re.findall(
        r"[a-z0-9]+(?:[-./][a-z0-9]+)*|[+%-]",
        normalized
    )

    return [
        token
        for token in tokens
        if token not in QUERY_STOPWORDS
        and len(token) > 1
    ]


def extract_identifiers(text):
    """
    Extract exact identifiers and numeric expressions.

    Examples:
    - 29477427
    - 99.8%
    - 5 MPa
    - 20210712
    - 25-40
    """
    normalized = normalize_retrieval_text(text)

    patterns = [
        r"\b\d{4,}\b",
        r"\b\d+(?:\.\d+)?\s*%",
        r"\b\d+(?:\.\d+)?\s*(?:mpa|kpa|kgy|mg|kg|g|ml|l|°c)\b",
        r"\b\d+(?:\.\d+)?\s*(?:-|to)\s*\+?\d+(?:\.\d+)?\s*°c\b",
        r"\b[a-z]+-\d+[a-z0-9-]*\b",
    ]

    identifiers = set()

    for pattern in patterns:
        identifiers.update(
            re.findall(
                pattern,
                normalized,
                flags=re.IGNORECASE
            )
        )

    return identifiers


def build_query_phrases(question):
    """
    Build useful multiword phrases from the query.

    Bigrams and trigrams help phrases such as:
    - operating pressure
    - part number
    - calf rennet
    """
    tokens = tokenize_retrieval_text(question)
    phrases = set()

    for size in (2, 3):
        for index in range(len(tokens) - size + 1):
            phrase = " ".join(
                tokens[index:index + size]
            )
            phrases.add(phrase)

    return phrases


def question_prefers_tables(question):
    """Return True when the query is likely asking for structured values."""
    tokens = set(tokenize_retrieval_text(question))
    return bool(tokens & TABLE_INTENT_TERMS)


def question_prefers_key_values(question):
    """Return True when a key–value record is especially appropriate."""
    tokens = set(tokenize_retrieval_text(question))
    return bool(tokens & KEY_VALUE_INTENT_TERMS)


def calculate_rerank_score(question, item):
    """
    Combine semantic similarity with lightweight lexical evidence.

    The original FAISS cosine similarity remains available as
    semantic_score.
    """
    semantic_score = float(
        item.get("semantic_score", item.get("score", 0.0))
    )

    document_text = normalize_retrieval_text(
        item.get("text_for_embedding", "")
    )

    query_tokens = set(
        tokenize_retrieval_text(question)
    )

    document_tokens = set(
        tokenize_retrieval_text(document_text)
    )

    score = semantic_score

    # Token-overlap bonus
    if query_tokens:
        overlap_ratio = (
            len(query_tokens & document_tokens)
            / len(query_tokens)
        )

        score += (
            RERANK_CONFIG["token_overlap_bonus"]
            * overlap_ratio
        )

    # Exact multiword phrase bonus
    matched_phrases = [
        phrase
        for phrase in build_query_phrases(question)
        if phrase in document_text
    ]

    if matched_phrases:
        score += RERANK_CONFIG["exact_phrase_bonus"]

    # Exact identifier and numeric-value bonus
    query_identifiers = extract_identifiers(question)
    document_identifiers = extract_identifiers(document_text)

    if query_identifiers:
        identifier_overlap = (
            len(query_identifiers & document_identifiers)
            / len(query_identifiers)
        )

        score += (
            RERANK_CONFIG["identifier_bonus"]
            * identifier_overlap
        )

    # Structured-content preference
    if (
        question_prefers_tables(question)
        and item.get("content_type") == "table"
    ):
        score += RERANK_CONFIG["table_intent_bonus"]

    if (
        question_prefers_key_values(question)
        and item.get("table_structure_type") == "key_value"
    ):
        score += RERANK_CONFIG["key_value_intent_bonus"]

    return float(score)


def retrieval_items_are_duplicates(first, second):
    """
    Detect near-duplicate retrieval results.

    Duplicates are restricted to the same file and page so unrelated
    documents cannot suppress one another.
    """
    if first.get("file") != second.get("file"):
        return False

    if first.get("page_start") != second.get("page_start"):
        return False

    first_text = normalize_retrieval_text(
        first.get("text_for_embedding", "")
    )

    second_text = normalize_retrieval_text(
        second.get("text_for_embedding", "")
    )

    if not first_text or not second_text:
        return False

    if first_text == second_text:
        return True

    similarity = SequenceMatcher(
        None,
        first_text,
        second_text
    ).ratio()

    return (
        similarity
        >= RERANK_CONFIG["duplicate_similarity_threshold"]
    )


def deduplicate_retrieval_results(results, top_k):
    """
    Keep the highest-ranked result from each near-duplicate group.
    """
    deduplicated = []

    for candidate in results:
        duplicate_found = any(
            retrieval_items_are_duplicates(
                candidate,
                existing
            )
            for existing in deduplicated
        )

        if duplicate_found:
            continue

        deduplicated.append(candidate)

        if len(deduplicated) >= int(top_k):
            break

    return deduplicated


def retrieve_relevant_chunks_hybrid(
    question,
    faiss_index,
    content_items,
    embedding_model,
    top_k=DEFAULT_TOP_K,
    document_filter="All documents",
    doc_type_filter="All types",
    content_type_filter=None,
):
    """
    Retrieve a larger FAISS candidate pool, apply metadata filters,
    rerank lexically, and remove near duplicates.
    """
    if faiss_index is None or not content_items:
        return []

    if content_type_filter is None:
        content_type_filter = SUPPORTED_CONTENT_TYPES

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    requested_top_k = int(top_k)

    candidate_k = max(
        requested_top_k
        * RERANK_CONFIG["candidate_multiplier"],
        RERANK_CONFIG["minimum_candidates"],
    )

    candidate_k = min(
        candidate_k,
        len(content_items)
    )

    scores, indices = faiss_index.search(
        question_embedding,
        candidate_k
    )

    candidates = []

    for semantic_score, item_index in zip(
        scores[0],
        indices[0]
    ):
        if item_index < 0:
            continue

        item = content_items[int(item_index)]

        if (
            document_filter != "All documents"
            and item.get("file") != document_filter
        ):
            continue

        if (
            doc_type_filter != "All types"
            and item.get("doc_type") != doc_type_filter
        ):
            continue

        if item.get("content_type") not in content_type_filter:
            continue

        result = dict(item)

        result["semantic_score"] = float(
            semantic_score
        )

        result["rerank_score"] = calculate_rerank_score(
            question,
            result
        )

        # The public score controls ordering and source display.
        result["score"] = result["rerank_score"]

        candidates.append(result)

    candidates.sort(
        key=lambda result: result["rerank_score"],
        reverse=True
    )

    return deduplicate_retrieval_results(
        candidates,
        requested_top_k
    )

