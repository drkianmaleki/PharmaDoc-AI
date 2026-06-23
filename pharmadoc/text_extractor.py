"""
pharmadoc/text_extractor.py

Section 3 - Digital text extraction and chunking
Source notebook cells: [11, 12]

Verbatim conversion: the code below this header is copied directly from
the notebook's cell source (mechanical extraction, not retyped). Only this
docstring and the import lines immediately below are new.
"""

# --- external imports (used by this file's verbatim code) ---
import fitz

# --- cross-module imports (this package's own files) ---
from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .metadata import create_content_item

# ===== NOTEBOOK CELLS [11, 12] (verbatim) =====

#@title CELL 08 — Extract digital text content objects

def extract_digital_text_items(pdf_path, document_record):
    """
    Extract digital text from each page of one PDF and convert it into
    unified content items.

    This handles selectable/searchable PDF text using PyMuPDF.
    It does not handle OCR, tables, or plots yet.
    """

    content_items = []

    document_id = document_record["document_id"]
    file_name = document_record["file"]
    doc_type = document_record["doc_type"]

    try:
        doc = fitz.open(pdf_path)

        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            page_text = page.get_text().strip()

            if not page_text:
                continue

            text_for_embedding = page_text

            text_for_llm = f"""
[TEXT CHUNK]
Source: {file_name}, page {page_number}
Document type: {doc_type}

{page_text}
""".strip()

            item = create_content_item(
                document_id=document_id,
                file_name=file_name,
                doc_type=doc_type,
                page_start=page_number,
                page_end=page_number,
                content_type="text",
                text_for_embedding=text_for_embedding,
                text_for_llm=text_for_llm,
                extraction_method="pymupdf_text",
                bbox=None,
                confidence=None
            )

            content_items.append(item)

        doc.close()

    except Exception as e:
        print(f"Could not extract digital text from {pdf_path}: {e}")

    return content_items


#@title CELL 09 — Chunk long text content items

def split_text_with_overlap(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split long text into overlapping chunks.

    This keeps retrieval focused while preserving some context between chunks.
    """

    if not text:
        return []

    text = text.strip()

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap

        if start < 0:
            start = 0

        if start >= len(text):
            break

    return chunks

