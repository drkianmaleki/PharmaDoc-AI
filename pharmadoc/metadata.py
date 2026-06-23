"""
Content-item and document-record schemas for PharmaDoc AI.

Factory functions for content items (the core unit passed through ingestion
and retrieval) and document registry records. Includes rule-based
document-type classification.
"""

import fitz
import re
import uuid



def create_content_item(
    document_id,
    file_name,
    doc_type,
    page_start,
    page_end,
    content_type,
    text_for_embedding,
    text_for_llm,
    extraction_method,
    bbox=None,
    confidence=None,
    **extra_metadata,
):
    source_id = (
        f"{document_id}_p{page_start}_{content_type}_"
        f"{uuid.uuid4().hex[:8]}"
    )

    item = {
        "source_id": source_id,
        "document_id": document_id,
        "file": file_name,
        "doc_type": doc_type,
        "page_start": page_start,
        "page_end": page_end,
        "content_type": content_type,
        "text_for_embedding": str(text_for_embedding or "").strip(),
        "text_for_llm": str(text_for_llm or "").strip(),
        "bbox": bbox,
        "extraction_method": extraction_method,
        "confidence": confidence,
    }

    item.update(extra_metadata)
    return item


# Purpose: Track each uploaded PDF at the document level before extraction.

def create_document_record(document_id, file_name, doc_type="Unknown", num_pages=0):
    return {
        "document_id": document_id,
        "file": file_name,
        "doc_type": doc_type,
        "num_pages": num_pages,
        "num_text_chunks": 0,
        "num_tables": 0,
        "num_ocr_regions": 0,
        "num_plot_tables": 0
    }



def detect_doc_type(text_sample, file_name=""):
    """
    Rule-based document-type classifier for metadata tags and UI filters.

    Priority matters for mixed-document PDFs. Broad regulatory/supporting
    collections are checked first, followed by specification evidence, then
    narrower certificate and record types.
    """
    text = f"{file_name}\n{text_sample or ''}".lower()
    text = re.sub(r"\s+", " ", text)

    if (
        "clinical trial summary report" in text
        or "phase iii clinical trial" in text
        or "phase 3 clinical trial" in text
        or (
            "clinical trial" in text
            and "methodology" in text
            and "efficacy" in text
        )
    ):
        return "Clinical Trial Report"

    if (
        "supporting documentation" in text
        or "regulatory supporting" in text
        or "bse/tse statement" in text
        or "origin of milk-statement" in text
        or "pharmaceutical lactose" in text
        or "compliance statement" in text
    ):
        return "Regulatory / Supporting Documentation"

    # Check specification evidence before narrow certificate labels.
    # The Cytiva test PDF is a mixed compilation that contains Certificates
    # of Quality as well as packaging/material specifications. Its intended
    # corpus-level label is therefore Specification.
    specification_markers = (
        "packaging component specification",
        "material description sheet",
        "product specification",
        "component specification",
        "materials of construction",
        "physical properties",
        "operating temperature",
        "operating pressure",
    )

    strong_specification_count = sum(
        marker in text
        for marker in specification_markers
    )

    if (
        strong_specification_count >= 2
        or "packaging component specification" in text
        or "material description sheet" in text
        or "product specification" in text
        or (
            ("specification" in text or "specifications" in text)
            and (
                "operating temperature" in text
                or "materials of construction" in text
                or "component material" in text
            )
        )
    ):
        return "Specification"

    if "certificate of analysis" in text or re.search(r"\bcoa\b", text):
        return "Certificate of Analysis"

    if "certificate of quality" in text:
        return "Certificate of Quality"

    if "safety data sheet" in text or re.search(r"\bsds\b", text):
        return "Safety Data Sheet"

    if "supplier qualification record" in text:
        return "Supplier Qualification Record"

    if "quality agreement" in text:
        return "Quality Agreement"

    if (
        "declaration regarding" in text
        or "declaration" in text
        or "statement" in text
        or "to whom it may concern" in text
    ):
        return "Declaration / Statement"

    return "Unknown"



def extract_text_sample(pdf_path, max_pages=2):
    """
    Extract a small text sample from the first few pages of a PDF.

    This is used for:
    1. document type detection
    2. quick document preview
    3. avoiding full extraction just to classify the document
    """

    sample_text = ""

    try:
        doc = fitz.open(pdf_path)

        for page_index in range(min(max_pages, len(doc))):
            page = doc[page_index]
            sample_text += page.get_text() + "\n"

        doc.close()

    except Exception as e:
        print(f"Could not extract text sample from {pdf_path}: {e}")

    return sample_text.strip()

