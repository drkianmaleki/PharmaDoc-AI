"""
pharmadoc/app.py

Sections 11-12 - Gradio application, theme, and launch
Source notebook cells: [44, 45, 46, 47, 49]

Verbatim conversion: the code below this header is copied directly from
the notebook's cell source (mechanical extraction, not retyped). Only this
docstring and the import lines immediately below are new.

NOTE: cells 45, 46, and 49 contain two user-approved patches:
(1) global statements replaced with an explicit RAGState object
threaded through Gradio's gr.State, and (2) the launch() call
guarded behind `if __name__ == "__main__":` so importing this
module does not start a live server as a side effect (share=True
left untouched). All other logic in this file, and every other
file in this package, is unmodified verbatim.
"""

# --- external imports (used by this file's verbatim code) ---
import gradio as gr

# --- cross-module imports (this package's own files) ---
from .config import CHUNK_OVERLAP, CHUNK_SIZE, DEFAULT_MODEL_CHOICE, DEFAULT_TOP_K, MODEL_CATALOG, SUPPORTED_CONTENT_TYPES
from .evaluation import answer_question_with_rag
from .ingestion import build_document_registry, extract_non_pdf_items, extract_plots_from_pdf, extract_selective_ocr_from_pdf
from .metadata import create_content_item
from .persistence import build_document_centroids, deduplicate_content_items, save_rag_artifacts, suggest_scaling_strategy
from .retrieval import build_faiss_index
from .tables import extract_structured_table_items
from .text_extractor import extract_digital_text_items, split_text_with_overlap
from .retrieval import embedding_model
from .state import RAGState

# ===== NOTEBOOK CELLS [44, 45, 46, 47, 49] (verbatim) =====

#@title CELL 29 — UI summaries and filter choices

from collections import defaultdict

def get_document_filter_choices(document_registry):
    return ["All documents"] + [record["file"] for record in document_registry.values()]


def get_doc_type_filter_choices(document_registry):
    values = sorted({record.get("doc_type", "Unknown") for record in document_registry.values()})
    return ["All types"] + values


def get_content_type_filter_choices(content_items):
    values = sorted({item.get("content_type", "unknown") for item in content_items or []})
    return ["All content"] + values


def format_document_registry_summary(document_registry):
    if not document_registry:
        return "No documents processed yet."

    totals = defaultdict(int)
    lines = [f"Documents processed: {len(document_registry)}", ""]

    for record in document_registry.values():
        totals["pages"] += int(record.get("num_pages", 0))
        totals["text"] += int(record.get("num_text_chunks", 0))
        totals["tables"] += int(record.get("num_tables", 0))
        totals["ocr"] += int(record.get("num_ocr_regions", 0))
        totals["ocr_tables"] += int(record.get("num_ocr_tables", 0))
        totals["plots"] += int(record.get("num_plot_tables", 0))

        lines.extend([
            f"- {record['file']}",
            f"  - ID: {record['document_id']}",
            f"  - Kind: {record.get('file_kind', 'pdf')}",
            f"  - Type: {record.get('doc_type', 'Unknown')}",
            f"  - Pages/sheets: {record.get('num_pages', 0)}",
            f"  - Text chunks: {record.get('num_text_chunks', 0)}",
            f"  - Digital tables: {record.get('num_tables', 0)}",
            f"  - OCR text items: {record.get('num_ocr_regions', 0)}",
            f"  - OCR tables: {record.get('num_ocr_tables', 0)}",
            f"  - Plot tables: {record.get('num_plot_tables', 0)}",
            "",
        ])

    lines.extend([
        "Overall totals:",
        f"- Pages/sheets: {totals['pages']}",
        f"- Text chunks: {totals['text']}",
        f"- Digital tables: {totals['tables']}",
        f"- OCR text items: {totals['ocr']}",
        f"- OCR tables: {totals['ocr_tables']}",
        f"- Plot tables: {totals['plots']}",
        f"- Scaling guidance: {suggest_scaling_strategy(len(document_registry))}",
    ])
    return "\n".join(lines)


#@title CELL 30 — Unified multi-format processing pipeline

def normalize_uploaded_paths(uploaded_files):
    paths = []
    for uploaded_file in uploaded_files or []:
        path = uploaded_file if isinstance(uploaded_file, str) else getattr(uploaded_file, "name", None)
        if path:
            paths.append(str(path))
    return paths


def chunk_textual_items(content_items):
    chunked = []
    for item in content_items:
        if item.get("content_type") not in {"text", "ocr_text"}:
            chunked.append(item)
            continue

        chunks = split_text_with_overlap(
            item.get("text_for_embedding", ""),
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        for chunk_index, chunk_text in enumerate(chunks, start=1):
            if not chunk_text.strip():
                continue
            label = "OCR TEXT CHUNK" if item.get("content_type") == "ocr_text" else "TEXT CHUNK"
            chunked.append(create_content_item(
                document_id=item["document_id"],
                file_name=item["file"],
                doc_type=item["doc_type"],
                page_start=item["page_start"],
                page_end=item["page_end"],
                content_type=item["content_type"],
                text_for_embedding=chunk_text,
                text_for_llm=(
                    f"[{label}]\nSource: {item['file']}, page {item['page_start']}\n"
                    f"Chunk: {chunk_index} of {len(chunks)}\n\n{chunk_text}"
                ),
                extraction_method=f"{item.get('extraction_method', 'unknown')}_chunked",
                bbox=item.get("bbox"),
                confidence=item.get("confidence"),
                parent_source_id=item.get("source_id"),
                chunk_index=chunk_index,
                num_chunks_from_parent=len(chunks),
            ))
    return chunked


def process_documents(file_paths_input, enable_ocr=True, enable_plot_extraction=True, persist_after_processing=False):
    registry = build_document_registry(file_paths_input)
    page_text_items, table_items = [], []
    ocr_text_items, ocr_table_items, plot_items = [], [], []

    for document_id, record in registry.items():
        path = record["file_path"]
        kind = record["file_kind"]
        try:
            if kind == "pdf":
                page_text_items.extend(extract_digital_text_items(path, record))
                table_items.extend(extract_structured_table_items(path, record))
                if enable_ocr:
                    ocr_text, ocr_tables = extract_selective_ocr_from_pdf(path, record)
                    ocr_text_items.extend(ocr_text)
                    ocr_table_items.extend(ocr_tables)
                if enable_plot_extraction:
                    plot_items.extend(extract_plots_from_pdf(path, record))
            else:
                text, tables, ocr, plots = extract_non_pdf_items(path, record)
                page_text_items.extend(text)
                table_items.extend(tables)
                if enable_ocr:
                    ocr_text_items.extend(ocr)
                if enable_plot_extraction:
                    plot_items.extend(plots)
        except Exception as error:
            record["warnings"].append(f"{type(error).__name__}: {error}")

    chunked_text = chunk_textual_items(page_text_items)
    chunked_ocr = chunk_textual_items(ocr_text_items)
    combined_items = deduplicate_content_items(
        chunked_text + table_items + chunked_ocr + ocr_table_items + plot_items
    )

    if not combined_items:
        raise ValueError("No searchable content was extracted from the uploaded files.")

    index, embeddings = build_faiss_index(combined_items, embedding_model)
    if index.ntotal != len(combined_items):
        raise RuntimeError("FAISS vectors and content metadata are not aligned.")

    for document_id, record in registry.items():
        items = [item for item in combined_items if item["document_id"] == document_id]
        record["num_text_chunks"] = sum(item["content_type"] == "text" for item in items)
        record["num_tables"] = sum(item["content_type"] == "table" for item in items)
        record["num_ocr_regions"] = sum(item["content_type"] == "ocr_text" for item in items)
        record["num_ocr_tables"] = sum(item["content_type"] == "ocr_table" for item in items)
        record["num_plot_tables"] = sum(item["content_type"] == "plot_table" for item in items)

    centroids = build_document_centroids(combined_items, embeddings)
    result = {
        "document_registry": registry,
        "all_content_items": page_text_items,
        "chunked_content_items": chunked_text,
        "all_table_items": table_items + ocr_table_items,
        "all_ocr_items": chunked_ocr,
        "all_plot_items": plot_items,
        "rag_content_items": combined_items,
        "faiss_index": index,
        "chunk_embeddings": embeddings,
        "document_centroids": centroids,
    }
    if persist_after_processing:
        result["persisted_to"] = save_rag_artifacts(
            index=index, content_items=combined_items, registry=registry, embeddings=embeddings
        )
    return result


def process_uploaded_documents_for_gradio(uploaded_files, enable_ocr, enable_plot_extraction, persist_after_processing, state):
    state.file_paths = normalize_uploaded_paths(uploaded_files)
    empty = (
        gr.update(choices=["All documents"], value="All documents"),
        gr.update(choices=["All types"], value="All types"),
        gr.update(choices=["All content"], value="All content"),
    )
    if not state.file_paths:
        return ("Please upload at least one supported file.", "No documents processed yet.", *empty, state)

    try:
        result = process_documents(
            state.file_paths, bool(enable_ocr), bool(enable_plot_extraction), bool(persist_after_processing)
        )
        state.document_registry = result["document_registry"]
        state.all_content_items = result["all_content_items"]
        state.chunked_content_items = result["chunked_content_items"]
        state.all_table_items = result["all_table_items"]
        state.all_ocr_items = result["all_ocr_items"]
        state.all_plot_items = result["all_plot_items"]
        state.rag_content_items = result["rag_content_items"]
        state.faiss_index = result["faiss_index"]
        state.chunk_embeddings = result["chunk_embeddings"]
        state.document_centroids = result["document_centroids"]

        counts = defaultdict(int)
        for item in state.rag_content_items:
            counts[item["content_type"]] += 1
        status = [
            "Processing complete.",
            f"Documents processed: {len(state.document_registry)}",
            f"Text chunks: {counts['text']}",
            f"Digital tables: {counts['table']}",
            f"OCR text chunks: {counts['ocr_text']}",
            f"OCR tables: {counts['ocr_table']}",
            f"Plot-derived tables: {counts['plot_table']}",
            f"Total indexed items: {len(state.rag_content_items)}",
            f"FAISS vectors: {state.faiss_index.ntotal}",
        ]
        if result.get("persisted_to"):
            status.append(f"Artifacts saved to: {result['persisted_to']}")
        warnings = [
            f"{record['file']}: {warning}"
            for record in state.document_registry.values()
            for warning in record.get("warnings", [])
        ]
        if warnings:
            status.append("Warnings:")
            status.extend(f"- {warning}" for warning in warnings)

        return (
            "\n".join(status),
            format_document_registry_summary(state.document_registry),
            gr.update(choices=get_document_filter_choices(state.document_registry), value="All documents"),
            gr.update(choices=get_doc_type_filter_choices(state.document_registry), value="All types"),
            gr.update(choices=get_content_type_filter_choices(state.rag_content_items), value="All content"),
            state,
        )
    except Exception as error:
        state.faiss_index = None
        state.rag_content_items = []
        return (f"Processing failed: {type(error).__name__}: {error}", "No searchable index was created.", *empty, state)


#@title CELL 31 — Gradio chat handler

import html
import tempfile
from datetime import datetime
from pathlib import Path


def _format_public_answer(result):
    """
    Show answer, sources, confidence, and chunk count by default.
    Internal diagnostics remain available in a collapsed section.
    """
    answer_text = str(result.get("answer", "")).strip()
    sources_text = str(result.get("sources", "No sources retrieved.")).strip()
    confidence = str(result.get("confidence", "Low"))
    chunks_used = int(result.get("chunks_used", result.get("chunk_count", 0)) or 0)

    public_section = (
        f"{answer_text}\n\n---\n\n"
        f"**Sources**\n\n{sources_text}\n\n"
        f"**Confidence:** {confidence}  \n"
        f"**Chunks used:** {chunks_used}"
    )

    evidence = result.get("evidence_check") or {}
    matched_terms = ", ".join(evidence.get("matched_terms", [])) or "none"
    missing_terms = ", ".join(evidence.get("missing_terms", [])) or "none"

    try:
        score_text = f"{float(result.get('top_score')):.4f}"
    except (TypeError, ValueError):
        score_text = "not available"

    technical_lines = [
        f"Answer route: {result.get('answer_route', 'llm_generation')}",
        f"Answer status: {result.get('answer_status', 'generated')}",
        f"Top semantic score: {score_text}",
        f"Evidence terms matched: {matched_terms}",
        f"Evidence terms missing: {missing_terms}",
        f"Model: {result.get('model_choice', 'not reported')}",
    ]

    technical_html = "<br>".join(html.escape(line) for line in technical_lines)

    return (
        public_section
        + "\n\n"
        + "<details class='technical-details'>"
        + "<summary>Technical details</summary>"
        + f"<div>{technical_html}</div>"
        + "</details>"
    )


def gradio_chat_handler(
    message,
    history,
    model_choice,
    top_k,
    document_filter,
    doc_type_filter,
    content_type_filter,
    state,
):
    history = list(history or [])

    if not message or not message.strip():
        return history, ""

    if state.faiss_index is None or not state.rag_content_items:
        history.extend([
            {"role": "user", "content": message},
            {
                "role": "assistant",
                "content": (
                    "Please upload and process at least one supported "
                    "document before asking a question."
                ),
            },
        ])
        return history, ""

    selected_types = (
        SUPPORTED_CONTENT_TYPES
        if content_type_filter == "All content"
        else [content_type_filter]
    )

    try:
        result = answer_question_with_rag(
            question=message,
            model_choice=model_choice,
            faiss_index=state.faiss_index,
            content_items=state.rag_content_items,
            embedding_model=embedding_model,
            top_k=int(top_k),
            document_filter=document_filter,
            doc_type_filter=doc_type_filter,
            content_type_filter=selected_types,
            max_tokens=80,
        )
        final_answer = _format_public_answer(result)

    except Exception as error:
        final_answer = (
            "Answer generation failed. Retry the question or inspect the "
            "technical details below.\n\n"
            "<details class='technical-details'>"
            "<summary>Technical details</summary>"
            f"<div>{html.escape(type(error).__name__ + ': ' + str(error))}</div>"
            "</details>"
        )

    history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": final_answer},
    ])
    return history, ""


def export_chat_history(history):
    """
    Export the current Gradio message history as a readable Markdown file.

    The export includes questions, answers, sources, confidence, chunk counts,
    and the collapsed technical details already stored in each assistant
    message.
    """
    messages = list(history or [])

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_lines = [
        "# PharmaDoc AI — Chat History",
        "",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    if not messages:
        export_lines.extend([
            "_No chat messages were available at the time of export._",
            "",
        ])
    else:
        for message in messages:
            role = str(message.get("role", "unknown")).strip().lower()
            content = str(message.get("content", "")).strip()

            if role == "user":
                heading = "## User"
            elif role == "assistant":
                heading = "## PharmaDoc AI"
            else:
                heading = f"## {role.title()}"

            export_lines.extend([
                heading,
                "",
                content,
                "",
                "---",
                "",
            ])

    export_text = "\n".join(export_lines).strip() + "\n"

    export_directory = Path(tempfile.gettempdir()) / "pharmadoc_ai_exports"
    export_directory.mkdir(parents=True, exist_ok=True)

    export_path = export_directory / (
        f"PharmaDoc_AI_Chat_History_{timestamp}.md"
    )

    export_path.write_text(
        export_text,
        encoding="utf-8",
    )

    return str(export_path)


#@title CELL 32 — Complete green-black Gradio theme

APP_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    spacing_size=gr.themes.sizes.spacing_md,
    font=[
        gr.themes.GoogleFont("Inter"),
        "Arial",
        "sans-serif",
    ],
).set(
    body_background_fill="#050806",
    body_background_fill_dark="#050806",

    block_background_fill="#0b110d",
    block_background_fill_dark="#0b110d",
    block_border_color="#263d2c",
    block_border_color_dark="#263d2c",

    input_background_fill="#0c140f",
    input_background_fill_dark="#0c140f",
    input_border_color="#365840",
    input_border_color_dark="#365840",

    button_primary_background_fill="#16a34a",
    button_primary_background_fill_dark="#16a34a",
    button_primary_background_fill_hover="#22c55e",
    button_primary_background_fill_hover_dark="#22c55e",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",

    button_secondary_background_fill="#172019",
    button_secondary_background_fill_dark="#172019",
    button_secondary_background_fill_hover="#223027",
    button_secondary_background_fill_hover_dark="#223027",
    button_secondary_text_color="#e5e7eb",
    button_secondary_text_color_dark="#e5e7eb",
)


CUSTOM_CSS = """
/* ========================================================================
   Shared color variables
   ======================================================================== */

:root,
.gradio-container {
    --pharma-page: #050806;
    --pharma-panel: #0b110d;
    --pharma-input: #0c140f;
    --pharma-input-soft: #101812;
    --pharma-border: #263d2c;
    --pharma-border-light: #365840;
    --pharma-label: #17351f;
    --pharma-label-hover: #1f492a;
    --pharma-green: #22c55e;
    --pharma-green-light: #86efac;
    --pharma-text: #e5e7eb;
    --pharma-muted: #9caea1;

    /*
       These variables control Gradio's floating component-label tabs.
       They are the main fix for the blue "Chat history" and
       "Upload documents" backgrounds.
    */
    --block-label-background-fill: #17351f !important;
    --block-label-background-fill-dark: #17351f !important;
    --block-label-text-color: #f0fdf4 !important;
    --block-label-text-color-dark: #f0fdf4 !important;
    --block-label-border-color: #2f7841 !important;
    --block-label-border-color-dark: #2f7841 !important;
}


/* ========================================================================
   Global page
   ======================================================================== */

html,
body,
.gradio-container {
    background: var(--pharma-page) !important;
    background-color: var(--pharma-page) !important;
    color: var(--pharma-text) !important;
}

.gradio-container {
    max-width: 1500px !important;
    margin: 0 auto !important;
    padding: 22px !important;
}


/* ========================================================================
   Header
   ======================================================================== */

.app-header {
    padding: 30px 24px;
    margin-bottom: 20px;
    text-align: center;
    border-radius: 20px;

    background:
        radial-gradient(
            circle at top center,
            rgba(34, 197, 94, 0.20),
            transparent 55%
        ),
        linear-gradient(
            135deg,
            #071009,
            #0d1710
        ) !important;

    border: 1px solid rgba(34, 197, 94, 0.45) !important;

    box-shadow:
        0 16px 40px rgba(0, 0, 0, 0.45),
        0 0 30px rgba(34, 197, 94, 0.08);
}

.app-header h1 {
    margin: 0 !important;
    color: #4ade80 !important;
    font-size: 38px !important;
    font-weight: 800 !important;
    letter-spacing: -0.8px !important;
}

.app-subtitle {
    margin-top: 8px;
    color: #cbd5e1 !important;
    font-size: 15px;
}


/* ========================================================================
   Typography
   ======================================================================== */

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container p,
.gradio-container span,
.gradio-container label,
.gradio-container .prose {
    color: var(--pharma-text);
}

.gradio-container h2 {
    color: var(--pharma-green-light) !important;
    font-weight: 700 !important;
}


/* ========================================================================
   FLOATING COMPONENT LABELS
   Fixes the navy tabs shown in the screenshots.
   ======================================================================== */

.gradio-container .block-label,
.gradio-container .label-wrap,
.gradio-container [data-testid="block-label"],
.gradio-container [class*="block-label"],
.gradio-container [class*="label-wrap"] {
    background: var(--pharma-label) !important;
    background-color: var(--pharma-label) !important;
    color: #f0fdf4 !important;

    border-color: #2f7841 !important;
    border-bottom-color: #2f7841 !important;

    box-shadow: none !important;
}


/* Force all text and icons inside floating labels to green-white. */

.gradio-container .block-label *,
.gradio-container .label-wrap *,
.gradio-container [data-testid="block-label"] *,
.gradio-container [class*="block-label"] *,
.gradio-container [class*="label-wrap"] * {
    color: #f0fdf4 !important;
    fill: #86efac !important;
    stroke: #86efac !important;
}


/* Target the two visible problem components directly. */

#pharmadoc-upload .block-label,
#pharmadoc-upload .label-wrap,
#pharmadoc-upload [data-testid="block-label"],
#pharmadoc-upload [class*="block-label"],
#pharmadoc-upload [class*="label-wrap"],

#pharmadoc-chatbot .block-label,
#pharmadoc-chatbot .label-wrap,
#pharmadoc-chatbot [data-testid="block-label"],
#pharmadoc-chatbot [class*="block-label"],
#pharmadoc-chatbot [class*="label-wrap"],

#processing-status .block-label,
#processing-status .label-wrap,
#processing-status [data-testid="block-label"],

#document-summary .block-label,
#document-summary .label-wrap,
#document-summary [data-testid="block-label"],

#question-input .block-label,
#question-input .label-wrap,
#question-input [data-testid="block-label"] {
    background: var(--pharma-label) !important;
    background-color: var(--pharma-label) !important;
    color: #f0fdf4 !important;
    border-color: #2f7841 !important;
}


/*
   Some Gradio versions attach the label styles directly to a top-level
   span. These selectors cover that markup as well.
*/

#pharmadoc-upload > span,
#pharmadoc-chatbot > span,
#processing-status > span,
#document-summary > span,
#question-input > span {
    background-color: var(--pharma-label) !important;
    color: #f0fdf4 !important;
    border-color: #2f7841 !important;
}


/* ========================================================================
   General blocks and wrappers
   ======================================================================== */

.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .wrap,
.gradio-container .container {
    background: var(--pharma-panel) !important;
    background-color: var(--pharma-panel) !important;
    border-color: var(--pharma-border) !important;
}

.gradio-container .block {
    border-radius: 14px !important;
}


/* ========================================================================
   Upload component
   ======================================================================== */

#pharmadoc-upload,
#pharmadoc-upload > div,
#pharmadoc-upload .block,
#pharmadoc-upload .wrap,
#pharmadoc-upload .container,
#pharmadoc-upload .upload-container,
#pharmadoc-upload .file-upload,
#pharmadoc-upload .file-preview,
#pharmadoc-upload .file-preview-holder,
#pharmadoc-upload [data-testid="file"],
#pharmadoc-upload [data-testid="upload"],
#pharmadoc-upload [data-testid="file-upload"] {
    background: var(--pharma-input-soft) !important;
    background-color: var(--pharma-input-soft) !important;
    color: var(--pharma-text) !important;
    border-color: var(--pharma-border-light) !important;
}

#pharmadoc-upload .upload-container,
#pharmadoc-upload [data-testid="upload"] {
    border: 1px dashed var(--pharma-border-light) !important;
    border-radius: 12px !important;
}

#pharmadoc-upload .upload-container:hover,
#pharmadoc-upload [data-testid="upload"]:hover {
    background: #132018 !important;
    border-color: var(--pharma-green) !important;
}

#pharmadoc-upload button {
    background: var(--pharma-label) !important;
    color: #ffffff !important;
    border-color: #2f7841 !important;
}

#pharmadoc-upload svg {
    color: var(--pharma-green-light) !important;
    fill: currentColor !important;
}

#pharmadoc-upload p,
#pharmadoc-upload span,
#pharmadoc-upload label {
    color: var(--pharma-text) !important;
}


/* Reapply the floating upload label after broad upload text rules. */

#pharmadoc-upload .block-label,
#pharmadoc-upload .label-wrap,
#pharmadoc-upload [data-testid="block-label"] {
    background: var(--pharma-label) !important;
    color: #f0fdf4 !important;
}


/* ========================================================================
   Processing status and document summary
   ======================================================================== */

#processing-status,
#processing-status > div,
#processing-status .block,
#processing-status .wrap,
#processing-status .container,
#document-summary,
#document-summary > div,
#document-summary .block,
#document-summary .wrap,
#document-summary .container {
    background: var(--pharma-panel) !important;
    background-color: var(--pharma-panel) !important;
    border-color: var(--pharma-border) !important;
}

#processing-status textarea,
#processing-status input,
#document-summary textarea,
#document-summary input {
    background: var(--pharma-input) !important;
    background-color: var(--pharma-input) !important;
    color: var(--pharma-text) !important;
    border: 1px solid var(--pharma-border-light) !important;
    box-shadow:
        inset 0 0 0 1px rgba(34, 197, 94, 0.03) !important;
}


/* ========================================================================
   Question input
   ======================================================================== */

#question-input,
#question-input > div,
#question-input .block,
#question-input .wrap,
#question-input .container {
    background: var(--pharma-panel) !important;
    background-color: var(--pharma-panel) !important;
    border-color: var(--pharma-border) !important;
}

#question-input textarea,
#question-input input {
    background: var(--pharma-input) !important;
    background-color: var(--pharma-input) !important;
    color: #f8fafc !important;
    border-color: var(--pharma-border-light) !important;
}


/* ========================================================================
   General text inputs and dropdowns
   ======================================================================== */

.gradio-container textarea,
.gradio-container select,
.gradio-container input:not([type="checkbox"]):not([type="range"]) {
    background: var(--pharma-input) !important;
    background-color: var(--pharma-input) !important;
    color: #f8fafc !important;
    border-color: var(--pharma-border-light) !important;
}

.gradio-container textarea::placeholder,
.gradio-container input:not([type="checkbox"])::placeholder {
    color: #718078 !important;
}

.gradio-container [role="listbox"],
.gradio-container [role="option"],
.gradio-container .options,
.gradio-container .dropdown,
.gradio-container .dropdown-container {
    background: var(--pharma-input-soft) !important;
    background-color: var(--pharma-input-soft) !important;
    color: #f8fafc !important;
    border-color: var(--pharma-border-light) !important;
}

.gradio-container [role="option"]:hover,
.gradio-container [role="option"][aria-selected="true"] {
    background: var(--pharma-label) !important;
    color: #ffffff !important;
}


/* ========================================================================
   Checkboxes
   ======================================================================== */

.gradio-container input[type="checkbox"] {
    appearance: auto !important;
    -webkit-appearance: checkbox !important;

    width: 18px !important;
    height: 18px !important;
    min-width: 18px !important;

    margin: 0 8px 0 0 !important;
    padding: 0 !important;

    accent-color: var(--pharma-green) !important;
    cursor: pointer !important;
    opacity: 1 !important;
    visibility: visible !important;
}

.gradio-container label:has(input[type="checkbox"]) {
    cursor: pointer !important;
    user-select: none !important;
}

.gradio-container input[type="checkbox"]:focus-visible {
    outline: 2px solid #4ade80 !important;
    outline-offset: 2px !important;
}


/* ========================================================================
   Sliders
   ======================================================================== */

.gradio-container input[type="range"] {
    accent-color: var(--pharma-green) !important;
    cursor: pointer !important;
}


/* ========================================================================
   Buttons
   ======================================================================== */

.gradio-container button.primary,
.gradio-container button[variant="primary"] {
    background: linear-gradient(
        135deg,
        #16a34a,
        #15803d
    ) !important;

    color: #ffffff !important;
    border: 1px solid var(--pharma-green) !important;
    font-weight: 700 !important;

    box-shadow:
        0 8px 20px rgba(22, 163, 74, 0.22);
}

.gradio-container button.primary:hover,
.gradio-container button[variant="primary"]:hover {
    background: linear-gradient(
        135deg,
        #22c55e,
        #16a34a
    ) !important;

    border-color: #4ade80 !important;
    transform: translateY(-1px);
}

.gradio-container button.secondary,
.gradio-container button:not(.primary) {
    background: #172019 !important;
    color: var(--pharma-text) !important;
    border: 1px solid var(--pharma-border-light) !important;
}

.gradio-container button.secondary:hover,
.gradio-container button:not(.primary):hover {
    background: #223027 !important;
    border-color: #4ade80 !important;
}


/* ========================================================================
   Chat history
   ======================================================================== */

#pharmadoc-chatbot,
#pharmadoc-chatbot > div,
#pharmadoc-chatbot .block,
#pharmadoc-chatbot .wrap,
#pharmadoc-chatbot .panel,
#pharmadoc-chatbot .chatbot,
#pharmadoc-chatbot .chatbot-container,
#pharmadoc-chatbot .conversation,
#pharmadoc-chatbot .messages,
#pharmadoc-chatbot [role="log"] {
    background: #08110b !important;
    background-color: #08110b !important;
    border-color: var(--pharma-border) !important;
}

#pharmadoc-chatbot .empty,
#pharmadoc-chatbot .placeholder,
#pharmadoc-chatbot .center,
#pharmadoc-chatbot [data-testid="chatbot-empty"] {
    background: #08110b !important;
    color: #718078 !important;
}

#pharmadoc-chatbot .message-row,
#pharmadoc-chatbot .message-wrap,
#pharmadoc-chatbot .message-container {
    background: transparent !important;
}

#pharmadoc-chatbot .message.user,
#pharmadoc-chatbot .user-message,
#pharmadoc-chatbot [data-testid="user"] {
    background: #17351f !important;
    color: #f8fafc !important;
    border: 1px solid #2f7841 !important;
    border-radius: 14px !important;
}

#pharmadoc-chatbot .message.bot,
#pharmadoc-chatbot .message.assistant,
#pharmadoc-chatbot .bot-message,
#pharmadoc-chatbot [data-testid="bot"],
#pharmadoc-chatbot [data-testid="assistant"] {
    background: #101a13 !important;
    color: #f8fafc !important;
    border: 1px solid #2c4433 !important;
    border-radius: 14px !important;
}

#pharmadoc-chatbot .message p,
#pharmadoc-chatbot .message span,
#pharmadoc-chatbot .message div,
#pharmadoc-chatbot .message li,
#pharmadoc-chatbot .message code,
#pharmadoc-chatbot .message pre {
    color: #f8fafc !important;
}


/* Reapply the floating chatbot label after broad chatbot rules. */

#pharmadoc-chatbot .block-label,
#pharmadoc-chatbot .label-wrap,
#pharmadoc-chatbot [data-testid="block-label"],
#pharmadoc-chatbot [class*="block-label"],
#pharmadoc-chatbot [class*="label-wrap"] {
    background: var(--pharma-label) !important;
    background-color: var(--pharma-label) !important;
    color: #f0fdf4 !important;
    border-color: #2f7841 !important;
}


/* ========================================================================
   Advanced settings
   ======================================================================== */

#advanced-settings,
#advanced-settings > div,
#advanced-settings .block,
#advanced-settings .wrap {
    background: var(--pharma-panel) !important;
    background-color: var(--pharma-panel) !important;
    border-color: var(--pharma-border) !important;
}

#advanced-settings {
    border: 1px solid var(--pharma-border) !important;
    border-radius: 14px !important;
}

#advanced-settings summary {
    color: var(--pharma-green-light) !important;
    font-weight: 700 !important;
}


/* ========================================================================
   Technical details
   ======================================================================== */

.technical-details {
    margin-top: 10px;
    padding: 8px 10px;
    background: #0a120c;
    border: 1px solid #294532;
    border-radius: 10px;
    color: #cbd5e1;
}

.technical-details summary {
    color: var(--pharma-green-light);
    cursor: pointer;
    font-weight: 650;
}

.technical-details div {
    margin-top: 8px;
    color: #aebbb2;
    font-size: 0.92em;
    line-height: 1.55;
}


/* ========================================================================
   Tables
   ======================================================================== */

.gradio-container table {
    background: var(--pharma-panel) !important;
    color: var(--pharma-text) !important;
}

.gradio-container th {
    background: #142219 !important;
    color: var(--pharma-green-light) !important;
}

.gradio-container td {
    background: var(--pharma-panel) !important;
    color: var(--pharma-text) !important;
    border-color: var(--pharma-border) !important;
}


/* ========================================================================
   Informational note
   ======================================================================== */

.ui-note {
    margin-top: 10px;
    padding: 12px 14px;
    border-radius: 12px;
    background: #0d1710;
    border: 1px solid var(--pharma-border);
    color: #aebbb2 !important;
    font-size: 13px;
}

.ui-note strong {
    color: var(--pharma-green-light) !important;
}


/* ========================================================================
   Scrollbars
   ======================================================================== */

.gradio-container ::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

.gradio-container ::-webkit-scrollbar-track {
    background: #070b08;
}

.gradio-container ::-webkit-scrollbar-thumb {
    background: #294532;
    border-radius: 10px;
}

.gradio-container ::-webkit-scrollbar-thumb:hover {
    background: #357446;
}
"""

#@title CELL 33 — Build and launch the green-black Gradio interface

with gr.Blocks(
    css=CUSTOM_CSS,
    theme=APP_THEME,
    title="PharmaDoc AI",
) as demo:

    gr.HTML(
        """
        <div class="app-header">
            <h1>PharmaDoc AI</h1>
            <div class="app-subtitle">
                Multi-format RAG with OCR, tables, plot digitization,
                evidence gating, and source tracing
            </div>
        </div>
        """
    )

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, min_width=360):

            gr.Markdown("## Document Control Center")

            document_upload = gr.File(
                label="Upload documents",
                file_count="multiple",
                file_types=[
                    ".pdf",
                    ".docx",
                    ".txt",
                    ".csv",
                    ".xlsx",
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".tif",
                    ".tiff",
                    ".bmp",
                ],
                type="filepath",
                elem_id="pharmadoc-upload",
            )

            with gr.Row():
                enable_ocr = gr.Checkbox(
                    value=True,
                    label="Selective OCR",
                )

                enable_plot_extraction = gr.Checkbox(
                    value=True,
                    label="Plot-to-table",
                )

            model_dropdown = gr.Dropdown(
                choices=list(MODEL_CATALOG.keys()),
                value=DEFAULT_MODEL_CHOICE,
                label="Answer-generation model",
            )

            process_button = gr.Button(
                "Process Documents",
                variant="primary",
                size="lg",
            )

            processing_status = gr.Textbox(
                label="Processing status",
                value="No documents processed yet.",
                lines=8,
                interactive=False,
                elem_id="processing-status",
            )

            document_summary_box = gr.Textbox(
                label="Processed document summary",
                value="No documents processed yet.",
                lines=14,
                interactive=False,
                elem_id="document-summary",
            )

            with gr.Accordion(
                "Advanced settings",
                open=False,
                elem_id="advanced-settings",
            ):
                persist_after_processing = gr.Checkbox(
                    value=False,
                    label="Save FAISS artifacts",
                )

                document_filter = gr.Dropdown(
                    choices=["All documents"],
                    value="All documents",
                    label="Document filter",
                )

                doc_type_filter = gr.Dropdown(
                    choices=["All types"],
                    value="All types",
                    label="Document type filter",
                )

                content_type_filter = gr.Dropdown(
                    choices=["All content"],
                    value="All content",
                    label="Content type filter",
                )

                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=DEFAULT_TOP_K,
                    step=1,
                    label="Base retrieved-item count",
                )

        with gr.Column(scale=2, min_width=620):

            gr.Markdown("## Chat")

            chatbot = gr.Chatbot(
                label="Chat history",
                height=620,
                type="messages",
                elem_id="pharmadoc-chatbot",
            )

            question_box = gr.Textbox(
                label="Ask a question about the uploaded documents",
                placeholder=(
                    "Example: List all part numbers and "
                    "operating temperatures."
                ),
                lines=2,
                elem_id="question-input",
            )

            with gr.Row():
                send_button = gr.Button(
                    "Send",
                    variant="primary",
                    size="lg",
                )

                clear_button = gr.Button(
                    "Clear chat",
                    variant="secondary",
                    size="lg",
                )

                download_chat_button = gr.DownloadButton(
                    "Download chat history",
                    variant="secondary",
                    size="lg",
                )

            gr.HTML(
                """
                <div class="ui-note">
                    <strong>Notes:</strong>
                    OCR tables require review. Plot-derived values are
                    approximate. Sources, confidence, and chunk counts are
                    shown with every answer.
                </div>
                """
            )

    app_state = gr.State(RAGState())

    process_button.click(
        fn=process_uploaded_documents_for_gradio,
        inputs=[
            document_upload,
            enable_ocr,
            enable_plot_extraction,
            persist_after_processing,
            app_state,
        ],
        outputs=[
            processing_status,
            document_summary_box,
            document_filter,
            doc_type_filter,
            content_type_filter,
            app_state,
        ],
    )

    send_button.click(
        fn=gradio_chat_handler,
        inputs=[
            question_box,
            chatbot,
            model_dropdown,
            top_k_slider,
            document_filter,
            doc_type_filter,
            content_type_filter,
            app_state,
        ],
        outputs=[
            chatbot,
            question_box,
        ],
    )

    question_box.submit(
        fn=gradio_chat_handler,
        inputs=[
            question_box,
            chatbot,
            model_dropdown,
            top_k_slider,
            document_filter,
            doc_type_filter,
            content_type_filter,
            app_state,
        ],
        outputs=[
            chatbot,
            question_box,
        ],
    )

    clear_button.click(
        fn=lambda: ([], ""),
        inputs=None,
        outputs=[
            chatbot,
            question_box,
        ],
    )

    download_chat_button.click(
        fn=export_chat_history,
        inputs=[chatbot],
        outputs=[download_chat_button],
    )


if __name__ == "__main__":
    demo.queue().launch(
        share=True,
    )

