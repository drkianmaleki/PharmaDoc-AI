"""
pharmadoc/evaluation.py

Section 10 - Evaluation framework
Source notebook cells: [40, 41, 42]

Verbatim conversion: the code below this header is copied directly from
the notebook's cell source (mechanical extraction, not retyped). Only this
docstring and the import lines immediately below are new.
"""

# --- external imports (used by this file's verbatim code) ---
from collections import defaultdict

# --- cross-module imports (this package's own files) ---
from .answer_routing import RAG_NOT_FOUND_MESSAGE, _select_default_model_choice, answer_question_with_rag_general
from .config import SUPPORTED_CONTENT_TYPES
from .generation import format_sources

# ===== NOTEBOOK CELLS [40, 41, 42] (verbatim) =====

#@title CELL 26 — Phase 8 evaluation framework

import time
import pandas as pd


def _normalize_expected_text(value):
    return re.sub(
        r"\s+",
        " ",
        str(value or "").lower(),
    ).strip()


def evaluate_answer_constraints(
    answer,
    required_terms=None,
    optional_terms=None,
    forbidden_terms=None,
    expect_not_found=False,
):
    """
    Evaluate answer text using explicit required, optional, and forbidden terms.

    This is more reliable than one hard-coded exact string because dates,
    punctuation, units, and concise wording can vary without changing meaning.
    """
    answer_text = _normalize_expected_text(answer)
    required_terms = required_terms or []
    optional_terms = optional_terms or []
    forbidden_terms = forbidden_terms or []

    required_matches = {
        term: _normalize_expected_text(term) in answer_text
        for term in required_terms
    }
    optional_matches = {
        term: _normalize_expected_text(term) in answer_text
        for term in optional_terms
    }
    forbidden_matches = {
        term: _normalize_expected_text(term) in answer_text
        for term in forbidden_terms
    }

    not_found_phrase = _normalize_expected_text(
        RAG_NOT_FOUND_MESSAGE
    )
    not_found_correct = (
        not_found_phrase in answer_text
        if expect_not_found
        else not_found_phrase not in answer_text
    )

    passed = (
        all(required_matches.values())
        and not any(forbidden_matches.values())
        and not_found_correct
    )

    return {
        "passed": bool(passed),
        "required_matches": required_matches,
        "optional_matches": optional_matches,
        "forbidden_matches": forbidden_matches,
        "not_found_correct": bool(not_found_correct),
    }


def evaluate_source_constraints(
    retrieved_chunks,
    expected_file=None,
    expected_page=None,
    expected_content_type=None,
    source_top_k=3,
):
    """Evaluate whether an expected source appears in the retrieved results."""
    considered = list(retrieved_chunks or [])[:max(1, int(source_top_k))]

    def file_matches(item):
        if expected_file is None:
            return True
        return _normalize_expected_text(expected_file) in _normalize_expected_text(
            item.get("file", "")
        )

    def page_matches(item):
        if expected_page is None:
            return True
        page_start = item.get("page_start", item.get("page"))
        page_end = item.get("page_end", page_start)
        try:
            return int(page_start) <= int(expected_page) <= int(page_end)
        except Exception:
            return str(expected_page) == str(page_start)

    def type_matches(item):
        if expected_content_type is None:
            return True
        return _normalize_expected_text(
            item.get("content_type", "")
        ) == _normalize_expected_text(expected_content_type)

    matching_ranks = [
        rank
        for rank, item in enumerate(considered, start=1)
        if file_matches(item)
        and page_matches(item)
        and type_matches(item)
    ]

    return {
        "source_passed": bool(matching_ranks) or (
            expected_file is None
            and expected_page is None
            and expected_content_type is None
        ),
        "matching_ranks": matching_ranks,
        "rank_1_match": bool(matching_ranks and matching_ranks[0] == 1),
        "top_k_match": bool(matching_ranks),
    }


def run_phase8_rag_evaluation(
    test_cases,
    model_choice=None,
    top_k=2,
    show_details=True,
    faiss_index=None,
    content_items=None,
    embedding_model=None,
):
    """
    Run answer, source, refusal, routing, and latency evaluation.

    Each test case may contain:
        question
        required_terms
        optional_terms
        forbidden_terms
        expect_not_found
        expected_file
        expected_page
        expected_content_type
        document_filter
        doc_type_filter
        content_type_filter
    """
    rows = []
    detailed_results = []

    if model_choice is None:
        model_choice = _select_default_model_choice()

    for test_number, test_case in enumerate(test_cases, start=1):
        started = time.perf_counter()

        try:
            result = answer_question_with_rag(
                question=test_case["question"],
                model_choice=model_choice,
                faiss_index=faiss_index,
                content_items=content_items,
                embedding_model=embedding_model,
                top_k=int(top_k),
                document_filter=test_case.get(
                    "document_filter",
                    "All documents",
                ),
                doc_type_filter=test_case.get(
                    "doc_type_filter",
                    "All types",
                ),
                content_type_filter=test_case.get(
                    "content_type_filter",
                    SUPPORTED_CONTENT_TYPES,
                ),
            )
            error = ""
        except Exception as exception:
            result = {
                "answer": f"ERROR: {exception}",
                "sources": "",
                "confidence": "",
                "top_score": None,
                "chunk_count": 0,
                "retrieved_chunks": [],
                "answer_route": "error",
                "answer_status": "error",
            }
            error = f"{type(exception).__name__}: {exception}"

        elapsed = time.perf_counter() - started

        answer_check = evaluate_answer_constraints(
            answer=result.get("answer", ""),
            required_terms=test_case.get("required_terms"),
            optional_terms=test_case.get("optional_terms"),
            forbidden_terms=test_case.get("forbidden_terms"),
            expect_not_found=bool(
                test_case.get("expect_not_found", False)
            ),
        )

        source_check = evaluate_source_constraints(
            retrieved_chunks=result.get("retrieved_chunks", []),
            expected_file=test_case.get("expected_file"),
            expected_page=test_case.get("expected_page"),
            expected_content_type=test_case.get(
                "expected_content_type"
            ),
            source_top_k=max(3, int(top_k)),
        )

        passed = (
            answer_check["passed"]
            and source_check["source_passed"]
            and not error
        )

        row = {
            "test": test_number,
            "category": test_case.get("category", "general"),
            "question": test_case["question"],
            "passed": bool(passed),
            "answer_passed": answer_check["passed"],
            "source_passed": source_check["source_passed"],
            "rank_1_source": source_check["rank_1_match"],
            "top_k_source": source_check["top_k_match"],
            "response_time_seconds": round(elapsed, 3),
            "confidence": result.get("confidence"),
            "top_score": result.get("top_score"),
            "chunk_count": result.get(
                "chunk_count",
                result.get("chunks_used", 0),
            ),
            "answer_route": result.get("answer_route"),
            "answer_status": result.get("answer_status"),
            "answer": result.get("answer", ""),
            "error": error,
        }

        rows.append(row)
        detailed_results.append({
            "test_case": test_case,
            "result": result,
            "answer_check": answer_check,
            "source_check": source_check,
            "row": row,
        })

    dataframe = pd.DataFrame(rows)

    if len(dataframe):
        metrics = {
            "tests": int(len(dataframe)),
            "overall_pass_rate": float(dataframe["passed"].mean()),
            "answer_accuracy": float(
                dataframe["answer_passed"].mean()
            ),
            "source_accuracy": float(
                dataframe["source_passed"].mean()
            ),
            "rank_1_source_accuracy": float(
                dataframe["rank_1_source"].mean()
            ),
            "top_k_source_recall": float(
                dataframe["top_k_source"].mean()
            ),
            "average_response_time_seconds": float(
                dataframe["response_time_seconds"].mean()
            ),
            "median_response_time_seconds": float(
                dataframe["response_time_seconds"].median()
            ),
        }
    else:
        metrics = {}

    if show_details:
        display(dataframe)
        print("\nPHASE 8 METRICS")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")

    return {
        "summary": dataframe,
        "details": detailed_results,
        "metrics": metrics,
    }


def run_evidence_gate_stability_test(
    question_variants,
    expected_not_found,
    model_choice=None,
    top_k=2,
):
    """
    Test whether small wording changes produce stable refusal decisions.
    """
    rows = []

    for question in question_variants:
        started = time.perf_counter()
        result = answer_question_with_rag(
            question=question,
            model_choice=(
                model_choice
                or _select_default_model_choice()
            ),
            top_k=top_k,
        )
        elapsed = time.perf_counter() - started
        answer = result.get("answer", "")
        refused = (
            _normalize_expected_text(RAG_NOT_FOUND_MESSAGE)
            in _normalize_expected_text(answer)
        )

        rows.append({
            "question": question,
            "expected_not_found": bool(expected_not_found),
            "refused": bool(refused),
            "stable_pass": bool(refused) == bool(expected_not_found),
            "answer_route": result.get("answer_route"),
            "response_time_seconds": round(elapsed, 3),
            "answer": answer,
        })

    dataframe = pd.DataFrame(rows)
    display(dataframe)

    return dataframe


#@title CELL 27 — Phase 8 plot extraction validation

from IPython.display import display


def validate_plot_items(
    content_items=None,
    expected_file_contains=None,
    expected_page=None,
    minimum_series=1,
    minimum_rows=3,
):
    """
    Validate plot detection independently from RAG answer generation.

    This distinguishes:
        1. Region/axis detection failure
        2. Calibration failure
        3. Series extraction failure
        4. Retrieval or answer-generation failure
    """
    content_items = (
        content_items
        if content_items is not None
        else globals().get("rag_content_items", [])
    )

    plot_items = [
        item
        for item in content_items or []
        if item.get("content_type") == "plot_table"
    ]

    if expected_file_contains:
        plot_items = [
            item
            for item in plot_items
            if _normalize_expected_text(expected_file_contains)
            in _normalize_expected_text(item.get("file", ""))
        ]

    if expected_page is not None:
        plot_items = [
            item
            for item in plot_items
            if int(item.get("page_start", -1))
            == int(expected_page)
        ]

    rows = []

    for index, item in enumerate(plot_items, start=1):
        structured_data = item.get("structured_data", [])
        series_names = item.get("series_names", [])

        row_count = (
            len(structured_data)
            if isinstance(structured_data, list)
            else 0
        )

        passed = (
            len(series_names) >= int(minimum_series)
            and row_count >= int(minimum_rows)
        )

        rows.append({
            "plot_index": index,
            "file": item.get("file"),
            "page": item.get("page_start"),
            "passed": passed,
            "series_count": len(series_names),
            "series_names": series_names,
            "row_count": row_count,
            "confidence": item.get("confidence"),
            "detection_method": item.get(
                "plot_detection_method"
            ),
            "extraction_method": item.get(
                "extraction_method"
            ),
            "bbox": item.get("bbox"),
        })

    dataframe = pd.DataFrame(rows)

    print(f"Detected matching plot items: {len(plot_items)}")

    if dataframe.empty:
        print(
            "PLOT VALIDATION FAILED: no matching plot_table item "
            "was created."
        )
    else:
        display(dataframe)

        for index, item in enumerate(plot_items, start=1):
            print("=" * 100)
            print(f"PLOT {index}: {item.get('file')} page {item.get('page_start')}")
            print(f"Series: {item.get('series_names')}")
            print(f"Confidence: {item.get('confidence')}")
            print("\nPlot-derived table:")
            structured_data = item.get("structured_data", [])

            if isinstance(structured_data, list):
                display(pd.DataFrame(structured_data))
            else:
                print(structured_data)

    return {
        "passed": bool(
            len(plot_items)
            and all(row["passed"] for row in rows)
        ),
        "items": plot_items,
        "summary": dataframe,
    }


def summarize_phase8_content_coverage(
    content_items=None,
    registry=None,
):
    """Report extraction coverage by document and content type."""
    content_items = (
        content_items
        if content_items is not None
        else globals().get("rag_content_items", [])
    )
    registry = (
        registry
        if registry is not None
        else globals().get("document_registry", {})
    )

    rows = []

    for document_id, record in registry.items():
        document_items = [
            item
            for item in content_items
            if item.get("document_id") == document_id
        ]

        counts = defaultdict(int)

        for item in document_items:
            counts[item.get("content_type", "unknown")] += 1

        rows.append({
            "document_id": document_id,
            "file": record.get("file"),
            "doc_type": record.get("doc_type"),
            "pages": record.get("num_pages", record.get("page_count", 0)),
            "text": counts["text"],
            "table": counts["table"],
            "ocr_text": counts["ocr_text"],
            "ocr_table": counts["ocr_table"],
            "plot_table": counts["plot_table"],
            "total_indexed_items": len(document_items),
            "warnings": " | ".join(record.get("warnings", [])),
        })

    dataframe = pd.DataFrame(rows)
    display(dataframe)

    return dataframe


#@title CELL 28 — Plot-specific retrieval and deterministic answering

import re
import numpy as np


# Preserve the validated general RAG backend only the first time this
# cell is executed. This prevents recursive wrapping when rerunning it.
_phase8_general_rag_backend = answer_question_with_rag_general


PLOT_QUERY_TERMS = {
    "plot",
    "graph",
    "chart",
    "figure",
    "curve",
    "trajectory",
    "trend",
    "series",
    "week",
    "weeks",
    "baseline",
    "placebo",
    "treatment",
    "treatment group",
    "symptom score",
    "neuropathy symptom score",
    "nss",
    "lowest",
    "highest",
}


def _normalize_plot_text(value):
    """
    Normalize plot-query and series text for matching.
    """
    value = str(value or "").lower()

    value = (
        value
        .replace("–", "-")
        .replace("—", "-")
        .replace("≥", ">=")
        .replace("≤", "<=")
    )

    value = re.sub(
        r"[^a-z0-9.%+\-=\s]",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def is_plot_question(question):
    """
    Identify questions that should preferentially use plot-derived data.

    This is intentionally based on several signals rather than requiring
    the literal word 'plot'.
    """
    question_text = _normalize_plot_text(
        question
    )

    if not question_text:
        return False

    if any(
        term in question_text
        for term in PLOT_QUERY_TERMS
    ):
        return True

    # Numeric x-position questions such as "at Week 12".
    if re.search(
        r"\b(?:week|day|month|year|time)\s*\d+(?:\.\d+)?\b",
        question_text,
    ):
        return True

    return False



def requires_general_comparison_route(question):
    """
    Prevent a multi-part efficacy/safety comparison from being intercepted
    by the single-value deterministic plot router.
    """
    question_text = _normalize_plot_text(question)

    comparison_requested = any(
        term in question_text
        for term in (
            "compare",
            "comparison",
            "difference",
            "differences",
        )
    )

    non_plot_metric_requested = any(
        term in question_text
        for term in (
            "adverse event",
            "adverse-event",
            "severe adverse",
            "sae",
            "safety",
            "discontinuation",
        )
    )

    # Statistical analysis plan / study design questions are never answered
    # from plot data even if they mention a time point like "Week 12".
    statistical_design_question = any(
        term in question_text
        for term in (
            "alpha allocation",
            "alpha level",
            "significance level",
            "p value",
            "multiplicity",
            "randomization",
            "estimand",
        )
    )

    return (
        (comparison_requested and non_plot_metric_requested)
        or statistical_design_question
    )


def _plot_rows_are_valid(structured_data):
    """
    Validate that plot data contains a usable monotonic numeric x-axis
    and at least two numeric series.
    """
    if (
        not isinstance(structured_data, list)
        or len(structured_data) < 3
    ):
        return False

    x_values = []

    for row in structured_data:
        if not isinstance(row, dict):
            return False

        try:
            x_values.append(float(row["x"]))
        except (KeyError, TypeError, ValueError):
            return False

    if len(set(x_values)) != len(x_values):
        return False

    differences = np.diff(
        np.asarray(x_values, dtype=float)
    )

    if not np.all(differences > 0):
        return False

    ignored_columns = {
        "x",
        "uncertainty_note",
    }

    possible_series = [
        key
        for key in structured_data[0].keys()
        if key not in ignored_columns
    ]

    numeric_series = []

    for series_name in possible_series:
        numeric_count = 0

        for row in structured_data:
            value = row.get(series_name)

            try:
                float(value)
                numeric_count += 1
            except (TypeError, ValueError):
                continue

        if numeric_count >= 2:
            numeric_series.append(series_name)

    return len(numeric_series) >= 2


def _is_valid_plot_item(item):
    """
    Apply final safety validation before a plot item can be retrieved.
    """
    if not isinstance(item, dict):
        return False

    if str(
        item.get("content_type", "")
    ).lower() != "plot_table":
        return False

    structured_data = item.get(
        "structured_data"
    )

    if not _plot_rows_are_valid(
        structured_data
    ):
        return False

    confidence = item.get(
        "confidence",
        item.get("plot_confidence", 0.0),
    )

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    # This removes low-quality residual candidates while retaining the
    # validated clinical-trial chart with confidence 0.80.
    if confidence < 0.70:
        return False

    return True


def _plot_item_search_text(item):
    """
    Build compact searchable text for a validated plot item.
    """
    parts = [
        item.get("file", ""),
        item.get("text_for_embedding", ""),
        item.get("text_for_llm", ""),
        " ".join(
            str(value)
            for value in item.get(
                "series_names",
                [],
            )
        ),
    ]

    return " ".join(
        str(part)
        for part in parts
        if part
    )


def retrieve_plot_items(
    question,
    content_items,
    embedding_model,
    top_k=2,
    document_filter="All documents",
):
    """
    Retrieve validated plot-derived items directly.

    Every returned result receives:
        semantic_score
        rerank_score
        score

    This keeps the result schema compatible with format_sources(),
    confidence estimation, Gradio, and Phase 8 evaluation.
    """
    if not content_items:
        return []

    valid_plot_items = []

    for item in content_items:
        if not _is_valid_plot_item(item):
            continue

        if (
            document_filter != "All documents"
            and item.get("file") != document_filter
        ):
            continue

        valid_plot_items.append(item)

    if not valid_plot_items:
        return []

    plot_texts = [
        _plot_item_search_text(item)
        for item in valid_plot_items
    ]

    query_embedding = embedding_model.encode(
        [str(question)],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")[0]

    plot_embeddings = embedding_model.encode(
        plot_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    semantic_scores = np.dot(
        plot_embeddings,
        query_embedding,
    )

    question_text = _normalize_plot_text(
        question
    )

    results = []

    for item, semantic_score in zip(
        valid_plot_items,
        semantic_scores,
    ):
        result = dict(item)

        semantic_score = float(
            semantic_score
        )

        lexical_bonus = 0.0

        item_text = _normalize_plot_text(
            _plot_item_search_text(item)
        )

        query_tokens = {
            token
            for token in question_text.split()
            if len(token) >= 3
        }

        item_tokens = set(
            item_text.split()
        )

        if query_tokens:
            overlap_ratio = (
                len(query_tokens & item_tokens)
                / len(query_tokens)
            )

            lexical_bonus += (
                0.12 * overlap_ratio
            )

        # Explicit plot-table preference.
        lexical_bonus += 0.08

        rerank_score = (
            semantic_score
            + lexical_bonus
        )

        result["semantic_score"] = (
            semantic_score
        )

        result["rerank_score"] = (
            float(rerank_score)
        )

        # Public score used by source formatting and evaluation.
        result["score"] = float(
            rerank_score
        )

        results.append(result)

    results.sort(
        key=lambda item: item.get(
            "rerank_score",
            0.0,
        ),
        reverse=True,
    )

    return results[:max(1, int(top_k))]


def _get_plot_series_names(plot_item):
    """
    Return numeric plot-series columns from structured_data.
    """
    structured_data = plot_item.get(
        "structured_data",
        [],
    )

    if not structured_data:
        return []

    ignored_columns = {
        "x",
        "uncertainty_note",
    }

    return [
        key
        for key in structured_data[0].keys()
        if key not in ignored_columns
    ]


def _extract_requested_x_value(question):
    """
    Extract a requested x-position such as Week 12.
    """
    question_text = _normalize_plot_text(
        question
    )

    patterns = [
        r"\bweek\s*(\d+(?:\.\d+)?)\b",
        r"\bat\s+(?:x\s*=\s*)?(\d+(?:\.\d+)?)\b",
        r"\bx\s*=\s*(\d+(?:\.\d+)?)\b",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            question_text,
        )

        if match:
            return float(
                match.group(1)
            )

    return None


def _find_plot_row_at_x(
    structured_data,
    requested_x,
):
    """
    Find an exact or nearest plot row for the requested x-position.
    """
    if requested_x is None:
        return None

    candidates = []

    for row in structured_data:
        try:
            row_x = float(row.get("x"))
        except (TypeError, ValueError):
            continue

        candidates.append(
            (
                abs(row_x - requested_x),
                row_x,
                row,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda result: result[0]
    )

    distance, row_x, row = candidates[0]

    x_values = sorted(
        {
            float(candidate[1])
            for candidate in candidates
        }
    )

    if len(x_values) >= 2:
        typical_step = float(
            np.median(
                np.diff(x_values)
            )
        )
    else:
        typical_step = 1.0

    # Do not silently answer from a distant x-position.
    if distance > max(
        0.25,
        typical_step * 0.30,
    ):
        return None

    return row


def _normalize_series_name(value):
    """
    Normalize a series label while removing sample-size annotations.
    """
    value = _normalize_plot_text(value)

    value = re.sub(
        r"\bn\s*=\s*\d+\b",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def _match_requested_series(
    question,
    series_names,
):
    """
    Match a named series such as placebo, Luminara 25 mg, or 50 mg.
    """
    question_text = _normalize_plot_text(
        question
    )

    # Strong dose-specific matching.
    dose_match = re.search(
        r"\b(25|50)\s*mg\b",
        question_text,
    )

    requested_dose = (
        dose_match.group(1)
        if dose_match
        else None
    )

    if "placebo" in question_text:
        for series_name in series_names:
            if "placebo" in _normalize_series_name(
                series_name
            ):
                return series_name

    if requested_dose:
        for series_name in series_names:
            normalized_series = (
                _normalize_series_name(
                    series_name
                )
            )

            if (
                requested_dose
                in normalized_series
                and "luminara" in normalized_series
            ):
                return series_name

    # General token-overlap fallback.
    best_series = None
    best_overlap = 0

    query_tokens = set(
        question_text.split()
    )

    for series_name in series_names:
        series_tokens = set(
            _normalize_series_name(
                series_name
            ).split()
        )

        overlap = len(
            query_tokens & series_tokens
        )

        if overlap > best_overlap:
            best_overlap = overlap
            best_series = series_name

    if best_overlap > 0:
        return best_series

    return None


def _numeric_plot_values(row, series_names):
    """
    Return valid numeric series values from one plot row.
    """
    values = {}

    for series_name in series_names:
        value = row.get(series_name)

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(numeric_value):
            continue

        values[series_name] = (
            numeric_value
        )

    return values


def _format_approximate_plot_value(value):
    """
    Format digitized values without implying excessive precision.
    """
    value = float(value)

    if abs(value) >= 100:
        return f"{value:.1f}"

    return f"{value:.2f}".rstrip(
        "0"
    ).rstrip(".")


def answer_plot_question_deterministically(
    question,
    plot_item,
):
    """
    Answer supported plot questions directly from structured plot data.

    Supported intents:
        - exact series value at an x-position
        - lowest series at an x-position
        - highest series at an x-position
    """
    structured_data = plot_item.get(
        "structured_data",
        [],
    )

    series_names = _get_plot_series_names(
        plot_item
    )

    if not structured_data or not series_names:
        return None

    requested_x = _extract_requested_x_value(
        question
    )

    if requested_x is None:
        return None

    row = _find_plot_row_at_x(
        structured_data,
        requested_x,
    )

    if row is None:
        return None

    numeric_values = _numeric_plot_values(
        row,
        series_names,
    )

    if not numeric_values:
        return None

    question_text = _normalize_plot_text(
        question
    )

    uncertainty_note = str(
        row.get(
            "uncertainty_note",
            "Approximate; digitized from rendered plot pixels",
        )
    )

    # ------------------------------------------------------------------
    # Lowest series
    # ------------------------------------------------------------------

    if any(
        term in question_text
        for term in (
            "lowest",
            "minimum",
            "smallest",
            "best symptom relief",
            "greatest symptom relief",
        )
    ):
        series_name, value = min(
            numeric_values.items(),
            key=lambda pair: pair[1],
        )

        answer = (
            f"At Week {_format_approximate_plot_value(requested_x)}, "
            f"{series_name} has the lowest mean Neuropathy Symptom "
            f"Score, approximately "
            f"{_format_approximate_plot_value(value)}."
        )

        return {
            "answer": answer,
            "answer_route": (
                "deterministic_plot_lowest"
            ),
            "uncertainty_note": (
                uncertainty_note
            ),
        }

    # ------------------------------------------------------------------
    # Highest series
    # ------------------------------------------------------------------

    if any(
        term in question_text
        for term in (
            "highest",
            "maximum",
            "largest",
        )
    ):
        series_name, value = max(
            numeric_values.items(),
            key=lambda pair: pair[1],
        )

        answer = (
            f"At Week {_format_approximate_plot_value(requested_x)}, "
            f"{series_name} has the highest mean Neuropathy Symptom "
            f"Score, approximately "
            f"{_format_approximate_plot_value(value)}."
        )

        return {
            "answer": answer,
            "answer_route": (
                "deterministic_plot_highest"
            ),
            "uncertainty_note": (
                uncertainty_note
            ),
        }

    # ------------------------------------------------------------------
    # Named series value
    # ------------------------------------------------------------------

    requested_series = _match_requested_series(
        question,
        series_names,
    )

    if (
        requested_series
        and requested_series in numeric_values
    ):
        value = numeric_values[
            requested_series
        ]

        answer = (
            f"At Week {_format_approximate_plot_value(requested_x)}, "
            f"the mean Neuropathy Symptom Score for "
            f"{requested_series} is approximately "
            f"{_format_approximate_plot_value(value)}."
        )

        return {
            "answer": answer,
            "answer_route": (
                "deterministic_plot_exact_value"
            ),
            "uncertainty_note": (
                uncertainty_note
            ),
        }

    return None


def _plot_confidence_label(plot_item):
    """
    Convert plot-extraction confidence to a user-facing label.
    """
    confidence = plot_item.get(
        "confidence",
        0.0,
    )

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    if confidence >= 0.80:
        return "High"

    if confidence >= 0.65:
        return "Medium"

    return "Low"


def answer_question_with_rag(
    question,
    model_choice=None,
    faiss_index=None,
    content_items=None,
    embedding_model=None,
    top_k=2,
    document_filter="All documents",
    doc_type_filter="All types",
    content_type_filter=None,
    max_tokens=80,
    model_name=None,
):
    """
    Phase 8 answer router.

    Plot questions:
        validated plot retrieval
        → deterministic plot answering
        → plot-specific source metadata

    All other questions:
        validated general RAG backend
    """
    if content_items is None:
        content_items = globals().get(
            "rag_content_items"
        )

    if embedding_model is None:
        embedding_model = globals().get(
            "embedding_model"
        )

    question_text = str(
        question or ""
    ).strip()

    if (
        question_text
        and is_plot_question(question_text)
        and not requires_general_comparison_route(question_text)
        and content_items
        and embedding_model is not None
    ):
        plot_results = retrieve_plot_items(
            question=question_text,
            content_items=content_items,
            embedding_model=embedding_model,
            top_k=top_k,
            document_filter=document_filter,
        )

        if plot_results:
            best_plot = plot_results[0]

            deterministic_result = (
                answer_plot_question_deterministically(
                    question=question_text,
                    plot_item=best_plot,
                )
            )

            if deterministic_result is not None:
                source_text = format_sources(
                    plot_results
                )

                top_score = float(
                    best_plot.get(
                        "semantic_score",
                        best_plot.get(
                            "score",
                            0.0,
                        ),
                    )
                )

                uncertainty_note = (
                    deterministic_result.get(
                        "uncertainty_note",
                        "",
                    )
                )

                answer = str(
                    deterministic_result[
                        "answer"
                    ]
                ).strip()

                if uncertainty_note:
                    answer += (
                        "\n\n"
                        "Note: "
                        + uncertainty_note
                        + "."
                    )

                return {
                    "answer": answer,
                    "sources": source_text,
                    "confidence": (
                        _plot_confidence_label(
                            best_plot
                        )
                    ),
                    "top_score": top_score,
                    "chunks_used": len(
                        plot_results
                    ),
                    "chunk_count": len(
                        plot_results
                    ),
                    "retrieved_chunks": (
                        plot_results
                    ),
                    "rag_prompt": "",
                    "model_choice": (
                        model_choice
                        or model_name
                        or globals().get(
                            "DEFAULT_MODEL_CHOICE",
                            "",
                        )
                    ),
                    "evidence_check": {
                        "has_evidence": True,
                        "focus_terms": [],
                        "matched_terms": [],
                        "missing_terms": [],
                        "match_ratio": 1.0,
                    },
                    "answer_status": (
                        "plot_answer"
                    ),
                    "answer_route": (
                        deterministic_result[
                            "answer_route"
                        ]
                    ),
                }

    # All non-plot questions and unsupported plot intents use the
    # previously validated full RAG backend.
    return _phase8_general_rag_backend(
        question=question,
        model_choice=model_choice,
        faiss_index=faiss_index,
        content_items=content_items,
        embedding_model=embedding_model,
        top_k=top_k,
        document_filter=document_filter,
        doc_type_filter=doc_type_filter,
        content_type_filter=content_type_filter,
        max_tokens=max_tokens,
        model_name=model_name,
    )

