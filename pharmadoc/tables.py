"""
pharmadoc/tables.py

Section 4 - Structured table extraction
Source notebook cells: [14, 15, 16, 17, 18]

Verbatim conversion: the code below this header is copied directly from
the notebook's cell source (mechanical extraction, not retyped). Only this
docstring and the import lines immediately below are new.
"""

# --- external imports (used by this file's verbatim code) ---
import fitz
import re

# --- cross-module imports (this package's own files) ---
from .metadata import create_content_item

# ===== NOTEBOOK CELLS [14, 15, 16, 17, 18] (verbatim) =====

#@title CELL 10 — Table classification and formatting helpers

TABLE_CONFIG = {
    "TOLERANCE": 3,
    "X_TOLERANCE": 10,
    "GAP_THRESHOLD": 20,
    "MIN_ROWS": 2,
    "MIN_COLUMNS": 2,

    # Header-recovery settings
    "HEADER_SEARCH_GAP": 25,
    "HEADER_Y_TOLERANCE": 3,
}


def clean_table_cell(value):
    """Normalize one table cell without changing its meaning."""
    if value is None:
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_table_matrix(table):
    """
    Clean a table and pad rows to a consistent number of columns.
    """
    if not table:
        return []

    cleaned_rows = []

    for row in table:
        cleaned_row = [clean_table_cell(cell) for cell in row]

        if any(cleaned_row):
            cleaned_rows.append(cleaned_row)

    if not cleaned_rows:
        return []

    max_columns = max(len(row) for row in cleaned_rows)

    return [
        row + [""] * (max_columns - len(row))
        for row in cleaned_rows
    ]


def strip_field_label_punctuation(value):
    """Remove trailing label punctuation."""
    return re.sub(r"\s*[:：]\s*$", "", clean_table_cell(value))


def looks_like_field_label(value):
    """Estimate whether a cell looks like a key–value field label."""
    value = clean_table_cell(value)

    if not value:
        return False

    if value.endswith((":", "：")):
        return True

    if re.fullmatch(
        r"[\d\s./+\-–—%°A-Za-z]*\d[\d\s./+\-–—%°A-Za-z]*",
        value
    ):
        return False

    words = value.split()

    if len(words) > 7:
        return False

    label_terms = {
        "product", "lot", "number", "date", "description", "revision",
        "manufacturer", "address", "component", "reference", "status",
        "temperature", "pressure", "weight", "dimensions", "shelf",
        "supplier", "audit", "result", "site", "warehouse", "agreement",
        "assessment", "code", "name", "article", "expiration",
        "manufacture", "valid", "effective", "drawing", "part",
        "qualification", "certification"
    }

    tokens = {
        re.sub(r"[^a-z0-9]+", "", token.lower())
        for token in words
    }

    return bool(tokens & label_terms)


def classify_table_structure(table):
    """
    Classify a detected structure as header_table or key_value.
    """
    table = normalize_table_matrix(table)

    if not table:
        return "header_table"

    column_count = len(table[0])

    if column_count != 2:
        return "header_table"

    first_column = [
        row[0]
        for row in table
        if row and clean_table_cell(row[0])
    ]

    if not first_column:
        return "header_table"

    colon_ratio = sum(
        clean_table_cell(value).endswith((":", "："))
        for value in first_column
    ) / len(first_column)

    label_ratio = sum(
        looks_like_field_label(value)
        for value in first_column
    ) / len(first_column)

    first_row_is_explicit_label = clean_table_cell(
        table[0][0]
    ).endswith((":", "："))

    if first_row_is_explicit_label:
        return "key_value"

    if colon_ratio >= 0.40:
        return "key_value"

    if label_ratio >= 0.75:
        return "key_value"

    return "header_table"


def header_table_to_markdown(table):
    """Convert a header-based table into Markdown."""
    table = normalize_table_matrix(table)

    if not table:
        return ""

    column_count = len(table[0])

    header = [
        value if value else f"Column {index + 1}"
        for index, value in enumerate(table[0])
    ]

    header = [
        value.replace("|", "\\|")
        for value in header
    ]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * column_count) + " |",
    ]

    for row in table[1:]:
        safe_row = [
            value.replace("|", "\\|")
            for value in row
        ]

        lines.append("| " + " | ".join(safe_row) + " |")

    return "\n".join(lines)


def key_value_to_markdown(table):
    """Convert a two-column key–value block into Markdown."""
    table = normalize_table_matrix(table)

    if not table:
        return ""

    lines = [
        "| Field | Value |",
        "| --- | --- |",
    ]

    for row in table:
        field = strip_field_label_punctuation(
            row[0]
        ).replace("|", "\\|")

        value = clean_table_cell(
            row[1]
        ).replace("|", "\\|")

        if not field and not value:
            continue

        lines.append(f"| {field} | {value} |")

    return "\n".join(lines)


def header_table_to_embedding_text(table):
    """Create embedding text for a header-based table."""
    table = normalize_table_matrix(table)

    if not table:
        return ""

    headers = table[0]
    body = table[1:]
    searchable_rows = []

    for row in body:
        field_value_pairs = []

        for column_index, value in enumerate(row):
            if not value:
                continue

            header = (
                headers[column_index]
                or f"Column {column_index + 1}"
            )

            field_value_pairs.append(
                f"{header}: {value}"
            )

        if field_value_pairs:
            searchable_rows.append(
                "; ".join(field_value_pairs)
            )

    if not searchable_rows:
        searchable_rows = [
            " | ".join(row)
            for row in table
            if any(row)
        ]

    return "\n".join(searchable_rows)


def key_value_to_embedding_text(table):
    """Create embedding text for a key–value block."""
    table = normalize_table_matrix(table)

    if not table:
        return ""

    searchable_rows = []

    for row in table:
        field = strip_field_label_punctuation(row[0])
        value = clean_table_cell(row[1])

        if field and value:
            searchable_rows.append(f"{field}: {value}")
        elif field:
            searchable_rows.append(field)
        elif value:
            searchable_rows.append(value)

    return "\n".join(searchable_rows)


def format_detected_structure(table):
    """
    Classify and format one detected structure.
    """
    table = normalize_table_matrix(table)
    structure_type = classify_table_structure(table)

    if structure_type == "key_value":
        embedding_text = key_value_to_embedding_text(table)
        markdown_text = key_value_to_markdown(table)
    else:
        embedding_text = header_table_to_embedding_text(table)
        markdown_text = header_table_to_markdown(table)

    return structure_type, embedding_text, markdown_text


def table_to_markdown(table):
    return format_detected_structure(table)[2]


def table_to_embedding_text(table):
    return format_detected_structure(table)[1]


def combine_bboxes(bboxes):
    """Return one bounding box enclosing all supplied boxes."""
    valid = [
        bbox for bbox in bboxes
        if bbox is not None and len(bbox) == 4
    ]

    if not valid:
        return None

    return [
        min(bbox[0] for bbox in valid),
        min(bbox[1] for bbox in valid),
        max(bbox[2] for bbox in valid),
        max(bbox[3] for bbox in valid),
    ]


#@title CELL 11 — Primary geometric table detector

def cluster_column_positions(table_rows, x_tolerance):
    """
    Infer table column anchors from repeated cell x-coordinates.

    This runs after the original geometric detector has already
    identified a table.
    """
    x_positions = sorted(
        cell["bbox"][0]
        for row in table_rows
        for cell in row
    )

    if not x_positions:
        return []

    clusters = [[x_positions[0]]]

    for x_position in x_positions[1:]:
        current_center = sum(clusters[-1]) / len(clusters[-1])

        if abs(x_position - current_center) <= x_tolerance:
            clusters[-1].append(x_position)
        else:
            clusters.append([x_position])

    return [
        sum(cluster) / len(cluster)
        for cluster in clusters
    ]


def geometric_rows_to_matrix(table_rows):
    """
    Convert geometrically detected rows into a rectangular matrix.

    Detection has already occurred. This step maps each cell into the
    nearest inferred column.
    """
    column_positions = cluster_column_positions(
        table_rows,
        TABLE_CONFIG["X_TOLERANCE"]
    )

    if not column_positions:
        return []

    matrix = []

    for row in table_rows:
        output_row = [""] * len(column_positions)

        for cell in row:
            cell_x = cell["bbox"][0]

            closest_column = min(
                range(len(column_positions)),
                key=lambda index: abs(
                    cell_x - column_positions[index]
                )
            )

            cell_text = clean_table_cell(cell["text"])

            if output_row[closest_column]:
                output_row[closest_column] += " " + cell_text
            else:
                output_row[closest_column] = cell_text

        matrix.append(output_row)

    return normalize_table_matrix(matrix)


def detect_geometric_tables_on_page(page):
    """
    Primary detector based on the user's original algorithm.

    Steps:
    1. Join spans into line-level text objects.
    2. Snap slightly misaligned lines into common rows.
    3. Retain rows containing multiple horizontally separated items.
    4. Validate fuzzy repeated x-coordinate alignment.
    5. Segment vertically separated groups into distinct tables.
    """
    page_dict = page.get_text("dict")

    candidate_rows = []
    rows_by_y = {}

    # STEP 1 — Snap line-level objects into common rows
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            line_text = "".join(
                span.get("text", "")
                for span in line.get("spans", [])
            ).strip()

            if not line_text:
                continue

            line_bbox = list(line["bbox"])
            y0 = line_bbox[1]

            matching_y = None

            for existing_y in list(rows_by_y.keys()):
                if abs(y0 - existing_y) < TABLE_CONFIG["TOLERANCE"]:
                    matching_y = existing_y
                    break

            line_object = {
                "text": line_text,
                "bbox": line_bbox
            }

            if matching_y is None:
                rows_by_y[y0] = [line_object]
            else:
                rows_by_y[matching_y].append(line_object)

    # STEP 2 — Keep rows containing multiple candidate columns
    for y_coordinate in rows_by_y:
        row = rows_by_y[y_coordinate]

        if len(row) > 1:
            candidate_rows.append(
                sorted(row, key=lambda item: item["bbox"][0])
            )

    # STEP 3 — Fuzzy repeated horizontal alignment
    final_page_candidates = []

    all_x0_positions = [
        item["bbox"][0]
        for row in candidate_rows
        for item in row
    ]

    for row in candidate_rows:
        is_valid_row = False

        for item in row:
            current_x0 = item["bbox"][0]

            matches = [
                x_position
                for x_position in all_x0_positions
                if abs(x_position - current_x0)
                <= TABLE_CONFIG["X_TOLERANCE"]
            ]

            # More than one match means this x-position repeats
            # elsewhere in the candidate rows.
            if len(matches) > 1:
                is_valid_row = True
                break

        if is_valid_row:
            final_page_candidates.append(row)

    # STEP 4 — Segment candidate rows into separate tables
    page_tables = []

    if final_page_candidates:
        final_page_candidates.sort(
            key=lambda row: row[0]["bbox"][1]
        )

        current_table = [final_page_candidates[0]]

        for row_index in range(1, len(final_page_candidates)):
            previous_bottom = current_table[-1][0]["bbox"][3]
            current_top = final_page_candidates[row_index][0]["bbox"][1]

            if (
                current_top - previous_bottom
                > TABLE_CONFIG["GAP_THRESHOLD"]
            ):
                if len(current_table) >= TABLE_CONFIG["MIN_ROWS"]:
                    page_tables.append(current_table)

                current_table = [final_page_candidates[row_index]]

            else:
                current_table.append(
                    final_page_candidates[row_index]
                )

        if len(current_table) >= TABLE_CONFIG["MIN_ROWS"]:
            page_tables.append(current_table)

    # Adapt detected tables to a standard internal structure
    detected_tables = []

    for table_rows in page_tables:
        table_matrix = geometric_rows_to_matrix(table_rows)

        if not table_matrix:
            continue

        if len(table_matrix) < TABLE_CONFIG["MIN_ROWS"]:
            continue

        if len(table_matrix[0]) < TABLE_CONFIG["MIN_COLUMNS"]:
            continue

        table_bbox = combine_bboxes([
            cell["bbox"]
            for row in table_rows
            for cell in row
        ])

        detected_tables.append({
            "matrix": table_matrix,
            "layout_rows": table_rows,
            "bbox": table_bbox,
            "extraction_method": "geometric_table_detection",
        })

    return detected_tables


#@title CELL 12 — PyMuPDF find_tables fallback

def detect_tables_with_pymupdf_fallback(page):
    """
    Secondary detector.

    This function must only be called when the primary geometric
    detector finds no tables on the page.
    """
    fallback_tables = []

    try:
        table_finder = page.find_tables()

        for found_table in table_finder.tables:
            extracted = found_table.extract()

            table_matrix = normalize_table_matrix(extracted)

            if not table_matrix:
                continue

            if len(table_matrix) < TABLE_CONFIG["MIN_ROWS"]:
                continue

            if len(table_matrix[0]) < TABLE_CONFIG["MIN_COLUMNS"]:
                continue

            table_bbox = None

            if getattr(found_table, "bbox", None) is not None:
                table_bbox = list(found_table.bbox)

            fallback_tables.append({
                "matrix": table_matrix,
                "layout_rows": None,
                "bbox": table_bbox,
                "extraction_method": "pymupdf_find_tables_fallback",
            })

    except Exception as error:
        print(
            f"page.find_tables() failed on page "
            f"{page.number + 1}: {error}"
        )

    return fallback_tables


#@title CELL 13 — Page table detector with conservative header recovery and strict fallback


def first_row_looks_like_data(matrix):
    """
    Return True when the first detected row looks more like data
    than a genuine column-header row.
    """
    matrix = normalize_table_matrix(matrix)

    if not matrix:
        return False

    first_row = matrix[0]

    symbol_cells = sum(
        bool(re.fullmatch(r"[\s+\-–—✓✗xX]+", cell))
        for cell in first_row
        if cell
    )

    numeric_cells = sum(
        bool(re.fullmatch(r"[\d\s.,/%°+\-–—]+", cell))
        for cell in first_row
        if cell
    )

    header_like_cells = sum(
        looks_like_field_label(cell)
        for cell in first_row
        if cell
    )

    # Symbols such as + and - strongly suggest that this is a data row.
    if symbol_cells >= 1:
        return True

    # Mostly numeric/value cells also indicate a data row.
    if numeric_cells >= max(1, len(first_row) // 2):
        return True

    # A row with no recognizable header labels may be data.
    if header_like_cells == 0 and len(first_row) >= 3:
        return True

    return False


def get_table_column_positions(detected_table):
    """
    Obtain geometric column anchors from the original detected rows.
    """
    layout_rows = detected_table.get("layout_rows")

    if not layout_rows:
        return []

    return cluster_column_positions(
        layout_rows,
        TABLE_CONFIG["X_TOLERANCE"]
    )


def find_header_words_above_table(page, table_bbox):
    """
    Find words in a narrow horizontal band immediately above a table.
    """
    if not table_bbox:
        return []

    table_left, table_top, table_right, _ = table_bbox

    search_top = max(
        0,
        table_top - TABLE_CONFIG["HEADER_SEARCH_GAP"]
    )

    words = page.get_text("words")

    candidate_words = []

    for word in words:
        x0, y0, x1, y1, text = word[:5]

        if y1 > table_top:
            continue

        if y0 < search_top:
            continue

        if x1 < table_left or x0 > table_right:
            continue

        text = clean_table_cell(text)

        if not text:
            continue

        candidate_words.append({
            "text": text,
            "bbox": [x0, y0, x1, y1]
        })

    return candidate_words


def group_words_into_rows(words):
    """
    Group nearby words into visual rows.
    """
    grouped_rows = []

    for word in sorted(
        words,
        key=lambda item: (
            item["bbox"][1],
            item["bbox"][0]
        )
    ):
        matching_row = None

        for row in grouped_rows:
            row_y = sum(
                item["bbox"][1]
                for item in row
            ) / len(row)

            if abs(
                word["bbox"][1] - row_y
            ) <= TABLE_CONFIG["HEADER_Y_TOLERANCE"]:
                matching_row = row
                break

        if matching_row is None:
            grouped_rows.append([word])
        else:
            matching_row.append(word)

    for row in grouped_rows:
        row.sort(key=lambda item: item["bbox"][0])

    return grouped_rows


def map_header_words_to_columns(words, column_positions):
    """
    Map recovered header words into the detected table columns.
    """
    if not words or not column_positions:
        return []

    header_row = [""] * len(column_positions)

    for word in words:
        word_x = word["bbox"][0]

        closest_column = min(
            range(len(column_positions)),
            key=lambda index: abs(
                word_x - column_positions[index]
            )
        )

        if header_row[closest_column]:
            header_row[closest_column] += " " + word["text"]
        else:
            header_row[closest_column] = word["text"]

    return [
        clean_table_cell(cell)
        for cell in header_row
    ]


def header_candidate_is_valid(header_row, matrix):
    """
    Validate a recovered header before adding it to the table.

    The validation is intentionally conservative. It accepts short,
    column-aligned labels such as:

        ["", "Milk", "Calf rennet*"]

    but rejects nearby numbered section headings such as:

        ["3. Sterilization", "Compatibility", ""]
        ["6. Performance", "Metrics (Last 12 Months)"]
    """
    if not header_row or not matrix:
        return False

    header_row = [
        clean_table_cell(cell)
        for cell in header_row
    ]

    populated_cells = [
        cell
        for cell in header_row
        if cell
    ]

    # A valid recovered header must label at least two table columns.
    if len(populated_cells) < 2:
        return False

    combined_text = clean_table_cell(
        " ".join(populated_cells)
    )

    if not combined_text:
        return False

    # Reject numbered document/section headings:
    # 3. Sterilization Compatibility
    # 6. Performance Metrics (Last 12 Months)
    # 2.1 Materials
    if re.match(
        r"^\s*\d+(?:\.\d+)*[.)]?\s+",
        combined_text
    ):
        return False

    # Reject common heading prefixes that are not column labels.
    if re.match(
        r"^\s*(section|chapter|appendix|figure|table)\s+"
        r"[A-Za-z0-9IVX.-]+\b",
        combined_text,
        flags=re.IGNORECASE
    ):
        return False

    # Reject long prose or sentence-like text above the table.
    if len(combined_text.split()) > 10:
        return False

    if combined_text.endswith((".", ";")):
        return False

    # Reject a section title split across adjacent columns.
    first_nonempty_cell = next(
        cell
        for cell in header_row
        if cell
    )

    if re.match(
        r"^\s*\d+(?:\.\d+)*[.)]?",
        first_nonempty_cell
    ):
        return False

    # Do not add a duplicate of the first existing row.
    normalized_matrix = normalize_table_matrix(matrix)

    if normalized_matrix and header_row == normalized_matrix[0]:
        return False

    return True


def recover_missing_table_header(page, detected_table):
    """
    Recover a missing header row immediately above a detected table.

    The geometric detector itself is unchanged. This function only
    post-processes its result.
    """
    matrix = normalize_table_matrix(
        detected_table.get("matrix", [])
    )

    if not matrix:
        return detected_table

    if not first_row_looks_like_data(matrix):
        return detected_table

    column_positions = get_table_column_positions(
        detected_table
    )

    if not column_positions:
        return detected_table

    words_above = find_header_words_above_table(
        page,
        detected_table.get("bbox")
    )

    if not words_above:
        return detected_table

    candidate_rows = group_words_into_rows(
        words_above
    )

    if not candidate_rows:
        return detected_table

    table_top = detected_table["bbox"][1]

    # Use the visual row closest to the table.
    candidate_rows.sort(
        key=lambda row: min(
            item["bbox"][1]
            for item in row
        ),
        reverse=True
    )

    for candidate_row in candidate_rows:
        candidate_bottom = max(
            item["bbox"][3]
            for item in candidate_row
        )

        vertical_gap = table_top - candidate_bottom

        if vertical_gap < 0:
            continue

        if vertical_gap > TABLE_CONFIG["HEADER_SEARCH_GAP"]:
            continue

        recovered_header = map_header_words_to_columns(
            candidate_row,
            column_positions
        )

        if not header_candidate_is_valid(
            recovered_header,
            matrix
        ):
            continue

        updated_table = dict(detected_table)

        updated_table["matrix"] = normalize_table_matrix(
            [recovered_header] + matrix
        )

        header_bbox = combine_bboxes([
            item["bbox"]
            for item in candidate_row
        ])

        updated_table["bbox"] = combine_bboxes([
            detected_table.get("bbox"),
            header_bbox
        ])

        updated_table["header_recovered"] = True
        updated_table["recovered_header"] = recovered_header

        return updated_table

    return detected_table


def detect_tables_on_page(page):
    """
    Detect tables using the required hierarchy:

    1. Run the geometric detector.
    2. Recover a nearby missing header when appropriate.
    3. If geometric tables exist, use only those tables.
    4. If none exist, call page.find_tables().
    """

    geometric_tables = detect_geometric_tables_on_page(page)

    if geometric_tables:
        return [
            recover_missing_table_header(
                page,
                detected_table
            )
            for detected_table in geometric_tables
        ]

    return detect_tables_with_pymupdf_fallback(page)


#@title CELL 14 — Convert detected structures into RAG content items

def extract_structured_table_items(pdf_path, document_record):
    """
    Detect table-like structures and convert each result into a unified
    RAG content item.

    Each result is classified as:
    - header_table: first row is a true column header
    - key_value: every row is a field/value pair

    Row definitions:
    - table_total_rows: all rows in the extracted matrix
    - table_header_rows: 1 for header tables, 0 for key–value blocks
    - table_data_rows: rows containing data
    - table_columns: total number of columns
    """

    table_items = []

    document_id = document_record["document_id"]
    file_name = document_record["file"]
    doc_type = document_record["doc_type"]

    document = fitz.open(pdf_path)

    try:
        for page_index, page in enumerate(document):
            page_number = page_index + 1

            # Strict hierarchy remains unchanged:
            # 1. geometric detector
            # 2. page.find_tables() only when geometric detector finds nothing
            detected_tables = detect_tables_on_page(page)

            for table_index, detected_table in enumerate(
                detected_tables,
                start=1
            ):
                table_matrix = normalize_table_matrix(
                    detected_table["matrix"]
                )

                if not table_matrix:
                    continue

                if len(table_matrix) < TABLE_CONFIG["MIN_ROWS"]:
                    continue

                if len(table_matrix[0]) < TABLE_CONFIG["MIN_COLUMNS"]:
                    continue

                (
                    structure_type,
                    embedding_text,
                    markdown_structure
                ) = format_detected_structure(table_matrix)

                if not embedding_text.strip():
                    continue

                extraction_method = detected_table["extraction_method"]

                if structure_type == "key_value":
                    structure_label = "KEY-VALUE BLOCK"
                    table_header_rows = 0
                    table_data_rows = len(table_matrix)
                else:
                    structure_label = "HEADER-BASED TABLE"
                    table_header_rows = 1
                    table_data_rows = max(len(table_matrix) - 1, 0)

                text_for_llm = f"""
[{structure_label}]
Source: {file_name}, page {page_number}
Document type: {doc_type}
Structure number on page: {table_index}
Structure type: {structure_type}
Extraction method: {extraction_method}

{markdown_structure}
""".strip()

                item = create_content_item(
                    document_id=document_id,
                    file_name=file_name,
                    doc_type=doc_type,
                    page_start=page_number,
                    page_end=page_number,
                    content_type="table",
                    text_for_embedding=embedding_text,
                    text_for_llm=text_for_llm,
                    extraction_method=extraction_method,
                    bbox=detected_table.get("bbox"),
                    confidence=None
                )

                item["table_index"] = table_index
                item["table_structure_type"] = structure_type
                item["table_total_rows"] = len(table_matrix)
                item["table_header_rows"] = table_header_rows
                item["table_data_rows"] = table_data_rows
                item["table_columns"] = len(table_matrix[0])
                item["structured_data"] = table_matrix
                item["layout_rows"] = detected_table.get("layout_rows")

                table_items.append(item)

    except Exception as error:
        print(
            f"Could not extract tables from {file_name}: {error}"
        )

    finally:
        document.close()

    return table_items

