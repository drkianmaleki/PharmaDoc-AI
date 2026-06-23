"""
Unit tests for standalone helper functions -- no models, no PDFs, no
network access needed. These exercise the converted package's pure-logic
helpers directly, the same way chatbot B's test suite tests its own
helpers, but against chatbot A's actual (verbatim) implementations.
"""
from pharmadoc.tables import (
    clean_table_cell,
    normalize_table_matrix,
    classify_table_structure,
    table_to_markdown,
)
from pharmadoc.text_extractor import split_text_with_overlap
from pharmadoc.metadata import detect_doc_type, create_content_item


class TestCleanTableCell:
    def test_strips_whitespace(self):
        assert clean_table_cell("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self):
        assert clean_table_cell("hello   world\n\tfoo") == "hello world foo"

    def test_none_becomes_empty_string(self):
        assert clean_table_cell(None) == ""

    def test_non_string_input_is_stringified(self):
        assert clean_table_cell(42) == "42"


class TestNormalizeTableMatrix:
    def test_cleans_every_cell(self):
        matrix = [["  a  ", "b\n\n"], [None, "  c"]]
        result = normalize_table_matrix(matrix)
        assert result == [["a", "b"], ["", "c"]]


class TestClassifyTableStructure:
    def test_header_table_with_multiple_data_rows(self):
        matrix = [
            ["Part Number", "Description"],
            ["29477427", "High Flow Kit F"],
            ["29184612", "High Flow Gradient C"],
        ]
        structure_type = classify_table_structure(matrix)
        assert structure_type == "header_table"

    def test_two_column_label_value_pairs_classified_as_key_value(self):
        matrix = [
            ["Lot Number", "18356721"],
            ["Product Article Number", "28 9301 82"],
            ["Date of Manufacture", "20240315"],
        ]
        structure_type = classify_table_structure(matrix)
        assert structure_type == "key_value"


class TestTableToMarkdown:
    def test_produces_pipe_delimited_markdown(self):
        matrix = [["A", "B"], ["1", "2"]]
        md = table_to_markdown(matrix)
        assert "| A | B |" in md
        assert "| 1 | 2 |" in md


class TestSplitTextWithOverlap:
    def test_short_text_returns_single_chunk(self):
        chunks = split_text_with_overlap("short text", chunk_size=1000, chunk_overlap=150)
        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_long_text_is_split_into_multiple_overlapping_chunks(self):
        text = "word " * 500  # well over chunk_size
        chunks = split_text_with_overlap(text, chunk_size=1000, chunk_overlap=150)
        assert len(chunks) > 1
        # consecutive chunks should share overlapping content
        assert chunks[0][-50:] in (chunks[0] + chunks[1])


class TestDetectDocType:
    def test_certificate_of_analysis_detected(self):
        sample = "CERTIFICATE OF ANALYSIS\nLot/Batchnumber: 10DWPXK\n"
        assert detect_doc_type(sample, "coa.pdf") == "Certificate of Analysis"

    def test_safety_data_sheet_detected(self):
        sample = "SAFETY DATA SHEET\nHazards Identification\n"
        assert detect_doc_type(sample, "safety.pdf") == "Safety Data Sheet"

    def test_unknown_falls_back_gracefully(self):
        result = detect_doc_type("completely unrelated content xyz", "mystery.pdf")
        assert isinstance(result, str)


class TestCreateContentItem:
    def test_required_fields_present(self):
        item = create_content_item(
            document_id="doc_001",
            file_name="example.pdf",
            doc_type="Specification",
            page_start=1,
            page_end=1,
            content_type="text",
            text_for_embedding="some text",
            text_for_llm="[TEXT]\nsome text",
            extraction_method="digital_text",
        )
        assert item["document_id"] == "doc_001"
        assert item["content_type"] == "text"
        assert "source_id" in item
