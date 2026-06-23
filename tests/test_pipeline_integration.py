"""
Integration tests for the document-processing pipeline (ingestion, table
detection, OCR, plot digitization) against the 4 real sample documents.
These need no embedding/generation models -- only the fixture files -- so
they run in any environment that has them copied into tests/fixtures/.

This is the layer that actually broke in a previous (abandoned) conversion
attempt: documents would process "successfully" but the extracted content
silently never reached the rest of the app due to a cross-module global-
state bug (see CONVERSION_NOTES.md). These tests exercise the real
extraction functions end-to-end against real documents to catch exactly
that class of regression.
"""
from pharmadoc.ingestion import (
    build_document_registry,
    extract_selective_ocr_from_pdf,
    extract_plots_from_pdf,
    extract_non_pdf_items,
)
from pharmadoc.text_extractor import extract_digital_text_items
from pharmadoc.tables import extract_structured_table_items


class TestBuildDocumentRegistry:
    def test_all_four_documents_registered(self, fixture_doc_paths):
        registry = build_document_registry(fixture_doc_paths)
        assert len(registry) == 4

    def test_doc_type_classification(self, fixture_doc_paths):
        registry = build_document_registry(fixture_doc_paths)
        by_name = {rec["file"]: rec for rec in registry.values()}
        assert by_name["01_Northbridge_Bioprocess_3pages_page-0001.jpg"]["doc_type"] == "Image Document"
        assert by_name["02_Virelion_Clinical_Trial_7pages.pdf"]["doc_type"] == "Unknown"


def _record_for(registry, filename):
    return next((r for r in registry.values() if r["file"] == filename), None)


class TestNorthbridgeBioprocessJpgs:
    """3 standalone JPG pages -- borderless and bordered tables, all via OCR."""

    def test_page1_borderless_table_burst_pressure(self, fixture_doc_paths):
        registry = build_document_registry(fixture_doc_paths)
        rec = _record_for(registry, "01_Northbridge_Bioprocess_3pages_page-0001.jpg")
        text_items, table_items, ocr_items, plot_items = extract_non_pdf_items(rec["file_path"], rec)
        joined = " ".join(t.get("text_for_embedding", "") for t in table_items + ocr_items)
        assert "1.24 MPa" in joined  # observed burst pressure

    def test_page2_bordered_table_flow_coefficient(self, fixture_doc_paths):
        registry = build_document_registry(fixture_doc_paths)
        rec = _record_for(registry, "01_Northbridge_Bioprocess_3pages_page-0002.jpg")
        text_items, table_items, ocr_items, plot_items = extract_non_pdf_items(rec["file_path"], rec)
        joined = " ".join(t.get("text_for_embedding", "") for t in table_items + ocr_items)
        # NOTE: OCR reads the unit inconsistently on this page ("L/min" ->
        # "Limin"/"Umin"), so we check the number only, not the unit text.
        assert "2.35" in joined  # upper limit, flow coefficient

    def test_page3_bordered_table_port_concentricity(self, fixture_doc_paths):
        registry = build_document_registry(fixture_doc_paths)
        rec = _record_for(registry, "01_Northbridge_Bioprocess_3pages_page-0003.jpg")
        text_items, table_items, ocr_items, plot_items = extract_non_pdf_items(rec["file_path"], rec)
        joined = " ".join(t.get("text_for_embedding", "") for t in table_items + ocr_items)
        assert "0.08 mm" in joined  # port concentricity result


class TestVirelionClinicalTrialPdf:
    """7-page digital PDF: tables, narrative prose, an image-only page, and a chart."""

    def _record(self, fixture_doc_paths):
        registry = build_document_registry(fixture_doc_paths)
        return _record_for(registry, "02_Virelion_Clinical_Trial_7pages.pdf")

    def test_page1_disposition_table(self, fixture_doc_paths):
        rec = self._record(fixture_doc_paths)
        tables = extract_structured_table_items(rec["file_path"], rec)
        joined = " ".join(t.get("text_for_embedding", "") for t in tables)
        assert "218" in joined  # Elarovex 40 mg randomized count

    def test_page2_alpha_allocation_table(self, fixture_doc_paths):
        rec = self._record(fixture_doc_paths)
        tables = extract_structured_table_items(rec["file_path"], rec)
        joined = " ".join(t.get("text_for_embedding", "") for t in tables)
        assert "0.025" in joined

    def test_page1_primary_endpoint_narrative_text(self, fixture_doc_paths):
        rec = self._record(fixture_doc_paths)
        text_items = extract_digital_text_items(rec["file_path"], rec)
        joined = " ".join(t.get("text_for_embedding", "") for t in text_items)
        assert "Neuropathic Symptom Index" in joined

    def test_page4_responder_rate_table(self, fixture_doc_paths):
        rec = self._record(fixture_doc_paths)
        tables = extract_structured_table_items(rec["file_path"], rec)
        joined = " ".join(t.get("text_for_embedding", "") for t in tables)
        assert "67.8%" in joined

    def test_page5_image_only_ocr_page(self, fixture_doc_paths):
        rec = self._record(fixture_doc_paths)
        ocr_text, ocr_tables = extract_selective_ocr_from_pdf(rec["file_path"], rec)
        joined = " ".join(t.get("text_for_embedding", "") for t in ocr_text + ocr_tables)
        assert "0.5%" in joined  # QTcF >480ms, 40 mg group

    def test_page6_adverse_events_table_underlying_data(self, fixture_doc_paths):
        rec = self._record(fixture_doc_paths)
        tables = extract_structured_table_items(rec["file_path"], rec)
        dizziness_rows = [t for t in tables if t.get("structured_data")
                           and t["structured_data"][0][0] == "Dizziness"]
        assert dizziness_rows, "expected a table whose first data row is 'Dizziness'"
        assert dizziness_rows[0]["structured_data"][0] == ["Dizziness", "4.2%", "8.7%", "14.0%"]

    def test_page7_pharmacokinetics_table(self, fixture_doc_paths):
        rec = self._record(fixture_doc_paths)
        tables = extract_structured_table_items(rec["file_path"], rec)
        joined = " ".join(t.get("text_for_embedding", "") for t in tables)
        assert "229 ng/mL" in joined

    def test_page7_benefit_risk_narrative_text(self, fixture_doc_paths):
        rec = self._record(fixture_doc_paths)
        text_items = extract_digital_text_items(rec["file_path"], rec)
        joined = " ".join(t.get("text_for_embedding", "") for t in text_items)
        assert "40 mg regimen demonstrated the greatest efficacy" in joined

    def test_page3_chart_digitization(self, fixture_doc_paths):
        rec = self._record(fixture_doc_paths)
        plot_items = extract_plots_from_pdf(rec["file_path"], rec)
        assert len(plot_items) >= 1
        joined = " ".join(p.get("text_for_embedding", "") for p in plot_items)
        assert "Luminara" in joined
