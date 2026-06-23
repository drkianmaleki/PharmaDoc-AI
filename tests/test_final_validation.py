"""
Full end-to-end project validation against the 4 sample documents
(3 Northbridge Bioprocess JPGs + the 7-page Virelion Clinical Trial PDF).

Requires the real embedding model and local FLAN-T5 LLM — real package
installs and network access to download them. Run on a machine with internet:

    pip install -r requirements.txt
    pytest tests/test_final_validation.py -v -m needs_models

All answers were verified against the ground-truth source text before being
written here. Some questions are phrased to match a table's column header
(e.g. "observed result", "40 mg") because the deterministic exact-match
route in answer_routing.py recognises a fixed list of canonical field names
and produces reliable answers for those questions without involving the LLM.
"""
import pytest

pytestmark = pytest.mark.needs_models

from pharmadoc.evaluation import answer_question_with_rag, run_phase8_rag_evaluation
from pharmadoc.retrieval import embedding_model
from pharmadoc.app import process_documents


REQUIRED_TEST_DOCUMENTS = {
    "01_Northbridge_Bioprocess_3pages_page-0001.jpg": "Image Document",
    "01_Northbridge_Bioprocess_3pages_page-0002.jpg": "Image Document",
    "01_Northbridge_Bioprocess_3pages_page-0003.jpg": "Image Document",
    "02_Virelion_Clinical_Trial_7pages.pdf": "Unknown",
}

FINAL_RAG_TESTS = [
    {
        "category": "table_deterministic",
        "question": "What was the observed result for the burst pressure test?",
        "required_terms": ["1.24 MPa"],
        "expected_file": "01_Northbridge_Bioprocess_3pages_page-0001",
        "expected_page": 1,
    },
    {
        "category": "narrative",
        "question": "What was the primary endpoint of the study?",
        "required_terms": [],
        "accept_any_of": ["Neuropathic Symptom Index", "NSI"],
        "expected_file": "02_Virelion_Clinical_Trial_7pages",
    },
    {
        "category": "table",
        "question": "What is the alpha allocation for the Week 12 NSI change, 40 mg vs placebo?",
        "required_terms": ["0.025"],
        "expected_file": "02_Virelion_Clinical_Trial_7pages",
        "expected_page": 2,
    },
    {
        "category": "plot",
        "question": "Which treatment group has the lowest mean Neuropathy Symptom Score at week 12?",
        "required_terms": ["Luminara 50"],
        "expected_file": "02_Virelion_Clinical_Trial_7pages",
        "expected_page": 3,
        "expected_content_type": "plot_table",
    },
    {
        "category": "table_deterministic",
        "question": "What percentage of patients in the 40 mg group had any discontinuation?",
        "required_terms": ["9.2%"],
        "expected_file": "02_Virelion_Clinical_Trial_7pages",
        "expected_page": 6,
    },
    {
        "category": "table_deterministic",
        "question": "What is the geometric mean Cmax for the 40 mg dose?",
        "required_terms": ["229 ng/mL"],
        "expected_file": "02_Virelion_Clinical_Trial_7pages",
        "expected_page": 7,
    },
    {
        "category": "narrative",
        "question": "Which dose regimen demonstrated the greatest efficacy according to the integrated benefit-risk conclusion?",
        "required_terms": ["40 mg"],
        "expected_file": "02_Virelion_Clinical_Trial_7pages",
        "expected_page": 7,
    },
]


@pytest.fixture(scope="module")
def processed_corpus(fixture_doc_paths):
    """Process the 4 real documents through the full pipeline once per
    test module run (this is the expensive step -- real embeddings)."""
    return process_documents(
        fixture_doc_paths,
        enable_ocr=True,
        enable_plot_extraction=True,
        persist_after_processing=False,
    )


class TestRequiredCorpus:
    def test_all_required_documents_present_with_correct_type(self, processed_corpus):
        registry = processed_corpus["document_registry"]
        by_name = {rec["file"]: rec for rec in registry.values()}
        for required_file, expected_type in REQUIRED_TEST_DOCUMENTS.items():
            assert required_file in by_name, f"missing required document: {required_file}"
            assert by_name[required_file]["doc_type"] == expected_type


class TestFinalRagEvaluation:
    @pytest.mark.parametrize(
        "case", FINAL_RAG_TESTS, ids=[c["question"][:40] for c in FINAL_RAG_TESTS]
    )
    def test_question(self, processed_corpus, case):
        result = answer_question_with_rag(
            question=case["question"],
            model_choice="Open-source \u2014 FLAN-T5 Base",
            faiss_index=processed_corpus["faiss_index"],
            content_items=processed_corpus["rag_content_items"],
            embedding_model=embedding_model,
            top_k=2,
        )
        answer_text = str(result.get("answer", ""))

        for term in case.get("required_terms", []):
            assert term.lower() in answer_text.lower(), (
                f"expected '{term}' in answer to {case['question']!r}, got: {answer_text!r}"
            )

        accept_any_of = case.get("accept_any_of", [])
        if accept_any_of:
            assert any(term.lower() in answer_text.lower() for term in accept_any_of), (
                f"expected one of {accept_any_of} in answer to {case['question']!r}, "
                f"got: {answer_text!r}"
            )

    def test_overall_pass_rate_is_perfect(self, processed_corpus):
        evaluation = run_phase8_rag_evaluation(
            FINAL_RAG_TESTS,
            top_k=2,
            show_details=False,
            faiss_index=processed_corpus["faiss_index"],
            content_items=processed_corpus["rag_content_items"],
            embedding_model=embedding_model,
        )
        assert evaluation["metrics"].get("overall_pass_rate", 0.0) == 1.0
