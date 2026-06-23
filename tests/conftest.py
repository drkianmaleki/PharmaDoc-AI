"""
Shared pytest fixtures for the PharmaDoc AI test suite.

The 4 real sample documents live in tests/fixtures/ but are excluded from
git (see .gitignore) -- they are sample documents, not something to
publish. Tests that need them skip cleanly with a clear message if they
are absent, rather than failing or erroring, so the rest of the suite
still runs in CI / on a fresh clone.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _install_offline_fallback_mocks_if_needed():
    """config.py imports sentence_transformers/transformers/openai at
    module load time (verbatim notebook CELL 02), even for tests that
    never touch embeddings. If those packages are genuinely installed
    (the normal case, per requirements.txt), this does nothing at all.
    It only kicks in -- with a loud warning -- in environments with no
    internet access to download those (large) packages, so the rest of
    the test suite can still run. NEVER use this flag to skip real
    model-dependent assertions; needs_models-marked tests still require
    the real packages and real network access."""
    import importlib
    import types

    try:
        importlib.import_module("sentence_transformers")
        return  # real package present, nothing to do
    except ImportError:
        pass

    print(
        "\n[conftest] sentence_transformers not installed -- installing a "
        "lightweight offline stand-in so non-model tests can still run. "
        "Tests marked @pytest.mark.needs_models will still be skipped/"
        "fail honestly, since they need the real package + network access."
    )

    st_mod = types.ModuleType("sentence_transformers")

    class _FakeSentenceTransformer:
        def __init__(self, model_name, *a, **k):
            self.model_name = model_name
            self._dim = 384

        def encode(self, texts, convert_to_numpy=True, show_progress_bar=False,
                   normalize_embeddings=True):
            import numpy as np
            rng = np.random.default_rng(0)
            return rng.random((len(texts), self._dim)).astype("float32")

    st_mod.SentenceTransformer = _FakeSentenceTransformer
    sys.modules["sentence_transformers"] = st_mod

    tf_mod = types.ModuleType("transformers")

    class _FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name, *a, **k):
            return cls()

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, name, *a, **k):
            return cls()

    tf_mod.AutoTokenizer = _FakeTokenizer
    tf_mod.AutoModelForSeq2SeqLM = _FakeModel
    sys.modules["transformers"] = tf_mod

    oa_mod = types.ModuleType("openai")

    class _FakeOpenAI:
        def __init__(self, *a, **k):
            pass

    oa_mod.OpenAI = _FakeOpenAI
    sys.modules["openai"] = oa_mod


_install_offline_fallback_mocks_if_needed()

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Real, empirically-verified doc_type values -- confirmed by actually
# running build_document_registry() against these files, not assumed.
# Northbridge JPGs classify as 'Image Document' (no text sample exists
# before OCR runs, so the PDF-text-based detect_doc_type() rules never
# apply to images). The Virelion PDF classifies as 'Unknown' because its
# title text ("Phase III Clinical Study Summary") doesn't match any of
# detect_doc_type()'s exact keyword phrases (e.g. "phase iii clinical
# trial").
REQUIRED_TEST_DOCUMENTS = {
    "01_Northbridge_Bioprocess_3pages_page-0001.jpg": "Image Document",
    "01_Northbridge_Bioprocess_3pages_page-0002.jpg": "Image Document",
    "01_Northbridge_Bioprocess_3pages_page-0003.jpg": "Image Document",
    "02_Virelion_Clinical_Trial_7pages.pdf": "Unknown",
}


def _fixture_paths():
    return [FIXTURES_DIR / name for name in REQUIRED_TEST_DOCUMENTS]


@pytest.fixture(scope="session")
def fixture_doc_paths():
    """The 4 real test documents, as absolute paths. Skips the test if
    any are missing (e.g. on a fresh clone where they were never copied
    into tests/fixtures/ locally -- they are gitignored on purpose)."""
    paths = _fixture_paths()
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "Real test documents not present in tests/fixtures/ (gitignored, "
            f"not part of the repo): missing {missing}. Copy the 4 sample "
            "documents into tests/fixtures/ locally to run this test."
        )
    return [str(p) for p in paths]


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_models: requires real embedding/generation model downloads"
    )
