"""
Configuration, constants, and runtime environment for PharmaDoc AI.

All tuneable parameters are centralised here: embedding model, chunking
settings, LLM model catalog, supported content types, and the persistence
directory. Override PHARMADOC_PERSIST_DIR to change where FAISS artifacts
are written.
"""



import os
import re
import io
import json
import math
import time
import uuid
import hashlib
import shutil
import platform
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
from importlib.metadata import version, PackageNotFoundError

import fitz
import faiss
import numpy as np
import pandas as pd
import gradio as gr
import cv2
import pytesseract

from PIL import Image
from docx import Document
from sentence_transformers import SentenceTransformer


def get_installed_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


DEPENDENCY_VERSIONS = {
    "Python": platform.python_version(),
    "PyMuPDF": get_installed_version("PyMuPDF"),
    "sentence-transformers": get_installed_version("sentence-transformers"),
    "faiss-cpu": get_installed_version("faiss-cpu"),
    "transformers": get_installed_version("transformers"),
    "accelerate": get_installed_version("accelerate"),
    "gradio": get_installed_version("gradio"),
    "pandas": get_installed_version("pandas"),
    "openpyxl": get_installed_version("openpyxl"),
    "python-docx": get_installed_version("python-docx"),
    "Pillow": get_installed_version("Pillow"),
    "pytesseract": get_installed_version("pytesseract"),
    "opencv-python-headless": get_installed_version("opencv-python-headless"),
    "openai": get_installed_version("openai"),
}

print("Runtime dependency versions")
for package_name, package_version in DEPENDENCY_VERSIONS.items():
    print(f"- {package_name}: {package_version}")



EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_LLM_MODEL_NAME = "google/flan-t5-base"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 2
MAX_UPLOAD_MB = 100
OCR_MIN_DIGITAL_CHARS = 60
OCR_RENDER_DPI = 200
OCR_MIN_CONFIDENCE = 35.0
PLOT_SAMPLE_POINTS = 30
PERSIST_DIR = Path(os.getenv("PHARMADOC_PERSIST_DIR", "rag_artifacts"))

SUPPORTED_CONTENT_TYPES = [
    "text",
    "table",
    "ocr_text",
    "ocr_table",
    "plot_table",
]

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".csv", ".xlsx",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
}

SUPPORTED_DOC_TYPES = [
    "Certificate of Analysis",
    "Safety Data Sheet",
    "Specification",
    "Declaration",
    "Spreadsheet",
    "Text Document",
    "Image Document",
    "Unknown",
]

TABLE_CONFIG = {
    "TOLERANCE": 3,
    "X_TOLERANCE": 10,
    "GAP_THRESHOLD": 20,
    "MIN_ROWS": 2,
    "MIN_COLUMNS": 2,
}

MODEL_CATALOG = {
    "Open-source — FLAN-T5 Base": {
        "provider": "local_huggingface",
        "model_name": LOCAL_LLM_MODEL_NAME,
        "requires_api_key": False,
        "assignment_compliant": True,
    },
    "OpenAI — GPT": {
        "provider": "openai",
        "model_name": "gpt-4o-mini",
        "requires_api_key": True,
        "assignment_compliant": False,
    },
}

DEFAULT_MODEL_CHOICE = "Open-source — FLAN-T5 Base"

file_paths = []
document_registry = {}
all_content_items = []
chunked_content_items = []
all_table_items = []
all_ocr_items = []
all_plot_items = []
rag_content_items = []
faiss_index = None
chunk_embeddings = None
embedding_model = None
_local_generator = None
document_centroids = {}

