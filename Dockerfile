# PharmaDoc AI -- multi-format RAG chatbot with OCR, table extraction,
# plot digitization, and evidence-gated answer generation.
FROM python:3.11-slim

# System packages:
#   - tesseract-ocr: matches the notebook's CELL 01 apt install exactly
#   - libgl1 / libglib2.0-0: required by opencv-python-headless at import
#     time even in headless mode, on a minimal Debian base
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# config.py (verbatim notebook CELL 03) hardcodes:
#     PERSIST_DIR = Path("/content/rag_artifacts")
# That line is intentionally left byte-identical to the original notebook
# (see CONVERSION_NOTES.md). Rather than edit the source, the directory
# is created here so the unmodified code works as-is in this container.
RUN mkdir -p /content/rag_artifacts

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pharmadoc/ ./pharmadoc/

EXPOSE 7860

# app.py's launch() call is guarded behind `if __name__ == "__main__":`
# (an approved, documented change -- see CONVERSION_NOTES.md), so the
# module can be run directly as the container's entry point.
CMD ["python", "-m", "pharmadoc.app"]
