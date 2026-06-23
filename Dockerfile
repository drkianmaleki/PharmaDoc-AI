# PharmaDoc AI — multi-format pharmaceutical document RAG system.
FROM python:3.11-slim

# System packages:
#   - tesseract-ocr: required for OCR extraction
#   - libgl1 / libglib2.0-0: required by opencv-python-headless at import time
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# FAISS artifacts are written to rag_artifacts/ relative to the working
# directory by default. Override with PHARMADOC_PERSIST_DIR if needed.
ENV PHARMADOC_PERSIST_DIR=rag_artifacts

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pharmadoc/ ./pharmadoc/

EXPOSE 7860

CMD ["python", "-m", "pharmadoc.app"]
