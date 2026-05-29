from __future__ import annotations

from src.schemas import UploadedDocument


def decode_uploaded_txt(filename: str, content: bytes) -> UploadedDocument:
    """Decode an uploaded .txt file into a validated document object."""

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1", errors="ignore")

    text = text.strip()

    if not text:
        raise ValueError(f"{filename} is empty or could not be decoded.")

    return UploadedDocument(filename=filename, text=text)


def load_uploaded_txt_files(uploaded_files) -> list[UploadedDocument]:
    """Convert Streamlit uploaded files into validated text documents."""

    documents: list[UploadedDocument] = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        content = uploaded_file.getvalue()
        documents.append(decode_uploaded_txt(filename, content))

    return documents
