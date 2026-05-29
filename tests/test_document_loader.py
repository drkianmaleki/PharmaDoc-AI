from io import BytesIO

from src.document_loader import decode_uploaded_txt


def test_decode_uploaded_txt():
    document = decode_uploaded_txt("sample.txt", b"hello world")

    assert document.filename == "sample.txt"
    assert document.text == "hello world"
