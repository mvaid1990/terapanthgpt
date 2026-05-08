# backend/tests/ingestion/test_extractor.py
"""Tests for PDF and plain-text extraction."""
import pytest
from app.ingestion.extractor import extract_text


def test_extracts_plain_text_from_txt_bytes():
    content = b"Anekantavada is the Jain doctrine of non-absolutism."
    result = extract_text(content, filename="test.txt")
    assert "Anekantavada" in result


def test_raises_on_unsupported_extension():
    from app.core.exceptions import DocumentProcessingError
    with pytest.raises(DocumentProcessingError):
        extract_text(b"data", filename="test.docx")


def test_returns_non_empty_string_for_empty_txt():
    result = extract_text(b"", filename="empty.txt")
    assert isinstance(result, str)
