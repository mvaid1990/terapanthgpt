# app/ingestion/extractor.py
# Extracts raw text from uploaded PDF or plain-text bytes.
# Called by: app/ingestion/service.py

import fitz  # PyMuPDF

from app.core.exceptions import DocumentProcessingError

_SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def extract_text(content: bytes, filename: str) -> str:
    """Return plain text extracted from file bytes.

    Args:
        content: raw file bytes
        filename: original filename, used to detect format

    Raises:
        DocumentProcessingError: if format unsupported or extraction fails
    """
    ext = _get_extension(filename)
    if ext not in _SUPPORTED_EXTENSIONS:
        raise DocumentProcessingError(
            f"Unsupported file type '{ext}'. Supported: {_SUPPORTED_EXTENSIONS}"
        )
    if ext == ".txt":
        return content.decode("utf-8", errors="replace")
    return _extract_pdf(content)


def _get_extension(filename: str) -> str:
    dot = filename.rfind(".")
    if dot == -1:
        return ""
    return filename[dot:].lower()


def _extract_pdf(content: bytes) -> str:
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        pages = [page.get_text() for page in doc]
        return "\n".join(pages)
    except Exception as exc:
        raise DocumentProcessingError(f"PDF extraction failed: {exc}") from exc
