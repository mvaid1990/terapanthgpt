# app/ingestion/chunker.py
# Splits raw document text into token-aware overlapping chunks.
# Called by: app/ingestion/service.py

import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping token chunks.

    Respects token boundaries. chunk_size and overlap are in tokens.
    Returns [] for empty input.
    """
    if not text.strip():
        return []

    tokens = _ENCODER.encode(text)
    chunks: list[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_ENCODER.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += chunk_size - overlap

    return chunks
