# backend/tests/ingestion/test_chunker.py
"""Tests for token-aware text chunking."""
import pytest
from app.ingestion.chunker import chunk_text


def test_short_text_returns_single_chunk():
    chunks = chunk_text("Short text.", chunk_size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."


def test_long_text_splits_into_multiple_chunks():
    long_text = " ".join(["word"] * 600)
    chunks = chunk_text(long_text, chunk_size=512, overlap=64)
    assert len(chunks) > 1


def test_chunks_overlap():
    long_text = " ".join([f"word{i}" for i in range(600)])
    chunks = chunk_text(long_text, chunk_size=512, overlap=64)
    last_words_of_first = chunks[0].split()[-5:]
    assert any(w in chunks[1] for w in last_words_of_first)


def test_empty_text_returns_empty_list():
    assert chunk_text("", chunk_size=512, overlap=64) == []
