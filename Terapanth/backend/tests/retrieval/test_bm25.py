# backend/tests/retrieval/test_bm25.py
"""Tests for PostgreSQL full-text search."""
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_bm25_search_returns_retrieved_chunks():
    from app.retrieval.bm25 import bm25_search
    from app.retrieval.models import RetrievedChunk

    mock_db = AsyncMock()
    row = MagicMock()
    row.chunk_id = uuid.uuid4()
    row.document_id = uuid.uuid4()
    row.content = "Syadvada means conditional predication."
    row.context = "This describes Syadvada."
    row.doc_title = "Niyamasara"
    row.doc_author = None
    row.trust_score = 0.8
    row.score = 0.5

    mock_result = MagicMock()
    mock_result.all.return_value = [row]
    mock_db.execute = AsyncMock(return_value=mock_result)

    results = await bm25_search(query="Syadvada", db=mock_db, top_k=20)
    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)
