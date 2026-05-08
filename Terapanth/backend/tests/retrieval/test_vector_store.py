# backend/tests/retrieval/test_vector_store.py
"""Tests for pgvector similarity search."""
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_vector_search_returns_retrieved_chunks():
    from app.retrieval.vector_store import vector_search
    from app.retrieval.models import RetrievedChunk

    mock_db = AsyncMock()
    chunk_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    row = MagicMock()
    row.chunk_id = chunk_id
    row.document_id = doc_id
    row.content = "Anekantavada is the doctrine of many-sidedness."
    row.context = "This describes Anekantavada."
    row.doc_title = "Tattvartha Sutra"
    row.doc_author = "Umasvati"
    row.trust_score = 0.9
    row.score = 0.87

    mock_result = MagicMock()
    mock_result.all.return_value = [row]
    mock_db.execute = AsyncMock(return_value=mock_result)

    results = await vector_search(
        query_embedding=[0.1] * 1536, db=mock_db, top_k=20
    )
    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].content == "Anekantavada is the doctrine of many-sidedness."
