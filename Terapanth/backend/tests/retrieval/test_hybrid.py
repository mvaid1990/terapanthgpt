# backend/tests/retrieval/test_hybrid.py
"""Tests for hybrid retrieval merging vector + BM25 results."""
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.retrieval.models import RetrievedChunk, RerankResult


def make_chunk(chunk_id: uuid.UUID, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=uuid.uuid4(),
        content=content,
        context="ctx",
        doc_title="Doc",
        doc_author=None,
        trust_score=0.8,
        score=0.6,
    )


@pytest.mark.asyncio
async def test_hybrid_deduplicates_results():
    from app.retrieval.hybrid import hybrid_retrieve

    shared_id = uuid.uuid4()
    chunk_a = make_chunk(shared_id, "shared chunk")
    chunk_b = make_chunk(uuid.uuid4(), "unique chunk")

    mock_db = AsyncMock()

    with patch("app.retrieval.hybrid.vector_search", new_callable=AsyncMock) as mock_vec, \
         patch("app.retrieval.hybrid.bm25_search", new_callable=AsyncMock) as mock_bm25, \
         patch("app.retrieval.hybrid.rerank", new_callable=AsyncMock) as mock_rerank:

        mock_vec.return_value = [chunk_a]
        mock_bm25.return_value = [chunk_a, chunk_b]
        mock_rerank.return_value = [
            RerankResult(chunk=chunk_a, rerank_score=0.9),
            RerankResult(chunk=chunk_b, rerank_score=0.7),
        ]

        results = await hybrid_retrieve(
            query="test",
            query_embedding=[0.0] * 1536,
            db=mock_db,
            cohere_api_key="test",
            top_k=5,
        )

    rerank_call_chunks = mock_rerank.call_args[1]["chunks"]
    ids = [c.chunk_id for c in rerank_call_chunks]
    assert len(ids) == len(set(str(i) for i in ids))
