# backend/tests/retrieval/test_reranker.py
"""Tests for Cohere reranker with trust_score boost."""
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.retrieval.models import RetrievedChunk, RerankResult


def make_chunk(content: str, trust_score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=content,
        context="Context.",
        doc_title="Test Doc",
        doc_author=None,
        trust_score=trust_score,
        score=0.5,
    )


@pytest.mark.asyncio
async def test_rerank_returns_top_k_results():
    from app.retrieval.reranker import rerank

    chunks = [make_chunk(f"chunk {i}") for i in range(10)]

    with patch("app.retrieval.reranker.cohere.AsyncClientV2") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.rerank = AsyncMock(return_value=MagicMock(
            results=[
                MagicMock(index=i, relevance_score=0.9 - i * 0.05)
                for i in range(10)
            ]
        ))

        results = await rerank(
            query="What is Anekantavada?",
            chunks=chunks,
            top_k=5,
            cohere_api_key="test",
        )

    assert len(results) == 5
    assert all(isinstance(r, RerankResult) for r in results)


@pytest.mark.asyncio
async def test_trust_score_boosts_rerank_score():
    from app.retrieval.reranker import rerank

    low_trust = make_chunk("chunk low", trust_score=0.5)
    high_trust = make_chunk("chunk high", trust_score=1.0)

    with patch("app.retrieval.reranker.cohere.AsyncClientV2") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.rerank = AsyncMock(return_value=MagicMock(
            results=[
                MagicMock(index=0, relevance_score=0.8),
                MagicMock(index=1, relevance_score=0.8),
            ]
        ))

        results = await rerank(
            query="test",
            chunks=[low_trust, high_trust],
            top_k=2,
            cohere_api_key="test",
        )

    scores = {r.chunk.content: r.rerank_score for r in results}
    assert scores["chunk high"] > scores["chunk low"]


@pytest.mark.asyncio
async def test_rerank_returns_empty_for_no_chunks():
    from app.retrieval.reranker import rerank

    results = await rerank(query="test", chunks=[], top_k=5, cohere_api_key="test")
    assert results == []
