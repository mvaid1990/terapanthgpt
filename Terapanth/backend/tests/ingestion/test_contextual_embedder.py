# backend/tests/ingestion/test_contextual_embedder.py
"""Tests for contextual embedding generation."""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_llm():
    provider = MagicMock()
    provider.chat = AsyncMock(return_value="This chunk describes the doctrine of Anekantavada.")
    provider.embed = AsyncMock(return_value=[0.1] * 1536)
    return provider


@pytest.mark.asyncio
async def test_returns_context_and_embedding(mock_llm):
    from app.ingestion.contextual_embedder import build_contextual_embedding
    context, embedding = await build_contextual_embedding(
        chunk="Anekantavada means many-sidedness.",
        doc_title="Tattvartha Sutra",
        doc_author="Umasvati",
        doc_topic="philosophy",
        llm=mock_llm,
    )
    assert "Anekantavada" in context
    assert len(embedding) == 1536


@pytest.mark.asyncio
async def test_embed_called_with_context_prepended(mock_llm):
    from app.ingestion.contextual_embedder import build_contextual_embedding
    await build_contextual_embedding(
        chunk="Anekantavada means many-sidedness.",
        doc_title="Tattvartha Sutra",
        doc_author="Umasvati",
        doc_topic="philosophy",
        llm=mock_llm,
    )
    embed_call_arg = mock_llm.embed.call_args[0][0]
    assert "Anekantavada" in embed_call_arg
    assert mock_llm.chat.return_value in embed_call_arg
