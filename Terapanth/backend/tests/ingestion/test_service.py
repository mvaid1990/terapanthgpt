# backend/tests/ingestion/test_service.py
"""Tests for the ingestion pipeline orchestration."""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_llm():
    provider = MagicMock()
    provider.chat = AsyncMock(return_value="Summary of chunk.")
    provider.embed = AsyncMock(return_value=[0.0] * 1536)
    return provider


@pytest.fixture
def mock_db():
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_process_document_creates_chunks(mock_db, mock_llm):
    from app.ingestion.service import process_approved_document
    from app.db.models import Document

    doc = Document(
        id=uuid.uuid4(),
        title="Tattvartha Sutra",
        author="Umasvati",
        topic="philosophy",
        trust_score=0.9,
        status="approved",
        file_path="/uploads/test.txt",
    )

    with patch("app.ingestion.service.extract_text", return_value="word " * 100):
        with patch("app.ingestion.service.chunk_text", return_value=["chunk one", "chunk two"]):
            await process_approved_document(doc=doc, db=mock_db, llm=mock_llm)

    assert mock_llm.embed.call_count == 2
    assert mock_db.add.call_count == 2
