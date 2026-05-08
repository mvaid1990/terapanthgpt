# backend/tests/chat/test_service.py
"""Tests for chat pipeline orchestration."""
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.retrieval.models import RetrievedChunk, RerankResult


def make_rerank_result(content: str) -> RerankResult:
    chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=content,
        context="ctx",
        doc_title="Tattvartha Sutra",
        doc_author="Umasvati",
        trust_score=0.9,
        score=0.8,
    )
    return RerankResult(chunk=chunk, rerank_score=0.9)


@pytest.mark.asyncio
async def test_chat_service_returns_answer_with_citations():
    from app.chat.service import answer_question
    from app.chat.models import ChatResponse

    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(
        side_effect=[
            '["What is non-absolutism?"]',
            '{"answer": "Anekantavada means...", "follow_ups": ["Tell me more"]}',
        ]
    )
    mock_llm.embed = AsyncMock(return_value=[0.0] * 1536)

    mock_db = AsyncMock()
    session_id = uuid.uuid4()
    conv_id = uuid.uuid4()

    retrieval_results = [make_rerank_result("Anekantavada passage")]

    with patch("app.chat.service.get_or_create_session", new_callable=AsyncMock) as mock_sess, \
         patch("app.chat.service.get_or_create_conversation", new_callable=AsyncMock) as mock_conv, \
         patch("app.chat.service.get_sliding_window", new_callable=AsyncMock, return_value=[]), \
         patch("app.chat.service.save_message", new_callable=AsyncMock) as mock_save, \
         patch("app.chat.service.hybrid_retrieve", new_callable=AsyncMock, return_value=retrieval_results):

        mock_sess.return_value = MagicMock(id=session_id)
        mock_conv.return_value = MagicMock(id=conv_id)
        saved_msg = MagicMock(
            id=uuid.uuid4(),
            role="assistant",
            content="Anekantavada means...",
            citations=[{"title": "Tattvartha Sutra", "author": "Umasvati", "chunk_ref": "abc"}],
            follow_ups=["Tell me more"],
        )
        mock_save.return_value = saved_msg

        response = await answer_question(
            session_id=session_id,
            message="What is Anekantavada?",
            conversation_id=None,
            db=mock_db,
            llm=mock_llm,
            cohere_api_key="test",
        )

    assert isinstance(response, ChatResponse)
    assert response.conversation_id == conv_id
