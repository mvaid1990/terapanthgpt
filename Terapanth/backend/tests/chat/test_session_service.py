# backend/tests/chat/test_session_service.py
"""Tests for session CRUD and sliding window history."""
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_get_or_create_session_creates_new():
    from app.chat.session_service import get_or_create_session
    mock_db = AsyncMock()
    mock_db.get = AsyncMock(return_value=None)

    session_id = uuid.uuid4()
    await get_or_create_session(session_id=session_id, db=mock_db)

    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_get_or_create_session_returns_existing():
    from app.chat.session_service import get_or_create_session
    from app.db.models import Session as SessionModel

    existing = MagicMock(spec=SessionModel)
    existing.id = uuid.uuid4()

    mock_db = AsyncMock()
    mock_db.get = AsyncMock(return_value=existing)

    session = await get_or_create_session(session_id=existing.id, db=mock_db)
    assert session is existing
    mock_db.add.assert_not_called()


@pytest.mark.asyncio
async def test_get_window_returns_last_10_messages():
    from app.chat.session_service import get_sliding_window
    from app.db.models import Message

    msgs = [
        MagicMock(spec=Message, role="user" if i % 2 == 0 else "assistant", content=f"msg{i}")
        for i in range(15)
    ]

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = msgs[-10:]
    mock_db.execute = AsyncMock(return_value=mock_result)

    window = await get_sliding_window(conversation_id=uuid.uuid4(), db=mock_db)
    assert len(window) == 10
