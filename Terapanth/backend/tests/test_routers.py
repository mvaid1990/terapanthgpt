# backend/tests/test_routers.py
"""HTTP-level integration tests for key endpoints."""
import pytest
import uuid
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_health_endpoint():
    from app.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_chat_requires_session_header():
    from app.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/chat", json={"message": "hello"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_conversations_returns_empty_for_new_session():
    from app.main import app
    from app.db.base import get_db

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute = AsyncMock(return_value=mock_result)

    app.dependency_overrides[get_db] = lambda: mock_db

    session_id = str(uuid.uuid4())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/api/v1/conversations",
            headers={"X-Session-ID": session_id},
        )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == []
