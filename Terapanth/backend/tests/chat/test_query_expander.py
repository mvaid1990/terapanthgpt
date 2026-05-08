# backend/tests/chat/test_query_expander.py
"""Tests for LLM-based query expansion."""
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_expander_returns_list_of_strings():
    from app.chat.query_expander import expand_query

    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(
        return_value='["What is non-absolutism in Jainism?", "Explain many-sidedness doctrine", "Jain epistemology overview"]'
    )

    results = await expand_query("What is Anekantavada?", llm=mock_llm)
    assert isinstance(results, list)
    assert len(results) >= 1
    assert all(isinstance(r, str) for r in results)


@pytest.mark.asyncio
async def test_expander_falls_back_on_malformed_json():
    from app.chat.query_expander import expand_query

    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(return_value="not valid json at all")

    results = await expand_query("What is Anekantavada?", llm=mock_llm)
    assert results == ["What is Anekantavada?"]
