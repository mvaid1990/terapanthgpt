# backend/tests/test_llm_factory.py
"""Tests that factory returns correct provider based on config."""

from unittest.mock import patch, MagicMock


def test_factory_returns_groq_provider():
    mock_settings = MagicMock()
    mock_settings.llm_provider = "groq"
    mock_settings.llm_model = "llama-3.1-70b-versatile"
    mock_settings.llm_base_url = "https://api.groq.com/openai/v1"
    mock_settings.groq_api_key = "test"
    mock_settings.embedding_base_url = "https://api.openai.com/v1"
    mock_settings.embedding_model = "text-embedding-3-small"
    mock_settings.openai_api_key = "test"

    with patch("app.llm.factory.get_settings", return_value=mock_settings):
        from app.llm import factory
        factory.get_llm_provider.cache_clear()
        from app.llm.groq_provider import GroqProvider
        provider = factory.get_llm_provider()
        assert isinstance(provider, GroqProvider)
        factory.get_llm_provider.cache_clear()


def test_factory_returns_ollama_provider():
    mock_settings = MagicMock()
    mock_settings.llm_provider = "ollama"
    mock_settings.llm_model = "llama3.1:8b"
    mock_settings.llm_base_url = "http://localhost:11434/v1"
    mock_settings.groq_api_key = ""
    mock_settings.embedding_base_url = "http://localhost:11434/v1"
    mock_settings.embedding_model = "nomic-embed-text"
    mock_settings.openai_api_key = ""

    with patch("app.llm.factory.get_settings", return_value=mock_settings):
        from app.llm import factory
        factory.get_llm_provider.cache_clear()
        from app.llm.ollama_provider import OllamaProvider
        provider = factory.get_llm_provider()
        assert isinstance(provider, OllamaProvider)
        factory.get_llm_provider.cache_clear()
