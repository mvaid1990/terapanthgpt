# backend/tests/test_config.py
"""Tests that Settings loads and validates environment variables correctly."""

import pytest
from pydantic import ValidationError


def test_settings_loads_with_required_fields(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "llama-3.1-70b-versatile")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RERANKER_PROVIDER", "cohere")
    monkeypatch.setenv("COHERE_API_KEY", "test-key")

    import importlib
    import app.config as cfg_module
    importlib.reload(cfg_module)
    cfg_module.get_settings.cache_clear()

    s = cfg_module.Settings()
    assert s.llm_provider == "groq"
    assert s.max_upload_bytes == 52428800


def test_settings_rejects_unknown_llm_provider(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("LLM_PROVIDER", "invalid")
    monkeypatch.setenv("LLM_MODEL", "model")
    monkeypatch.setenv("LLM_BASE_URL", "http://example.com")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://example.com")
    monkeypatch.setenv("EMBEDDING_MODEL", "model")

    import importlib
    import app.config as cfg_module
    importlib.reload(cfg_module)

    with pytest.raises(ValidationError):
        cfg_module.Settings()
