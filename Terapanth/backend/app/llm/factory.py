# app/llm/factory.py
# Returns the configured LLMProvider instance.
# Imported by: app/main.py (stored on app.state), app/chat/service.py,
#              app/ingestion/contextual_embedder.py

from functools import lru_cache

from app.config import get_settings
from app.llm.groq_provider import GroqProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.provider import LLMProvider


@lru_cache
def get_llm_provider() -> LLMProvider:
    s = get_settings()
    if s.llm_provider == "groq":
        return GroqProvider(
            llm_model=s.llm_model,
            llm_base_url=s.llm_base_url,
            groq_api_key=s.groq_api_key,
            embedding_model=s.embedding_model,
            embedding_base_url=s.embedding_base_url,
            openai_api_key=s.openai_api_key,
        )
    return OllamaProvider(
        llm_model=s.llm_model,
        llm_base_url=s.llm_base_url,
        embedding_model=s.embedding_model,
        embedding_base_url=s.embedding_base_url,
    )
