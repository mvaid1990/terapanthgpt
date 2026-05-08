# app/config.py
# Application settings loaded from environment variables.
# Imported by: app/main.py, app/db/base.py, app/llm/factory.py,
#              app/retrieval/reranker.py, app/ingestion/service.py

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    llm_provider: Literal["groq", "ollama"]
    llm_model: str
    llm_base_url: str
    groq_api_key: str = ""
    embedding_base_url: str
    embedding_model: str
    openai_api_key: str = ""
    jina_api_key: str = ""
    upload_dir: str = "./uploads"
    max_upload_bytes: int = 52_428_800  # 50 MB


@lru_cache
def get_settings() -> Settings:
    return Settings()
