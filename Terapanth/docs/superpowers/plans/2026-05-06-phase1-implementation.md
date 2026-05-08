# Terapanth Learning Platform — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a citation-backed Q&A platform for Jain (Terapanth) philosophy with an admin ingestion pipeline, hybrid RAG retrieval, and a guest chat interface.

**Architecture:** Admin uploads PDFs → approved documents are chunked with contextual embeddings → stored in PostgreSQL + pgvector. Users ask questions → query expansion + hybrid vector/BM25 retrieval → Cohere reranking → Groq/Ollama generates answer with citations and follow-up suggestions.

**Tech Stack:** FastAPI 0.115.5 · SQLAlchemy 2.0 async · PostgreSQL 16 + pgvector · Alembic · Next.js 15 · Groq / Ollama (OpenAI-compatible) · Cohere Rerank · Docker

---

## File Map

```
backend/
  app/
    main.py                        # FastAPI app, lifespan, global error handler
    config.py                      # pydantic-settings: DB URL, LLM provider, API keys
    core/
      exceptions.py                # Domain exceptions + HTTP error shapes
    db/
      base.py                      # Async engine, session factory, Base declarative
      models.py                    # SQLAlchemy ORM models: Session, Conversation, Message, Document, Chunk
      migrations/
        env.py                     # Alembic env (async)
        versions/
          001_initial_schema.py    # Creates all tables + pgvector extension + ts_vector trigger
    ingestion/
      router.py                    # Admin HTTP endpoints — imported by main.py
      service.py                   # Pipeline orchestrator — called by router.py
      extractor.py                 # PDF/text → raw string — called by service.py
      chunker.py                   # Raw text → list[str] chunks — called by service.py
      contextual_embedder.py       # chunk → (context_summary, embedding) — called by service.py
      models.py                    # Pydantic: DocumentUpload, DocumentOut, ApproveRequest
    retrieval/
      vector_store.py              # pgvector similarity search — called by hybrid.py
      bm25.py                      # PostgreSQL ts_rank search — called by hybrid.py
      reranker.py                  # Cohere rerank + trust_score boost — called by hybrid.py
      hybrid.py                    # Merges vector + BM25, calls reranker — called by chat/service.py
      models.py                    # RetrievedChunk, RerankResult
    chat/
      router.py                    # User HTTP endpoints — imported by main.py
      service.py                   # Orchestrates session → expand → retrieve → generate
      session_service.py           # UUID session CRUD + sliding window history
      query_expander.py            # Generates 2-3 query variants via LLM
      models.py                    # ChatRequest, ChatResponse, ConversationOut, MessageOut
    llm/
      provider.py                  # LLMProvider Protocol (chat + embed)
      groq_provider.py             # Groq via openai SDK — returned by factory.py
      ollama_provider.py           # Ollama via openai SDK — returned by factory.py
      factory.py                   # get_llm_provider() → LLMProvider from config
  tests/
    conftest.py                    # pytest fixtures: async DB session, mock LLM provider
    ingestion/
      test_extractor.py
      test_chunker.py
      test_contextual_embedder.py
      test_service.py
    retrieval/
      test_vector_store.py
      test_bm25.py
      test_reranker.py
      test_hybrid.py
    chat/
      test_session_service.py
      test_query_expander.py
      test_service.py
    test_routers.py                # HTTP integration tests via httpx.AsyncClient
  requirements.txt
  requirements-dev.txt
  Dockerfile
  alembic.ini

frontend/
  src/
    app/
      page.tsx                     # Home: conversation list + new chat button
      chat/[id]/page.tsx           # Active chat view
      layout.tsx                   # Root layout, session init
    components/
      ChatMessage.tsx              # Single message bubble with citations
      CitationBadge.tsx            # Inline source pill — used by ChatMessage.tsx
      FollowUpSuggestions.tsx      # Clickable follow-up chips — used by chat/[id]/page.tsx
      ConversationList.tsx         # Sidebar list — used by page.tsx
    services/
      api.ts                       # Axios/fetch wrapper, injects X-Session-ID header
      session.ts                   # localStorage UUID: get or create
    hooks/
      useChat.ts                   # Chat state, send message, load history
  package.json

docker-compose.yml
.env.example
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `backend/requirements.txt`
- Create: `backend/requirements-dev.txt`
- Create: `backend/Dockerfile`

- [ ] **Step 1: Create docker-compose.yml**

```yaml
# docker-compose.yml
# Runs postgres (with pgvector), and the FastAPI backend.
# Usage: docker compose up --build
# Note: add ollama service manually if running LLMs locally.

services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: terapanth
      POSTGRES_USER: terapanth
      POSTGRES_PASSWORD: terapanth
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U terapanth"]
      interval: 5s
      timeout: 5s
      retries: 5

  api:
    build: ./backend
    ports:
      - "8000:8000"
    env_file: .env
    environment:
      DATABASE_URL: postgresql+asyncpg://terapanth:terapanth@postgres:5432/terapanth
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - ./uploads:/app/uploads

volumes:
  postgres_data:
```

- [ ] **Step 2: Create .env.example**

```bash
# .env.example — copy to .env and fill in values

DATABASE_URL=postgresql+asyncpg://terapanth:terapanth@localhost:5432/terapanth

# "groq" or "ollama"
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-70b-versatile
LLM_BASE_URL=https://api.groq.com/openai/v1
GROQ_API_KEY=your_groq_key_here

# Embedding model (openai SDK, points to OpenAI or Ollama)
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=your_openai_key_here

# Reranker ("cohere" or "local")
RERANKER_PROVIDER=cohere
COHERE_API_KEY=your_cohere_key_here

UPLOAD_DIR=./uploads
MAX_UPLOAD_BYTES=52428800
```

- [ ] **Step 3: Create backend/requirements.txt**

```
fastapi==0.115.5
uvicorn[standard]==0.32.1
sqlalchemy==2.0.36
alembic==1.14.0
asyncpg==0.30.0
pgvector==0.3.6
pydantic==2.10.3
pydantic-settings==2.6.1
python-multipart==0.0.12
pymupdf==1.25.1
tiktoken==0.8.0
openai==1.57.4
cohere==5.13.0
aiofiles==24.1.0
```

- [ ] **Step 4: Create backend/requirements-dev.txt**

```
-r requirements.txt
pytest==8.3.4
pytest-asyncio==0.24.0
pytest-mock==3.14.0
httpx==0.28.1
```

- [ ] **Step 5: Create backend/Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/uploads

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 6: Create directory structure**

```bash
mkdir -p backend/app/core
mkdir -p backend/app/db/migrations/versions
mkdir -p backend/app/ingestion
mkdir -p backend/app/retrieval
mkdir -p backend/app/chat
mkdir -p backend/app/llm
mkdir -p backend/tests/ingestion
mkdir -p backend/tests/retrieval
mkdir -p backend/tests/chat
touch backend/app/__init__.py
touch backend/app/core/__init__.py
touch backend/app/db/__init__.py
touch backend/app/ingestion/__init__.py
touch backend/app/retrieval/__init__.py
touch backend/app/chat/__init__.py
touch backend/app/llm/__init__.py
touch backend/tests/__init__.py
touch backend/tests/ingestion/__init__.py
touch backend/tests/retrieval/__init__.py
touch backend/tests/chat/__init__.py
```

- [ ] **Step 7: Commit**

```bash
git init
git add docker-compose.yml .env.example backend/
git commit -m "chore: project scaffolding and dependency lockfile"
```

---

## Task 2: Config and Core Exceptions

**Files:**
- Create: `backend/app/config.py`
- Create: `backend/app/core/exceptions.py`

- [ ] **Step 1: Write failing test for config**

```python
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

    from app.config import Settings
    s = Settings()
    assert s.llm_provider == "groq"
    assert s.max_upload_bytes == 52428800


def test_settings_rejects_unknown_llm_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "invalid")
    with pytest.raises(ValidationError):
        from app.config import Settings
        Settings()
```

- [ ] **Step 2: Run test — expect failure**

```bash
cd backend
pytest tests/test_config.py -v
```
Expected: `ImportError` or `ModuleNotFoundError`

- [ ] **Step 3: Create backend/app/config.py**

```python
# app/config.py
# Application settings loaded from environment variables.
# Imported by: app/main.py, app/db/base.py, app/llm/factory.py,
#              app/retrieval/reranker.py, app/ingestion/service.py

from functools import lru_cache
from typing import Literal
from pydantic import field_validator
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
    reranker_provider: Literal["cohere", "local"] = "cohere"
    cohere_api_key: str = ""
    upload_dir: str = "./uploads"
    max_upload_bytes: int = 52_428_800  # 50 MB


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Create backend/app/core/exceptions.py**

```python
# app/core/exceptions.py
# Domain exception classes for the Terapanth platform.
# Imported by: all service modules. Handled globally in app/main.py.


class TerapanthError(Exception):
    """Base exception for all domain errors."""


class DocumentNotFoundError(TerapanthError):
    def __init__(self, document_id: str):
        super().__init__(f"Document {document_id} not found")
        self.document_id = document_id


class DocumentTooLargeError(TerapanthError):
    def __init__(self, size_bytes: int, max_bytes: int):
        super().__init__(f"File {size_bytes} bytes exceeds limit {max_bytes} bytes")


class DocumentProcessingError(TerapanthError):
    """Raised when chunking or embedding fails."""


class ConversationNotFoundError(TerapanthError):
    def __init__(self, conversation_id: str):
        super().__init__(f"Conversation {conversation_id} not found")
        self.conversation_id = conversation_id


class SessionNotFoundError(TerapanthError):
    def __init__(self, session_id: str):
        super().__init__(f"Session {session_id} not found")
        self.session_id = session_id


class RetrievalError(TerapanthError):
    """Raised when retrieval or reranking fails."""


class LLMError(TerapanthError):
    """Raised when LLM call fails."""
```

- [ ] **Step 5: Run tests — expect pass**

```bash
pytest tests/test_config.py -v
```
Expected: 1 pass, 1 pass (ValidationError test may need import reload — use `importlib.reload` or restructure)

- [ ] **Step 6: Commit**

```bash
git add backend/app/config.py backend/app/core/exceptions.py backend/tests/test_config.py
git commit -m "feat: config settings and domain exceptions"
```

---

## Task 3: Database Models and Migration

**Files:**
- Create: `backend/app/db/base.py`
- Create: `backend/app/db/models.py`
- Create: `backend/alembic.ini`
- Create: `backend/app/db/migrations/env.py`
- Create: `backend/app/db/migrations/versions/001_initial_schema.py`

- [ ] **Step 1: Create backend/app/db/base.py**

```python
# app/db/base.py
# Async SQLAlchemy engine and session factory.
# Imported by: app/main.py (lifespan), all repository/service modules.

from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def _make_engine():
    settings = get_settings()
    return create_async_engine(settings.database_url, pool_pre_ping=True)


engine = _make_engine()
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
```

- [ ] **Step 2: Create backend/app/db/models.py**

```python
# app/db/models.py
# SQLAlchemy ORM models for all tables.
# Imported by: all service modules and Alembic migrations.

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Integer,
    Numeric,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    last_active: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )
    # Phase 2: add user_id FK here
    # Phase 2: add knowledge_level column here

    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    last_active: Mapped[datetime] = mapped_column(server_default=func.now())

    session: Mapped["Session"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (CheckConstraint("role IN ('user', 'assistant')", name="ck_role"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[Any] | None] = mapped_column(JSONB, nullable=True)
    follow_ups: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'approved', 'rejected')", name="ck_doc_status"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str | None] = mapped_column(Text, nullable=True)
    sect: Mapped[str | None] = mapped_column(Text, nullable=True)
    topic: Mapped[str | None] = mapped_column(Text, nullable=True)
    trust_score: Mapped[float] = mapped_column(Numeric(3, 2), default=0.8)
    language: Mapped[str] = mapped_column(Text, default="en")
    status: Mapped[str] = mapped_column(Text, default="pending")
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    ts_vector: Mapped[Any | None] = mapped_column(TSVECTOR, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    document: Mapped["Document"] = relationship(back_populates="chunks")
```

- [ ] **Step 3: Set up Alembic**

```bash
cd backend
pip install -r requirements-dev.txt
alembic init app/db/migrations
```

- [ ] **Step 4: Replace backend/app/db/migrations/env.py**

```python
# app/db/migrations/env.py
# Alembic async migration environment.
# Used only by Alembic CLI — not imported by application code.

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings
from app.db.base import Base
from app.db import models  # noqa: F401 — registers models with Base.metadata

config = context.config
fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=get_settings().database_url,
        target_metadata=target_metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = create_async_engine(get_settings().database_url)
    async with engine.begin() as conn:
        await conn.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 5: Create initial migration**

```bash
alembic revision --autogenerate -m "initial_schema"
```

Then open the generated file in `app/db/migrations/versions/` and **append** this to the `upgrade()` function body after the table creation:

```python
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # IVFFlat index for vector similarity search
    op.execute(
        "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
        "ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )

    # GIN index for full-text search
    op.execute("CREATE INDEX IF NOT EXISTS chunks_ts_vector_idx ON chunks USING GIN (ts_vector)")

    # Trigger: auto-update ts_vector when chunk content changes
    op.execute("""
        CREATE OR REPLACE FUNCTION update_chunk_ts_vector()
        RETURNS trigger AS $$
        BEGIN
            NEW.ts_vector := to_tsvector('english', COALESCE(NEW.content, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)
    op.execute("""
        CREATE TRIGGER chunk_ts_vector_update
        BEFORE INSERT OR UPDATE OF content ON chunks
        FOR EACH ROW EXECUTE FUNCTION update_chunk_ts_vector()
    """)
```

- [ ] **Step 6: Start postgres and run migration**

```bash
docker compose up postgres -d
alembic upgrade head
```
Expected: Migration runs without errors. `alembic current` shows `head`.

- [ ] **Step 7: Commit**

```bash
git add backend/app/db/ backend/alembic.ini
git commit -m "feat: database models and initial Alembic migration"
```

---

## Task 4: LLM Provider Abstraction

**Files:**
- Create: `backend/app/llm/provider.py`
- Create: `backend/app/llm/groq_provider.py`
- Create: `backend/app/llm/ollama_provider.py`
- Create: `backend/app/llm/factory.py`
- Create: `backend/tests/test_llm_factory.py`

- [ ] **Step 1: Write failing test**

```python
# backend/tests/test_llm_factory.py
"""Tests that factory returns correct provider based on config."""

from unittest.mock import patch
import pytest


def test_factory_returns_groq_provider():
    with patch("app.config.get_settings") as mock:
        mock.return_value.llm_provider = "groq"
        mock.return_value.llm_model = "llama-3.1-70b-versatile"
        mock.return_value.llm_base_url = "https://api.groq.com/openai/v1"
        mock.return_value.groq_api_key = "test"
        mock.return_value.embedding_base_url = "https://api.openai.com/v1"
        mock.return_value.embedding_model = "text-embedding-3-small"
        mock.return_value.openai_api_key = "test"

        from app.llm.factory import get_llm_provider
        from app.llm.groq_provider import GroqProvider
        provider = get_llm_provider()
        assert isinstance(provider, GroqProvider)


def test_factory_returns_ollama_provider():
    with patch("app.config.get_settings") as mock:
        mock.return_value.llm_provider = "ollama"
        mock.return_value.llm_model = "llama3.1:8b"
        mock.return_value.llm_base_url = "http://localhost:11434/v1"
        mock.return_value.groq_api_key = ""
        mock.return_value.embedding_base_url = "http://localhost:11434/v1"
        mock.return_value.embedding_model = "nomic-embed-text"
        mock.return_value.openai_api_key = ""

        from app.llm.factory import get_llm_provider
        from app.llm.ollama_provider import OllamaProvider
        provider = get_llm_provider()
        assert isinstance(provider, OllamaProvider)
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_llm_factory.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create backend/app/llm/provider.py**

```python
# app/llm/provider.py
# Protocol (interface) for LLM providers.
# Imported by: app/llm/groq_provider.py, app/llm/ollama_provider.py,
#              app/chat/service.py, app/ingestion/contextual_embedder.py,
#              app/chat/query_expander.py

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str: ...

    async def embed(self, text: str) -> list[float]: ...
```

- [ ] **Step 4: Create backend/app/llm/groq_provider.py**

```python
# app/llm/groq_provider.py
# Groq LLM provider using the OpenAI-compatible API.
# Returned by app/llm/factory.py when LLM_PROVIDER=groq.

from openai import AsyncOpenAI

from app.core.exceptions import LLMError


class GroqProvider:
    """Chat via Groq API; embeddings via OpenAI API."""

    def __init__(
        self,
        llm_model: str,
        llm_base_url: str,
        groq_api_key: str,
        embedding_model: str,
        embedding_base_url: str,
        openai_api_key: str,
    ) -> None:
        self._chat_client = AsyncOpenAI(
            api_key=groq_api_key, base_url=llm_base_url
        )
        self._embed_client = AsyncOpenAI(
            api_key=openai_api_key, base_url=embedding_base_url
        )
        self._llm_model = llm_model
        self._embedding_model = embedding_model

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        try:
            response = await self._chat_client.chat.completions.create(
                model=self._llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMError(f"Groq chat failed: {exc}") from exc

    async def embed(self, text: str) -> list[float]:
        try:
            response = await self._embed_client.embeddings.create(
                model=self._embedding_model, input=text
            )
            return response.data[0].embedding
        except Exception as exc:
            raise LLMError(f"Embedding failed: {exc}") from exc
```

- [ ] **Step 5: Create backend/app/llm/ollama_provider.py**

```python
# app/llm/ollama_provider.py
# Ollama LLM provider via OpenAI-compatible local API.
# Returned by app/llm/factory.py when LLM_PROVIDER=ollama.

from openai import AsyncOpenAI

from app.core.exceptions import LLMError


class OllamaProvider:
    """Chat and embeddings via local Ollama instance."""

    def __init__(
        self,
        llm_model: str,
        llm_base_url: str,
        embedding_model: str,
        embedding_base_url: str,
    ) -> None:
        self._chat_client = AsyncOpenAI(api_key="ollama", base_url=llm_base_url)
        self._embed_client = AsyncOpenAI(api_key="ollama", base_url=embedding_base_url)
        self._llm_model = llm_model
        self._embedding_model = embedding_model

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        try:
            response = await self._chat_client.chat.completions.create(
                model=self._llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMError(f"Ollama chat failed: {exc}") from exc

    async def embed(self, text: str) -> list[float]:
        try:
            response = await self._embed_client.embeddings.create(
                model=self._embedding_model, input=text
            )
            return response.data[0].embedding
        except Exception as exc:
            raise LLMError(f"Ollama embed failed: {exc}") from exc
```

- [ ] **Step 6: Create backend/app/llm/factory.py**

```python
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
```

- [ ] **Step 7: Run tests — expect pass**

```bash
pytest tests/test_llm_factory.py -v
```
Expected: 2 passed

- [ ] **Step 8: Commit**

```bash
git add backend/app/llm/ backend/tests/test_llm_factory.py
git commit -m "feat: LLM provider abstraction (Groq + Ollama)"
```

---

## Task 5: Document Extractor and Chunker

**Files:**
- Create: `backend/app/ingestion/extractor.py`
- Create: `backend/app/ingestion/chunker.py`
- Create: `backend/tests/ingestion/test_extractor.py`
- Create: `backend/tests/ingestion/test_chunker.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/ingestion/test_extractor.py
"""Tests for PDF and plain-text extraction."""
import io
import pytest
from app.ingestion.extractor import extract_text


def test_extracts_plain_text_from_txt_bytes():
    content = b"Anekantavada is the Jain doctrine of non-absolutism."
    result = extract_text(content, filename="test.txt")
    assert "Anekantavada" in result


def test_raises_on_unsupported_extension():
    from app.core.exceptions import DocumentProcessingError
    with pytest.raises(DocumentProcessingError):
        extract_text(b"data", filename="test.docx")


def test_returns_non_empty_string_for_empty_txt():
    result = extract_text(b"", filename="empty.txt")
    assert isinstance(result, str)
```

```python
# backend/tests/ingestion/test_chunker.py
"""Tests for token-aware text chunking."""
import pytest
from app.ingestion.chunker import chunk_text


def test_short_text_returns_single_chunk():
    chunks = chunk_text("Short text.", chunk_size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."


def test_long_text_splits_into_multiple_chunks():
    long_text = " ".join(["word"] * 600)
    chunks = chunk_text(long_text, chunk_size=512, overlap=64)
    assert len(chunks) > 1


def test_chunks_overlap():
    # With overlap, the last tokens of chunk N appear at start of chunk N+1
    long_text = " ".join([f"word{i}" for i in range(600)])
    chunks = chunk_text(long_text, chunk_size=512, overlap=64)
    # Last few tokens of chunk 0 should appear in chunk 1
    last_words_of_first = chunks[0].split()[-5:]
    assert any(w in chunks[1] for w in last_words_of_first)


def test_empty_text_returns_empty_list():
    assert chunk_text("", chunk_size=512, overlap=64) == []
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/ingestion/ -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create backend/app/ingestion/extractor.py**

```python
# app/ingestion/extractor.py
# Extracts raw text from uploaded PDF or plain-text bytes.
# Called by: app/ingestion/service.py

import fitz  # PyMuPDF

from app.core.exceptions import DocumentProcessingError

_SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def extract_text(content: bytes, filename: str) -> str:
    """Return plain text extracted from file bytes.

    Args:
        content: raw file bytes
        filename: original filename, used to detect format

    Raises:
        DocumentProcessingError: if format unsupported or extraction fails
    """
    ext = _get_extension(filename)
    if ext not in _SUPPORTED_EXTENSIONS:
        raise DocumentProcessingError(
            f"Unsupported file type '{ext}'. Supported: {_SUPPORTED_EXTENSIONS}"
        )
    if ext == ".txt":
        return content.decode("utf-8", errors="replace")
    return _extract_pdf(content)


def _get_extension(filename: str) -> str:
    dot = filename.rfind(".")
    if dot == -1:
        return ""
    return filename[dot:].lower()


def _extract_pdf(content: bytes) -> str:
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        pages = [page.get_text() for page in doc]
        return "\n".join(pages)
    except Exception as exc:
        raise DocumentProcessingError(f"PDF extraction failed: {exc}") from exc
```

- [ ] **Step 4: Create backend/app/ingestion/chunker.py**

```python
# app/ingestion/chunker.py
# Splits raw document text into token-aware overlapping chunks.
# Called by: app/ingestion/service.py

import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping token chunks.

    Respects token boundaries. chunk_size and overlap are in tokens.
    Returns [] for empty input.
    """
    if not text.strip():
        return []

    tokens = _ENCODER.encode(text)
    chunks: list[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_ENCODER.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += chunk_size - overlap

    return chunks
```

- [ ] **Step 5: Run tests — expect pass**

```bash
pytest tests/ingestion/test_extractor.py tests/ingestion/test_chunker.py -v
```
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add backend/app/ingestion/extractor.py backend/app/ingestion/chunker.py \
        backend/tests/ingestion/test_extractor.py backend/tests/ingestion/test_chunker.py
git commit -m "feat: document extractor and token-aware chunker"
```

---

## Task 6: Contextual Embedder

**Files:**
- Create: `backend/app/ingestion/contextual_embedder.py`
- Create: `backend/tests/ingestion/test_contextual_embedder.py`

- [ ] **Step 1: Write failing test**

```python
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
    # Context summary should be prepended
    assert mock_llm.chat.return_value in embed_call_arg
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/ingestion/test_contextual_embedder.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create backend/app/ingestion/contextual_embedder.py**

```python
# app/ingestion/contextual_embedder.py
# Generates a contextual summary for each chunk and embeds the combined text.
# This is Anthropic's "contextual retrieval" technique — summarising the chunk
# within its document context before embedding significantly improves recall.
# Called by: app/ingestion/service.py

from app.llm.provider import LLMProvider

_CONTEXT_PROMPT = """\
Document: {title} by {author}. Topic: {topic}.

Summarize in 1-2 sentences what the following passage covers, so it can be \
retrieved in isolation without the rest of the document:

{chunk}
"""


async def build_contextual_embedding(
    chunk: str,
    doc_title: str,
    doc_author: str | None,
    doc_topic: str | None,
    llm: LLMProvider,
) -> tuple[str, list[float]]:
    """Return (context_summary, embedding) for a single chunk.

    The embedding is computed on "summary + chunk" so the vector captures
    both location context and content.
    """
    prompt = _CONTEXT_PROMPT.format(
        title=doc_title,
        author=doc_author or "Unknown",
        topic=doc_topic or "general",
        chunk=chunk,
    )
    context = await llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150,
    )
    combined = f"{context}\n\n{chunk}"
    embedding = await llm.embed(combined)
    return context, embedding
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/ingestion/test_contextual_embedder.py -v
```
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingestion/contextual_embedder.py \
        backend/tests/ingestion/test_contextual_embedder.py
git commit -m "feat: contextual embedding generation"
```

---

## Task 7: Ingestion Service, Pydantic Models, and Admin Router

**Files:**
- Create: `backend/app/ingestion/models.py`
- Create: `backend/app/ingestion/service.py`
- Create: `backend/app/ingestion/router.py`
- Create: `backend/tests/ingestion/test_service.py`

- [ ] **Step 1: Create backend/app/ingestion/models.py**

```python
# app/ingestion/models.py
# Pydantic request/response schemas for the ingestion API.
# Imported by: app/ingestion/router.py

import uuid
from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: uuid.UUID
    title: str
    author: str | None
    sect: str | None
    topic: str | None
    trust_score: float
    language: str
    status: str
    chunk_count: int = 0

    model_config = {"from_attributes": True}


class RejectRequest(BaseModel):
    reason: str = Field(min_length=1)
```

- [ ] **Step 2: Write failing test for ingestion service**

```python
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

    # embed called once per chunk
    assert mock_llm.embed.call_count == 2
    # db.add called twice (one per chunk)
    assert mock_db.add.call_count == 2
```

- [ ] **Step 3: Run — expect failure**

```bash
pytest tests/ingestion/test_service.py -v
```

- [ ] **Step 4: Create backend/app/ingestion/service.py**

```python
# app/ingestion/service.py
# Orchestrates the document ingestion pipeline: extract → chunk → embed → store.
# Called by: app/ingestion/router.py on document approval.

import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DocumentProcessingError
from app.db.models import Chunk, Document
from app.ingestion.chunker import chunk_text
from app.ingestion.contextual_embedder import build_contextual_embedding
from app.ingestion.extractor import extract_text
from app.llm.provider import LLMProvider


async def process_approved_document(
    doc: Document,
    db: AsyncSession,
    llm: LLMProvider,
) -> int:
    """Run the full ingestion pipeline for an approved document.

    Returns the number of chunks created.
    Raises DocumentProcessingError on any failure.
    """
    try:
        raw_bytes = Path(doc.file_path).read_bytes()
        filename = Path(doc.file_path).name
        text = extract_text(raw_bytes, filename)
        texts = chunk_text(text, chunk_size=512, overlap=64)
    except Exception as exc:
        raise DocumentProcessingError(f"Pre-processing failed: {exc}") from exc

    for index, chunk_text_str in enumerate(texts):
        try:
            context, embedding = await build_contextual_embedding(
                chunk=chunk_text_str,
                doc_title=doc.title,
                doc_author=doc.author,
                doc_topic=doc.topic,
                llm=llm,
            )
        except Exception as exc:
            raise DocumentProcessingError(
                f"Embedding failed for chunk {index}: {exc}"
            ) from exc

        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            content=chunk_text_str,
            context=context,
            chunk_index=index,
            embedding=embedding,
        )
        db.add(chunk)

    await db.commit()
    return len(texts)
```

- [ ] **Step 5: Create backend/app/ingestion/router.py**

```python
# app/ingestion/router.py
# Admin HTTP endpoints for document upload, review, and status management.
# Imported by: app/main.py (included as router with prefix /api/v1/admin)

import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.exceptions import DocumentNotFoundError, DocumentTooLargeError
from app.db.base import get_db
from app.db.models import Chunk, Document
from app.ingestion.models import DocumentOut, RejectRequest
from app.ingestion.service import process_approved_document
from app.llm.factory import get_llm_provider

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.post("/documents", response_model=DocumentOut, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    author: str = Form(""),
    sect: str = Form(""),
    topic: str = Form(""),
    trust_score: float = Form(0.8),
    db: AsyncSession = Depends(get_db),
):
    settings = get_settings()
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_upload_bytes} byte limit",
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    doc_id = uuid.uuid4()
    safe_name = f"{doc_id}_{Path(file.filename or 'upload').name}"
    file_path = upload_dir / safe_name

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    doc = Document(
        id=doc_id,
        title=title,
        author=author or None,
        sect=sect or None,
        topic=topic or None,
        trust_score=trust_score,
        language="en",
        status="pending",
        file_path=str(file_path),
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return DocumentOut.model_validate(doc)


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    docs = result.scalars().all()
    out = []
    for doc in docs:
        count_result = await db.execute(
            select(func.count()).where(Chunk.document_id == doc.id)
        )
        count = count_result.scalar_one()
        d = DocumentOut.model_validate(doc)
        d = d.model_copy(update={"chunk_count": count})
        out.append(d)
    return out


@router.get("/documents/{document_id}", response_model=DocumentOut)
async def get_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    count_result = await db.execute(
        select(func.count()).where(Chunk.document_id == doc.id)
    )
    count = count_result.scalar_one()
    return DocumentOut.model_validate(doc).model_copy(update={"chunk_count": count})


@router.patch("/documents/{document_id}/approve", response_model=DocumentOut)
async def approve_document(
    document_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status != "pending":
        raise HTTPException(status_code=400, detail=f"Document is already {doc.status}")

    doc.status = "approved"
    await db.commit()

    llm = get_llm_provider()
    chunk_count = await process_approved_document(doc=doc, db=db, llm=llm)
    return DocumentOut.model_validate(doc).model_copy(update={"chunk_count": chunk_count})


@router.patch("/documents/{document_id}/reject", response_model=DocumentOut)
async def reject_document(
    document_id: uuid.UUID,
    body: RejectRequest,
    db: AsyncSession = Depends(get_db),
):
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    doc.status = "rejected"
    await db.commit()
    return DocumentOut.model_validate(doc)
```

- [ ] **Step 6: Run tests — expect pass**

```bash
pytest tests/ingestion/ -v
```
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add backend/app/ingestion/ backend/tests/ingestion/
git commit -m "feat: document ingestion pipeline and admin API"
```

---

## Task 8: Vector Store and BM25 Retrieval

**Files:**
- Create: `backend/app/retrieval/models.py`
- Create: `backend/app/retrieval/vector_store.py`
- Create: `backend/app/retrieval/bm25.py`
- Create: `backend/tests/retrieval/test_vector_store.py`
- Create: `backend/tests/retrieval/test_bm25.py`

- [ ] **Step 1: Create backend/app/retrieval/models.py**

```python
# app/retrieval/models.py
# Data transfer objects for retrieval results.
# Imported by: app/retrieval/vector_store.py, app/retrieval/bm25.py,
#              app/retrieval/reranker.py, app/retrieval/hybrid.py

import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    context: str
    doc_title: str
    doc_author: str | None
    trust_score: float
    score: float  # raw retrieval score (cosine or ts_rank)


@dataclass(frozen=True)
class RerankResult:
    chunk: RetrievedChunk
    rerank_score: float  # final score after reranking + trust boost
```

- [ ] **Step 2: Write failing tests**

```python
# backend/tests/retrieval/test_vector_store.py
"""Tests for pgvector similarity search."""
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_vector_search_returns_retrieved_chunks():
    from app.retrieval.vector_store import vector_search
    from app.retrieval.models import RetrievedChunk

    mock_db = AsyncMock()
    chunk_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    row = MagicMock()
    row.chunk_id = chunk_id
    row.document_id = doc_id
    row.content = "Anekantavada is the doctrine of many-sidedness."
    row.context = "This describes Anekantavada."
    row.doc_title = "Tattvartha Sutra"
    row.doc_author = "Umasvati"
    row.trust_score = 0.9
    row.score = 0.87

    mock_result = MagicMock()
    mock_result.all.return_value = [row]
    mock_db.execute = AsyncMock(return_value=mock_result)

    results = await vector_search(
        query_embedding=[0.1] * 1536, db=mock_db, top_k=20
    )
    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].content == "Anekantavada is the doctrine of many-sidedness."
```

```python
# backend/tests/retrieval/test_bm25.py
"""Tests for PostgreSQL full-text search."""
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_bm25_search_returns_retrieved_chunks():
    from app.retrieval.bm25 import bm25_search
    from app.retrieval.models import RetrievedChunk

    mock_db = AsyncMock()
    row = MagicMock()
    row.chunk_id = uuid.uuid4()
    row.document_id = uuid.uuid4()
    row.content = "Syadvada means conditional predication."
    row.context = "This describes Syadvada."
    row.doc_title = "Niyamasara"
    row.doc_author = None
    row.trust_score = 0.8
    row.score = 0.5

    mock_result = MagicMock()
    mock_result.all.return_value = [row]
    mock_db.execute = AsyncMock(return_value=mock_result)

    results = await bm25_search(query="Syadvada", db=mock_db, top_k=20)
    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)
```

- [ ] **Step 3: Run — expect failure**

```bash
pytest tests/retrieval/test_vector_store.py tests/retrieval/test_bm25.py -v
```

- [ ] **Step 4: Create backend/app/retrieval/vector_store.py**

```python
# app/retrieval/vector_store.py
# Runs pgvector cosine similarity search against the chunks table.
# Called by: app/retrieval/hybrid.py

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.models import RetrievedChunk


async def vector_search(
    query_embedding: list[float],
    db: AsyncSession,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """Return top_k chunks ranked by cosine similarity to query_embedding."""
    sql = text("""
        SELECT
            c.id            AS chunk_id,
            c.document_id,
            c.content,
            c.context,
            d.title         AS doc_title,
            d.author        AS doc_author,
            d.trust_score,
            1 - (c.embedding <=> CAST(:embedding AS vector)) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.status = 'approved' AND c.embedding IS NOT NULL
        ORDER BY c.embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """)
    result = await db.execute(
        sql, {"embedding": str(query_embedding), "top_k": top_k}
    )
    return [
        RetrievedChunk(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            content=row.content,
            context=row.context,
            doc_title=row.doc_title,
            doc_author=row.doc_author,
            trust_score=float(row.trust_score),
            score=float(row.score),
        )
        for row in result.all()
    ]
```

- [ ] **Step 5: Create backend/app/retrieval/bm25.py**

```python
# app/retrieval/bm25.py
# Runs PostgreSQL full-text search (BM25-style ts_rank) against chunks.
# Called by: app/retrieval/hybrid.py

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.models import RetrievedChunk


async def bm25_search(
    query: str,
    db: AsyncSession,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """Return top_k chunks ranked by ts_rank full-text relevance."""
    sql = text("""
        SELECT
            c.id            AS chunk_id,
            c.document_id,
            c.content,
            c.context,
            d.title         AS doc_title,
            d.author        AS doc_author,
            d.trust_score,
            ts_rank(c.ts_vector, plainto_tsquery('english', :query)) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.status = 'approved'
          AND c.ts_vector @@ plainto_tsquery('english', :query)
        ORDER BY score DESC
        LIMIT :top_k
    """)
    result = await db.execute(sql, {"query": query, "top_k": top_k})
    return [
        RetrievedChunk(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            content=row.content,
            context=row.context,
            doc_title=row.doc_title,
            doc_author=row.doc_author,
            trust_score=float(row.trust_score),
            score=float(row.score),
        )
        for row in result.all()
    ]
```

- [ ] **Step 6: Run tests — expect pass**

```bash
pytest tests/retrieval/test_vector_store.py tests/retrieval/test_bm25.py -v
```

- [ ] **Step 7: Commit**

```bash
git add backend/app/retrieval/models.py backend/app/retrieval/vector_store.py \
        backend/app/retrieval/bm25.py backend/tests/retrieval/
git commit -m "feat: vector store and BM25 retrieval"
```

---

## Task 9: Reranker and Hybrid Retrieval

**Files:**
- Create: `backend/app/retrieval/reranker.py`
- Create: `backend/app/retrieval/hybrid.py`
- Create: `backend/tests/retrieval/test_reranker.py`
- Create: `backend/tests/retrieval/test_hybrid.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/retrieval/test_reranker.py
"""Tests for Cohere reranker with trust_score boost."""
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.retrieval.models import RetrievedChunk, RerankResult


def make_chunk(content: str, trust_score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=content,
        context="Context.",
        doc_title="Test Doc",
        doc_author=None,
        trust_score=trust_score,
        score=0.5,
    )


@pytest.mark.asyncio
async def test_rerank_returns_top_k_results():
    from app.retrieval.reranker import rerank

    chunks = [make_chunk(f"chunk {i}") for i in range(10)]

    with patch("app.retrieval.reranker.cohere.AsyncClientV2") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.rerank = AsyncMock(return_value=MagicMock(
            results=[
                MagicMock(index=i, relevance_score=0.9 - i * 0.05)
                for i in range(10)
            ]
        ))

        results = await rerank(
            query="What is Anekantavada?",
            chunks=chunks,
            top_k=5,
            cohere_api_key="test",
        )

    assert len(results) == 5
    assert all(isinstance(r, RerankResult) for r in results)


@pytest.mark.asyncio
async def test_trust_score_boosts_rerank_score():
    from app.retrieval.reranker import rerank

    low_trust = make_chunk("chunk low", trust_score=0.5)
    high_trust = make_chunk("chunk high", trust_score=1.0)

    with patch("app.retrieval.reranker.cohere.AsyncClientV2") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.rerank = AsyncMock(return_value=MagicMock(
            results=[
                MagicMock(index=0, relevance_score=0.8),
                MagicMock(index=1, relevance_score=0.8),
            ]
        ))

        results = await rerank(
            query="test",
            chunks=[low_trust, high_trust],
            top_k=2,
            cohere_api_key="test",
        )

    # Same relevance but different trust — high trust should rank higher
    scores = {r.chunk.content: r.rerank_score for r in results}
    assert scores["chunk high"] > scores["chunk low"]
```

```python
# backend/tests/retrieval/test_hybrid.py
"""Tests for hybrid retrieval merging vector + BM25 results."""
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.retrieval.models import RetrievedChunk, RerankResult


def make_chunk(chunk_id: uuid.UUID, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=uuid.uuid4(),
        content=content,
        context="ctx",
        doc_title="Doc",
        doc_author=None,
        trust_score=0.8,
        score=0.6,
    )


@pytest.mark.asyncio
async def test_hybrid_deduplicates_results():
    from app.retrieval.hybrid import hybrid_retrieve

    shared_id = uuid.uuid4()
    chunk_a = make_chunk(shared_id, "shared chunk")
    chunk_b = make_chunk(uuid.uuid4(), "unique chunk")

    mock_db = AsyncMock()

    with patch("app.retrieval.hybrid.vector_search", new_callable=AsyncMock) as mock_vec, \
         patch("app.retrieval.hybrid.bm25_search", new_callable=AsyncMock) as mock_bm25, \
         patch("app.retrieval.hybrid.rerank", new_callable=AsyncMock) as mock_rerank:

        mock_vec.return_value = [chunk_a]
        mock_bm25.return_value = [chunk_a, chunk_b]
        mock_rerank.return_value = [
            RerankResult(chunk=chunk_a, rerank_score=0.9),
            RerankResult(chunk=chunk_b, rerank_score=0.7),
        ]

        results = await hybrid_retrieve(
            query="test",
            query_embedding=[0.0] * 1536,
            db=mock_db,
            cohere_api_key="test",
            top_k=5,
        )

    # rerank should receive deduplicated candidates (2, not 3)
    rerank_call_chunks = mock_rerank.call_args[1]["chunks"]
    ids = [c.chunk_id for c in rerank_call_chunks]
    assert len(ids) == len(set(str(i) for i in ids))
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/retrieval/test_reranker.py tests/retrieval/test_hybrid.py -v
```

- [ ] **Step 3: Create backend/app/retrieval/reranker.py**

```python
# app/retrieval/reranker.py
# Reranks retrieved chunks using Cohere Rerank API with trust_score boost.
# Called by: app/retrieval/hybrid.py

import cohere

from app.retrieval.models import RerankResult, RetrievedChunk


async def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int,
    cohere_api_key: str,
) -> list[RerankResult]:
    """Rerank chunks against query. Applies trust_score as a multiplicative boost.

    Returns top_k results ordered by final score descending.
    """
    if not chunks:
        return []

    client = cohere.AsyncClientV2(api_key=cohere_api_key)
    documents = [chunk.content for chunk in chunks]

    response = await client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=len(chunks),
    )

    results = [
        RerankResult(
            chunk=chunks[r.index],
            rerank_score=r.relevance_score * chunks[r.index].trust_score,
        )
        for r in response.results
    ]

    results.sort(key=lambda r: r.rerank_score, reverse=True)
    return results[:top_k]
```

- [ ] **Step 4: Create backend/app/retrieval/hybrid.py**

```python
# app/retrieval/hybrid.py
# Merges vector search and BM25 results, deduplicates, then reranks.
# Called by: app/chat/service.py
# Phase 3: add a graph retriever to the pipeline here.

from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.bm25 import bm25_search
from app.retrieval.models import RerankResult, RetrievedChunk
from app.retrieval.reranker import rerank
from app.retrieval.vector_store import vector_search


async def hybrid_retrieve(
    query: str,
    query_embedding: list[float],
    db: AsyncSession,
    cohere_api_key: str,
    top_k: int = 5,
) -> list[RerankResult]:
    """Run vector + BM25 retrieval, deduplicate, and rerank.

    query_embedding should be the embedding of the *original* query
    (not the expanded variants) — expansion happens upstream in chat/service.py.
    """
    vector_results = await vector_search(query_embedding, db, top_k=20)
    bm25_results = await bm25_search(query, db, top_k=20)

    seen: dict[str, RetrievedChunk] = {}
    for chunk in vector_results + bm25_results:
        key = str(chunk.chunk_id)
        if key not in seen:
            seen[key] = chunk

    candidates = list(seen.values())
    return await rerank(
        query=query,
        chunks=candidates,
        top_k=top_k,
        cohere_api_key=cohere_api_key,
    )
```

- [ ] **Step 5: Run tests — expect pass**

```bash
pytest tests/retrieval/ -v
```
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add backend/app/retrieval/reranker.py backend/app/retrieval/hybrid.py \
        backend/tests/retrieval/test_reranker.py backend/tests/retrieval/test_hybrid.py
git commit -m "feat: reranker and hybrid retrieval pipeline"
```

---

## Task 10: Session Service and Query Expander

**Files:**
- Create: `backend/app/chat/models.py`
- Create: `backend/app/chat/session_service.py`
- Create: `backend/app/chat/query_expander.py`
- Create: `backend/tests/chat/test_session_service.py`
- Create: `backend/tests/chat/test_query_expander.py`

- [ ] **Step 1: Create backend/app/chat/models.py**

```python
# app/chat/models.py
# Pydantic request/response schemas for the chat API.
# Imported by: app/chat/router.py, app/chat/service.py

import uuid
from pydantic import BaseModel


class Citation(BaseModel):
    title: str
    author: str | None
    chunk_ref: str  # chunk_id as string


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    citations: list[Citation] | None = None
    follow_ups: list[str] | None = None

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: uuid.UUID
    title: str

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    message: str
    conversation_id: uuid.UUID | None = None


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    message: MessageOut
```

- [ ] **Step 2: Write failing tests**

```python
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
    session = await get_or_create_session(session_id=session_id, db=mock_db)

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
```

```python
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
    # Falls back to returning the original query
    assert results == ["What is Anekantavada?"]
```

- [ ] **Step 3: Run — expect failure**

```bash
pytest tests/chat/test_session_service.py tests/chat/test_query_expander.py -v
```

- [ ] **Step 4: Create backend/app/chat/session_service.py**

```python
# app/chat/session_service.py
# Session CRUD and sliding window message history for guest sessions.
# Called by: app/chat/service.py

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, Message
from app.db.models import Session as SessionModel

_WINDOW_SIZE = 10


async def get_or_create_session(
    session_id: uuid.UUID, db: AsyncSession
) -> SessionModel:
    existing = await db.get(SessionModel, session_id)
    if existing:
        return existing

    session = SessionModel(id=session_id)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get_or_create_conversation(
    session_id: uuid.UUID,
    conversation_id: uuid.UUID | None,
    first_message: str,
    db: AsyncSession,
) -> Conversation:
    if conversation_id:
        conv = await db.get(Conversation, conversation_id)
        if conv and conv.session_id == session_id:
            return conv

    title = first_message[:60]
    conv = Conversation(session_id=session_id, title=title)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


async def get_sliding_window(
    conversation_id: uuid.UUID, db: AsyncSession
) -> list[Message]:
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(_WINDOW_SIZE)
    )
    messages = result.scalars().all()
    return list(reversed(messages))


async def save_message(
    conversation_id: uuid.UUID,
    role: str,
    content: str,
    db: AsyncSession,
    citations: list | None = None,
    follow_ups: list[str] | None = None,
) -> Message:
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        citations=citations,
        follow_ups=follow_ups,
    )
    db.add(msg)

    conv = await db.get(Conversation, conversation_id)
    if conv:
        conv.last_active = datetime.utcnow()

    await db.commit()
    await db.refresh(msg)
    return msg
```

- [ ] **Step 5: Create backend/app/chat/query_expander.py**

```python
# app/chat/query_expander.py
# Generates 2-3 alternative phrasings of a user query for broader retrieval.
# Called by: app/chat/service.py

import json
import logging

from app.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

_EXPAND_PROMPT = """\
Generate 2-3 alternative phrasings of the following question about Jain philosophy.
Return ONLY a valid JSON array of strings. No explanation.

Question: {query}

Example output: ["alternate phrasing 1", "alternate phrasing 2", "alternate phrasing 3"]
"""


async def expand_query(query: str, llm: LLMProvider) -> list[str]:
    """Return list of query variants including fallback to original on failure."""
    try:
        raw = await llm.chat(
            [{"role": "user", "content": _EXPAND_PROMPT.format(query=query)}],
            temperature=0.3,
            max_tokens=200,
        )
        variants = json.loads(raw)
        if isinstance(variants, list) and all(isinstance(v, str) for v in variants):
            return variants
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Query expansion failed, using original. Error: %s", exc)
    return [query]
```

- [ ] **Step 6: Run tests — expect pass**

```bash
pytest tests/chat/ -v
```

- [ ] **Step 7: Commit**

```bash
git add backend/app/chat/models.py backend/app/chat/session_service.py \
        backend/app/chat/query_expander.py backend/tests/chat/
git commit -m "feat: session service and query expander"
```

---

## Task 11: Chat Service and Router

**Files:**
- Create: `backend/app/chat/service.py`
- Create: `backend/app/chat/router.py`
- Create: `backend/tests/chat/test_service.py`

- [ ] **Step 1: Write failing test**

```python
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
            '["What is non-absolutism?"]',                     # query expansion
            '{"answer": "Anekantavada means...", "follow_ups": ["Tell me more"]}',  # generation
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
        saved_msg = MagicMock(id=uuid.uuid4(), role="assistant",
                              content="Anekantavada means...",
                              citations=[{"title": "Tattvartha Sutra", "author": "Umasvati", "chunk_ref": "abc"}],
                              follow_ups=["Tell me more"])
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
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/chat/test_service.py -v
```

- [ ] **Step 3: Create backend/app/chat/service.py**

```python
# app/chat/service.py
# Orchestrates the full query pipeline: session → expand → retrieve → generate → store.
# Called by: app/chat/router.py
# Phase 2: replace _build_system_prompt() with a router that selects expert prompts.

import json
import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.models import ChatResponse, Citation, MessageOut
from app.chat.query_expander import expand_query
from app.chat.session_service import (
    get_or_create_conversation,
    get_or_create_session,
    get_sliding_window,
    save_message,
)
from app.llm.provider import LLMProvider
from app.retrieval.hybrid import hybrid_retrieve
from app.retrieval.models import RerankResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a knowledgeable and respectful guide to Jain (Terapanth) philosophy.
Answer questions accurately based only on the provided context passages.
Always cite the source document for each claim. Never speculate or add information
not present in the context. If the context does not contain enough information, say so.

After your answer, provide a JSON structure in this exact format:
{"answer": "your answer here", "follow_ups": ["question 1", "question 2"]}
"""


async def answer_question(
    session_id: uuid.UUID,
    message: str,
    conversation_id: uuid.UUID | None,
    db: AsyncSession,
    llm: LLMProvider,
    cohere_api_key: str,
) -> ChatResponse:
    session = await get_or_create_session(session_id, db)
    conversation = await get_or_create_conversation(
        session_id=session.id,
        conversation_id=conversation_id,
        first_message=message,
        db=db,
    )

    history = await get_sliding_window(conversation.id, db)

    query_variants = await expand_query(message, llm)
    primary_query = query_variants[0]

    query_embedding = await llm.embed(primary_query)

    retrieval_results = await hybrid_retrieve(
        query=primary_query,
        query_embedding=query_embedding,
        db=db,
        cohere_api_key=cohere_api_key,
        top_k=5,
    )

    context_text = _format_context(retrieval_results)
    chat_messages = _build_messages(history, message, context_text)

    raw_response = await llm.chat(chat_messages, temperature=0.2, max_tokens=1024)
    answer, follow_ups = _parse_response(raw_response)
    citations = _build_citations(retrieval_results)

    saved = await save_message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        db=db,
        citations=[c.model_dump() for c in citations],
        follow_ups=follow_ups,
    )

    return ChatResponse(
        conversation_id=conversation.id,
        message=MessageOut(
            id=saved.id,
            role="assistant",
            content=answer,
            citations=citations,
            follow_ups=follow_ups,
        ),
    )


def _format_context(results: list[RerankResult]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] Source: {r.chunk.doc_title}"
            + (f" by {r.chunk.doc_author}" if r.chunk.doc_author else "")
            + f"\n{r.chunk.content}"
        )
    return "\n\n".join(parts)


def _build_messages(
    history: list, current_message: str, context: str
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {current_message}",
    })
    return messages


def _parse_response(raw: str) -> tuple[str, list[str]]:
    try:
        data = json.loads(raw)
        return data.get("answer", raw), data.get("follow_ups", [])
    except json.JSONDecodeError:
        return raw, []


def _build_citations(results: list[RerankResult]) -> list[Citation]:
    return [
        Citation(
            title=r.chunk.doc_title,
            author=r.chunk.doc_author,
            chunk_ref=str(r.chunk.chunk_id),
        )
        for r in results
    ]
```

- [ ] **Step 4: Create backend/app/chat/router.py**

```python
# app/chat/router.py
# User-facing HTTP endpoints: send messages, list/delete conversations.
# Imported by: app/main.py (included as router with prefix /api/v1)

import uuid

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.models import ChatRequest, ChatResponse, ConversationOut, MessageOut
from app.chat.service import answer_question
from app.config import get_settings
from app.db.base import get_db
from app.db.models import Conversation, Message
from app.llm.factory import get_llm_provider

router = APIRouter(prefix="/api/v1", tags=["chat"])


def _parse_session_id(x_session_id: str = Header(...)) -> uuid.UUID:
    try:
        return uuid.UUID(x_session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid X-Session-ID header")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    session_id: uuid.UUID = Depends(_parse_session_id),
    db: AsyncSession = Depends(get_db),
):
    llm = get_llm_provider()
    settings = get_settings()
    return await answer_question(
        session_id=session_id,
        message=body.message,
        conversation_id=body.conversation_id,
        db=db,
        llm=llm,
        cohere_api_key=settings.cohere_api_key,
    )


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(
    session_id: uuid.UUID = Depends(_parse_session_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.session_id == session_id)
        .order_by(Conversation.last_active.desc())
    )
    return [ConversationOut.model_validate(c) for c in result.scalars().all()]


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
async def get_messages(
    conversation_id: uuid.UUID,
    session_id: uuid.UUID = Depends(_parse_session_id),
    db: AsyncSession = Depends(get_db),
):
    conv = await db.get(Conversation, conversation_id)
    if not conv or conv.session_id != session_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    return [MessageOut.model_validate(m) for m in result.scalars().all()]


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: uuid.UUID,
    session_id: uuid.UUID = Depends(_parse_session_id),
    db: AsyncSession = Depends(get_db),
):
    conv = await db.get(Conversation, conversation_id)
    if not conv or conv.session_id != session_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await db.delete(conv)
    await db.commit()
```

- [ ] **Step 5: Run tests — expect pass**

```bash
pytest tests/chat/ -v
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/chat/service.py backend/app/chat/router.py \
        backend/tests/chat/test_service.py
git commit -m "feat: chat service and router"
```

---

## Task 12: FastAPI App Entry Point

**Files:**
- Create: `backend/app/main.py`
- Create: `backend/tests/test_routers.py`

- [ ] **Step 1: Create backend/app/main.py**

```python
# app/main.py
# FastAPI application entry point. Registers routers and global error handler.
# Run with: uvicorn app.main:app --reload

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.chat.router import router as chat_router
from app.core.exceptions import (
    ConversationNotFoundError,
    DocumentNotFoundError,
    DocumentProcessingError,
    DocumentTooLargeError,
    LLMError,
    TerapanthError,
)
from app.ingestion.router import router as ingestion_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Terapanth Learning API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(ingestion_router)


@app.exception_handler(DocumentNotFoundError)
@app.exception_handler(ConversationNotFoundError)
async def not_found_handler(request: Request, exc: TerapanthError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(DocumentTooLargeError)
async def too_large_handler(request: Request, exc: DocumentTooLargeError):
    return JSONResponse(status_code=413, content={"detail": str(exc)})


@app.exception_handler(DocumentProcessingError)
async def processing_error_handler(request: Request, exc: DocumentProcessingError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 2: Write router integration tests**

```python
# backend/tests/test_routers.py
"""HTTP-level integration tests for key endpoints."""
import pytest
import uuid
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch, MagicMock


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
    assert response.status_code == 422  # Missing header


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
```

- [ ] **Step 3: Run all tests**

```bash
pytest tests/ -v
```
Expected: all tests pass

- [ ] **Step 4: Start the full stack and verify health endpoint**

```bash
docker compose up -d
cd backend && uvicorn app.main:app --reload --port 8000
```
Open `http://localhost:8000/health` — expect `{"status": "ok"}`
Open `http://localhost:8000/docs` — verify all endpoints visible in Swagger UI

- [ ] **Step 5: Commit**

```bash
git add backend/app/main.py backend/tests/test_routers.py
git commit -m "feat: FastAPI app entry point and router integration tests"
```

---

## Task 13: Frontend Setup and Session Management

**Files:**
- Create: `frontend/` (Next.js project)
- Create: `frontend/src/services/session.ts`
- Create: `frontend/src/services/api.ts`

- [ ] **Step 1: Scaffold Next.js app**

```bash
cd frontend
npx create-next-app@15.3.1 . --typescript --tailwind --app --no-src-dir --import-alias "@/*"
# When prompted: Yes to TypeScript, Yes to Tailwind, Yes to App Router
# Rename the created src directory if needed: mv app src/app
```

Lock the Next.js version in `package.json`:
```json
{
  "dependencies": {
    "next": "15.3.1",
    "react": "19.0.0",
    "react-dom": "19.0.0"
  },
  "devDependencies": {
    "@types/node": "22.10.2",
    "@types/react": "19.0.1",
    "@types/react-dom": "19.0.1",
    "typescript": "5.7.2",
    "tailwindcss": "4.0.0",
    "@tailwindcss/postcss": "4.0.0"
  }
}
```

Run `npm install` after updating.

- [ ] **Step 2: Create frontend/src/services/session.ts**

```typescript
// src/services/session.ts
// Manages the anonymous guest session UUID in localStorage.
// Imported by: src/services/api.ts, src/app/layout.tsx

const SESSION_KEY = "terapanth_session_id";

function generateUUID(): string {
  return crypto.randomUUID();
}

export function getSessionId(): string {
  if (typeof window === "undefined") return "";
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = generateUUID();
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}
```

- [ ] **Step 3: Create frontend/src/services/api.ts**

```typescript
// src/services/api.ts
// Centralized API client. Injects X-Session-ID header on every request.
// Imported by: src/hooks/useChat.ts, src/app/page.tsx

import { getSessionId } from "./session";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Citation {
  title: string;
  author: string | null;
  chunk_ref: string;
}

export interface MessageOut {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[] | null;
  follow_ups: string[] | null;
}

export interface ConversationOut {
  id: string;
  title: string;
}

export interface ChatResponse {
  conversation_id: string;
  message: MessageOut;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      "X-Session-ID": getSessionId(),
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail ?? "Request failed");
  }
  return res.json() as Promise<T>;
}

export const api = {
  listConversations: (): Promise<ConversationOut[]> =>
    request("/api/v1/conversations"),

  getMessages: (conversationId: string): Promise<MessageOut[]> =>
    request(`/api/v1/conversations/${conversationId}/messages`),

  deleteConversation: (conversationId: string): Promise<void> =>
    request(`/api/v1/conversations/${conversationId}`, { method: "DELETE" }),

  sendMessage: (
    message: string,
    conversationId?: string
  ): Promise<ChatResponse> =>
    request("/api/v1/chat", {
      method: "POST",
      body: JSON.stringify({ message, conversation_id: conversationId ?? null }),
    }),
};
```

- [ ] **Step 4: Commit**

```bash
git add frontend/
git commit -m "feat: Next.js frontend scaffold with session and API client"
```

---

## Task 14: Frontend — Conversation List and Chat Pages

**Files:**
- Create: `frontend/src/hooks/useChat.ts`
- Create: `frontend/src/components/ConversationList.tsx`
- Create: `frontend/src/components/ChatMessage.tsx`
- Create: `frontend/src/components/CitationBadge.tsx`
- Create: `frontend/src/components/FollowUpSuggestions.tsx`
- Create: `frontend/src/app/page.tsx`
- Create: `frontend/src/app/chat/[id]/page.tsx`
- Create: `frontend/src/app/layout.tsx`

- [ ] **Step 1: Create frontend/src/hooks/useChat.ts**

```typescript
// src/hooks/useChat.ts
// Manages chat state: messages, sending, loading, error.
// Used by: src/app/chat/[id]/page.tsx

"use client";

import { useCallback, useState } from "react";
import { api, MessageOut } from "../services/api";

export function useChat(conversationId: string) {
  const [messages, setMessages] = useState<MessageOut[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadHistory = useCallback(async () => {
    try {
      const msgs = await api.getMessages(conversationId);
      setMessages(msgs);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load messages");
    }
  }, [conversationId]);

  const sendMessage = useCallback(
    async (text: string): Promise<string | null> => {
      const userMsg: MessageOut = {
        id: crypto.randomUUID(),
        role: "user",
        content: text,
        citations: null,
        follow_ups: null,
      };
      setMessages((prev) => [...prev, userMsg]);
      setLoading(true);
      setError(null);

      try {
        const response = await api.sendMessage(text, conversationId);
        setMessages((prev) => [...prev, response.message]);
        return response.conversation_id;
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to send message");
        return null;
      } finally {
        setLoading(false);
      }
    },
    [conversationId]
  );

  return { messages, loading, error, loadHistory, sendMessage };
}
```

- [ ] **Step 2: Create frontend/src/components/CitationBadge.tsx**

```tsx
// src/components/CitationBadge.tsx
// Displays a single source citation as a small pill.
// Used by: src/components/ChatMessage.tsx

import { Citation } from "../services/api";

interface Props {
  citation: Citation;
  index: number;
}

export function CitationBadge({ citation, index }: Props) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-indigo-950 border border-indigo-700 px-2 py-0.5 text-xs text-indigo-300">
      [{index + 1}] {citation.title}
      {citation.author && <span className="text-indigo-500">· {citation.author}</span>}
    </span>
  );
}
```

- [ ] **Step 3: Create frontend/src/components/FollowUpSuggestions.tsx**

```tsx
// src/components/FollowUpSuggestions.tsx
// Row of clickable follow-up question chips shown below assistant messages.
// Used by: src/app/chat/[id]/page.tsx

interface Props {
  suggestions: string[];
  onSelect: (suggestion: string) => void;
}

export function FollowUpSuggestions({ suggestions, onSelect }: Props) {
  if (suggestions.length === 0) return null;
  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {suggestions.map((s, i) => (
        <button
          key={i}
          onClick={() => onSelect(s)}
          className="rounded-full border border-indigo-700 bg-indigo-950/50 px-3 py-1 text-xs text-indigo-300 hover:bg-indigo-900 transition-colors"
        >
          {s}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 4: Create frontend/src/components/ChatMessage.tsx**

```tsx
// src/components/ChatMessage.tsx
// Renders a single chat message bubble with optional citations and follow-ups.
// Used by: src/app/chat/[id]/page.tsx

import { CitationBadge } from "./CitationBadge";
import { FollowUpSuggestions } from "./FollowUpSuggestions";
import { MessageOut } from "../services/api";

interface Props {
  message: MessageOut;
  onFollowUp: (q: string) => void;
}

export function ChatMessage({ message, onFollowUp }: Props) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`max-w-2xl rounded-2xl px-4 py-3 text-sm ${
          isUser
            ? "bg-indigo-600 text-white"
            : "bg-slate-800 text-slate-100 border border-slate-700"
        }`}
      >
        <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>

        {!isUser && message.citations && message.citations.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1 border-t border-slate-700 pt-2">
            {message.citations.map((c, i) => (
              <CitationBadge key={c.chunk_ref} citation={c} index={i} />
            ))}
          </div>
        )}

        {!isUser && message.follow_ups && (
          <FollowUpSuggestions
            suggestions={message.follow_ups}
            onSelect={onFollowUp}
          />
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Create frontend/src/components/ConversationList.tsx**

```tsx
// src/components/ConversationList.tsx
// Sidebar list of past conversations with a new chat button.
// Used by: src/app/page.tsx

"use client";

import Link from "next/link";
import { ConversationOut } from "../services/api";

interface Props {
  conversations: ConversationOut[];
  onNewChat: () => void;
}

export function ConversationList({ conversations, onNewChat }: Props) {
  return (
    <div className="flex flex-col gap-2">
      <button
        onClick={onNewChat}
        className="w-full rounded-xl border border-indigo-600 bg-indigo-600/20 px-4 py-2.5 text-sm font-medium text-indigo-300 hover:bg-indigo-600/30 transition-colors"
      >
        + New Conversation
      </button>
      {conversations.length === 0 && (
        <p className="text-center text-xs text-slate-500 mt-4">
          No conversations yet. Ask your first question.
        </p>
      )}
      {conversations.map((conv) => (
        <Link
          key={conv.id}
          href={`/chat/${conv.id}`}
          className="rounded-xl border border-slate-700 bg-slate-800/50 px-4 py-3 text-sm text-slate-300 hover:bg-slate-800 transition-colors truncate"
        >
          {conv.title}
        </Link>
      ))}
    </div>
  );
}
```

- [ ] **Step 6: Create frontend/src/app/layout.tsx**

```tsx
// src/app/layout.tsx
// Root layout. Sets dark background and initialises session on first load.

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Terapanth Learning Platform",
  description: "Guided learning for Jain philosophy",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-slate-950 text-slate-100 antialiased">
        {children}
      </body>
    </html>
  );
}
```

- [ ] **Step 7: Create frontend/src/app/page.tsx**

```tsx
// src/app/page.tsx
// Home page: lists past conversations and provides a new chat entry point.

"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ConversationList } from "../components/ConversationList";
import { api, ConversationOut } from "../services/api";

export default function HomePage() {
  const router = useRouter();
  const [conversations, setConversations] = useState<ConversationOut[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.listConversations()
      .then(setConversations)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  function handleNewChat() {
    router.push("/chat/new");
  }

  return (
    <main className="mx-auto max-w-xl px-4 py-12">
      <h1 className="mb-2 text-2xl font-semibold text-slate-100">
        Terapanth Learning
      </h1>
      <p className="mb-8 text-sm text-slate-400">
        Ask questions about Jain philosophy, guided by authoritative texts.
      </p>
      {loading ? (
        <p className="text-sm text-slate-500">Loading...</p>
      ) : (
        <ConversationList
          conversations={conversations}
          onNewChat={handleNewChat}
        />
      )}
    </main>
  );
}
```

- [ ] **Step 8: Create frontend/src/app/chat/[id]/page.tsx**

```tsx
// src/app/chat/[id]/page.tsx
// Active chat view. Loads history and handles message sending.

"use client";

import { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { ChatMessage } from "../../../components/ChatMessage";
import { useChat } from "../../../hooks/useChat";

export default function ChatPage() {
  const params = useParams();
  const router = useRouter();
  const rawId = params.id as string;
  const isNew = rawId === "new";

  const [activeId, setActiveId] = useState<string>(isNew ? "" : rawId);
  const { messages, loading, error, loadHistory, sendMessage } = useChat(activeId);
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (activeId) loadHistory();
  }, [activeId, loadHistory]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend(text: string) {
    if (!text.trim()) return;
    setInput("");
    const returnedId = await sendMessage(text);
    if (returnedId && isNew) {
      setActiveId(returnedId);
      router.replace(`/chat/${returnedId}`, { scroll: false });
    }
  }

  return (
    <div className="flex h-screen flex-col">
      <header className="border-b border-slate-800 px-4 py-3 text-sm text-slate-400">
        <a href="/" className="hover:text-slate-200">← All conversations</a>
      </header>

      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto max-w-2xl">
          {messages.length === 0 && !loading && (
            <p className="text-center text-sm text-slate-500">
              Ask anything about Jain philosophy to start.
            </p>
          )}
          {messages.map((msg) => (
            <ChatMessage
              key={msg.id}
              message={msg}
              onFollowUp={(q) => handleSend(q)}
            />
          ))}
          {loading && (
            <div className="flex justify-start mb-4">
              <div className="rounded-2xl bg-slate-800 border border-slate-700 px-4 py-3 text-sm text-slate-400">
                Thinking...
              </div>
            </div>
          )}
          {error && (
            <p className="text-center text-xs text-red-400">{error}</p>
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="border-t border-slate-800 px-4 py-4">
        <div className="mx-auto flex max-w-2xl gap-2">
          <input
            className="flex-1 rounded-xl border border-slate-700 bg-slate-800 px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            placeholder="Ask about Jain philosophy..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend(input)}
            disabled={loading}
          />
          <button
            onClick={() => handleSend(input)}
            disabled={loading || !input.trim()}
            className="rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50 transition-colors"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 9: Start frontend and verify golden path**

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000`. Verify:
1. Home page loads with "New Conversation" button
2. Clicking "New Conversation" navigates to `/chat/new`
3. Sending a message (with backend running) returns an answer with citations
4. Follow-up chips appear and clicking one pre-fills + sends the question
5. Navigating back shows the conversation in the list

- [ ] **Step 10: Commit**

```bash
git add frontend/src/
git commit -m "feat: chat UI with conversation list and citation display"
```

---

## Task 15: CLAUDE.md and Final Wiring

**Files:**
- Create: `CLAUDE.md`
- Create: `.gitignore`

- [ ] **Step 1: Create .gitignore**

```
# .gitignore
.env
uploads/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
backend/.venv/
node_modules/
frontend/.next/
frontend/out/
.superpowers/
```

- [ ] **Step 2: Create CLAUDE.md**

```markdown
# Terapanth Learning Platform

## Overview
MoE-inspired Jain (Terapanth) philosophy Q&A platform. Phase 1: platform-curated RAG with hybrid retrieval. Phase 2: MoE routing + expert agents. Phase 3: Graph RAG + Neo4j.

## Tech Stack
- Backend: FastAPI (Python 3.12), SQLAlchemy 2.0 async, PostgreSQL 16 + pgvector, Alembic
- Frontend: Next.js 15, TypeScript, Tailwind
- LLM: Groq (primary) / Ollama (local backup) via OpenAI-compatible API
- Reranker: Cohere Rerank API
- Infra: Docker + docker-compose

## How to Run

### Backend (local)
```bash
cd backend
pip install -r requirements-dev.txt
cp .env.example .env  # fill in keys
docker compose up postgres -d
alembic upgrade head
uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

### Full stack
```bash
docker compose up --build
```

## Project Structure
See `docs/superpowers/specs/2026-05-06-phase1-design.md` for full architecture.

## Key Decisions
- Documents are platform-curated only — no user uploads
- Guest sessions via UUID in localStorage, stored in PostgreSQL
- Contextual embedding: LLM-generated summary prepended to each chunk before embedding
- Hybrid retrieval: pgvector (semantic) + PostgreSQL full-text search (BM25-style)
- Reranking: Cohere Rerank with trust_score multiplier
- LLM provider abstraction: swap Groq ↔ Ollama via LLM_PROVIDER env var
```

- [ ] **Step 3: Final full test run**

```bash
cd backend && pytest tests/ -v
```
Expected: all tests pass

- [ ] **Step 4: Final commit**

```bash
git add CLAUDE.md .gitignore
git commit -m "chore: CLAUDE.md, gitignore, project wiring complete"
```
