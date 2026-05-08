# Terapanth Learning Platform — Phase 1 Design Spec

**Date:** 2026-05-06  
**Scope:** Phase 1 MVP — platform-curated RAG, single expert, admin ingestion, guest chat  
**Stack:** FastAPI · Next.js · PostgreSQL + pgvector · Groq / Ollama · Docker

---

## 1. Vision

An intelligent, citation-backed Q&A platform for Jain (Terapanth) philosophy. Users ask questions in natural language and receive accurate, sourced answers with suggested follow-up questions. Phase 1 establishes the core retrieval and answer pipeline. Phases 2 and 3 layer MoE routing, expert agents, and graph RAG on top.

---

## 2. Scope Boundaries

**In Phase 1:**
- Admin document ingestion pipeline (upload → approve → chunk → contextual embed → store)
- Hybrid retrieval (vector + BM25) with reranking
- Single expert prompt (general Jain guide persona)
- Guest chat sessions (no accounts)
- Conversation history per session (sliding window, last 10 messages)
- Citations and follow-up suggestions in every response
- LLM provider abstraction (Groq primary, Ollama backup)

**Out of Phase 1 (deferred):**
- MoE router and multiple expert agents (Phase 2)
- User accounts, knowledge levels, adaptive learning (Phase 2)
- Graph RAG, Neo4j, PageRank (Phase 3)
- Community contributions or user uploads (never — platform-curated only)
- Multi-language support, voice interface

---

## 3. Architecture

### 3.1 High-Level Components

```
Admin Browser
    │
    ▼
FastAPI (Admin Router)
    │
    ├── Document Upload & Approval
    ├── Chunking Service
    ├── Contextual Embedding Service
    └── PostgreSQL + pgvector

User Browser (Next.js)
    │
    ▼
FastAPI (Chat Router)
    │
    ├── Session Service
    ├── Query Expansion Service
    ├── Hybrid Retrieval Service (pgvector + BM25)
    ├── Reranker Service (Cohere / BGE)
    ├── LLM Service (Groq / Ollama)
    └── PostgreSQL (sessions, conversations, messages)
```

### 3.2 Project Structure

```
terapanth/
  backend/
    app/
      main.py                   # FastAPI app, middleware, lifespan
      config.py                 # Settings via pydantic-settings, LLM provider config
      ingestion/
        router.py               # Admin endpoints: upload, approve, reject, list
        service.py              # Orchestrates chunk → embed → store pipeline
        chunker.py              # Text splitting logic (512 tok / 64 overlap)
        contextual_embedder.py  # Prepends LLM-generated summary before embedding
        extractor.py            # PDF/text extraction
        models.py               # Pydantic schemas: DocumentUpload, DocumentStatus
      retrieval/
        router.py               # Internal — not exposed directly, used by chat
        hybrid.py               # Merges pgvector results + BM25 results
        vector_store.py         # pgvector queries
        bm25.py                 # PostgreSQL full-text search queries
        reranker.py             # Cohere Rerank or BGE-reranker, trust_score boost
        models.py               # RetrievedChunk, RerankResult
      chat/
        router.py               # POST /api/v1/chat, GET /api/v1/conversations
        service.py              # Orchestrates query expand → retrieve → rerank → generate
        query_expander.py       # Generates 2-3 alternate phrasings via LLM
        session_service.py      # Create/load session, sliding window history
        models.py               # ChatRequest, ChatResponse, Message, Citation
      llm/
        provider.py             # Abstract LLMProvider interface
        groq_provider.py        # Groq implementation
        ollama_provider.py      # Ollama implementation
        factory.py              # Returns provider from config
      db/
        base.py                 # SQLAlchemy async engine, session factory
        migrations/             # Alembic migration files
    requirements.txt            # All versions pinned exactly
    Dockerfile
  frontend/
    src/
      app/
        page.tsx                # Home — conversation list + new chat button
        chat/[id]/page.tsx      # Active chat view
      components/
        ChatMessage.tsx         # Renders single message with citations
        CitationBadge.tsx       # Inline source citation pill
        FollowUpSuggestions.tsx # Row of clickable follow-up question chips
        ConversationList.tsx    # Sidebar list of past conversations
      services/
        api.ts                  # Centralized API client, injects X-Session-ID
        session.ts              # localStorage UUID management
      hooks/
        useChat.ts              # Chat state, streaming, message history
  docker-compose.yml
  .env.example
```

---

## 4. Database Schema

```sql
-- sessions: anonymous users identified by UUID from browser localStorage
CREATE TABLE sessions (
    id          UUID PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- conversations: each chat thread belongs to a session
CREATE TABLE conversations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    title       TEXT NOT NULL,              -- auto-generated from first message
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- messages: individual turns within a conversation
CREATE TABLE messages (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id  UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    citations        JSONB,                 -- [{title, author, chunk_ref}]
    follow_ups       JSONB,                 -- ["question1", "question2"]
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- documents: uploaded and approved source texts
CREATE TABLE documents (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title        TEXT NOT NULL,
    author       TEXT,
    sect         TEXT,                      -- e.g. "Terapanth", "Digambara"
    topic        TEXT,                      -- e.g. "philosophy", "practice"
    trust_score  NUMERIC(3,2) NOT NULL DEFAULT 0.8,
    language     TEXT NOT NULL DEFAULT 'en',
    status       TEXT NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending', 'approved', 'rejected')),
    file_path    TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- chunks: processed text segments from approved documents
CREATE TABLE chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content      TEXT NOT NULL,             -- original chunk text
    context      TEXT NOT NULL,             -- LLM-generated contextual summary
    chunk_index  INTEGER NOT NULL,
    embedding    vector(1536),              -- pgvector column
    ts_vector    TSVECTOR,                  -- for BM25 full-text search
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON chunks USING GIN (ts_vector);
```

---

## 5. Ingestion Pipeline

**Trigger:** Admin uploads a document via `POST /api/v1/admin/documents`.

**Steps:**

1. **Upload** — save file to disk, create `documents` row with `status=pending`, extract raw text (PyMuPDF for PDF, plain read for .txt).
2. **Admin review** — `PATCH /api/v1/admin/documents/{id}/approve` or `/reject`. Rejection requires a reason.
3. **Chunk** — on approval, split text into 512-token chunks with 64-token overlap. Respect paragraph boundaries where possible.
4. **Contextual embedding** — for each chunk, call the LLM with:
   ```
   Document: {title} by {author}. Topic: {topic}.
   Summarize in 1-2 sentences what the following passage covers, 
   so it can be retrieved in isolation:
   {chunk_text}
   ```
   Prepend the summary to the chunk text. Embed the combined string.
5. **Store** — insert `chunks` rows with `content`, `context`, `embedding`, and `ts_vector` (generated by PostgreSQL trigger on `content`).

**File size limit:** 50MB. Documents over this limit are rejected at upload with a clear error.

---

## 6. Query Pipeline

**Trigger:** User sends `POST /api/v1/chat` with `{ conversation_id, message }` and `X-Session-ID` header.

**Steps:**

1. **Session** — load or create session from header. Load conversation. Fetch last 10 messages as chat history.
2. **Query expansion** — call LLM to generate 2-3 alternate phrasings of the user's question. All variants used in retrieval.
3. **Hybrid retrieval** — for each query variant:
   - Vector search: top-20 chunks by cosine similarity (pgvector)
   - BM25 search: top-20 chunks by full-text rank (PostgreSQL `ts_rank`)
   - Merge and deduplicate by chunk ID (union).
4. **Rerank** — score all candidates against the original query using Cohere Rerank API (or `BAAI/bge-reranker-base` self-hosted). Apply `trust_score` multiplier from the parent document. Select top-5.
5. **Generate** — call LLM with:
   - System prompt: Jain guide persona, enforce citations, no hallucination
   - Chat history (last 10 messages)
   - Retrieved chunks as context
   - Instruction to generate 2-3 follow-up questions
6. **Store and return** — save assistant message to DB. Return `{ answer, citations, follow_ups }`.

---

## 7. LLM Provider Abstraction

`llm/provider.py` defines an abstract interface:

```python
class LLMProvider(Protocol):
    async def chat(self, messages: list[Message], **kwargs) -> str: ...
    async def embed(self, text: str) -> list[float]: ...
```

`config.yml` (or env var `LLM_PROVIDER`) selects the implementation:
- `groq` → `llama-3.1-70b-versatile`, base URL `https://api.groq.com/openai/v1`
- `ollama` → `llama3.1:8b`, base URL `http://localhost:11434/v1`

Both use the OpenAI-compatible chat completions format. Switching providers requires no code changes.

---

## 8. Session & Chat History

- Session UUID generated once in browser, stored in `localStorage`, sent as `X-Session-ID` on every request.
- On first use, server creates a `sessions` row. Auto-creates a `conversations` row on first message if no `conversation_id` provided.
- Conversation title: auto-generated from the first user message (truncated to 60 chars).
- **Sliding window:** last 10 `messages` rows loaded and passed to LLM as chat history. Oldest messages outside the window are retained in DB (not deleted) for future Phase 2 summary compression.
- Sessions inactive for 30 days are deleted by a nightly cleanup job (cascade deletes conversations and messages).

---

## 9. API Endpoints

### Chat
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/chat` | Send message, get answer + citations + follow-ups |
| GET | `/api/v1/conversations` | List conversations for session |
| GET | `/api/v1/conversations/{id}/messages` | Load message history |
| DELETE | `/api/v1/conversations/{id}` | Delete a conversation |

### Admin (ingestion)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/admin/documents` | Upload document (multipart) |
| GET | `/api/v1/admin/documents` | List documents with status |
| PATCH | `/api/v1/admin/documents/{id}/approve` | Approve and trigger pipeline |
| PATCH | `/api/v1/admin/documents/{id}/reject` | Reject with reason |
| GET | `/api/v1/admin/documents/{id}` | Document detail + chunk count |

---

## 10. Coding Conventions

- **Dependency versions:** All packages pinned to exact versions in `requirements.txt`. No open ranges.
- **File organization:** Files split strictly by responsibility. Each file owns one concern. Feature-domain structure (not technical layer).
- **File-level comments:** Every module starts with a comment block: what it does, which other files import it.
- **No mutation:** Use immutable Pydantic models for data passing between layers.
- **Error handling:** Custom exception classes in `core/exceptions.py`. Global handler via `@app.exception_handler`. Never swallow exceptions.
- **No secrets in code:** All keys via environment variables. Validated at startup via `pydantic-settings`.

---

## 11. Extension Points for Phase 2 & 3

These interfaces are designed into Phase 1 so future phases don't require rewrites:

| Extension | How Phase 1 prepares for it |
|---|---|
| MoE Router | `chat/service.py` has a single `_select_prompt()` call — Phase 2 replaces it with a router |
| Expert prompts | Prompt lives in one place (`chat/service.py`) — Phase 2 externalizes to `experts/` module |
| User accounts | `sessions` table has an optional `user_id` FK ready to be populated in Phase 2 |
| Knowledge level | `sessions` table has a `knowledge_level` column (nullable, unused in Phase 1) |
| Graph retrieval | `retrieval/hybrid.py` accepts a list of retrieval strategies — Phase 3 appends a graph retriever |
| PageRank boost | Reranker accepts a score dict — Phase 3 passes PageRank scores alongside trust_score |

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| Incorrect religious interpretation | Admin approval gate before any document enters the index. trust_score boosts authoritative primary texts. |
| Poor retrieval for domain-specific terms | BM25 handles exact scripture/term matches that vector search misses. Query expansion handles paraphrasing. |
| LLM hallucination | System prompt enforces citations. Chunks are injected verbatim — answer must reference them. |
| Groq rate limits | Ollama fallback. LLM provider abstraction makes switching seamless. |
