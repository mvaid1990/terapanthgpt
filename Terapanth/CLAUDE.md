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

### Infrastructure only (postgres)
```bash
docker compose up -d
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
- All Python dependency versions are pinned exactly in requirements.txt
- Files split by responsibility — each file owns one concern
- Every module has a file-level comment: what it does and which files import it
