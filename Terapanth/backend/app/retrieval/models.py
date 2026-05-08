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
