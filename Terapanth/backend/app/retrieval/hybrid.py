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
    jina_api_key: str,
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
        jina_api_key=jina_api_key,
    )
