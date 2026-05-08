# app/retrieval/reranker.py
# Reranks retrieved chunks using Jina Rerank API with trust_score boost.
# Falls back to score-based ranking if no API key is provided.
# Called by: app/retrieval/hybrid.py

import httpx

from app.retrieval.models import RerankResult, RetrievedChunk

_JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
_JINA_MODEL = "jina-reranker-v3"


async def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int,
    jina_api_key: str,
) -> list[RerankResult]:
    """Rerank chunks against query using Jina. Applies trust_score as a multiplicative boost.

    Falls back to raw score × trust_score ranking when no API key is provided.
    Returns top_k results ordered by final score descending.
    """
    if not chunks:
        return []

    if not jina_api_key:
        fallback = [
            RerankResult(chunk=c, rerank_score=c.score * c.trust_score)
            for c in chunks
        ]
        fallback.sort(key=lambda r: r.rerank_score, reverse=True)
        return fallback[:top_k]

    payload = {
        "model": _JINA_MODEL,
        "query": query,
        "top_n": len(chunks),
        "documents": [c.content for c in chunks],
        "return_documents": False,
    }
    headers = {
        "Authorization": f"Bearer {jina_api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(_JINA_RERANK_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    results = [
        RerankResult(
            chunk=chunks[r["index"]],
            rerank_score=r["relevance_score"] * chunks[r["index"]].trust_score,
        )
        for r in data["results"]
    ]

    results.sort(key=lambda r: r.rerank_score, reverse=True)
    return results[:top_k]
