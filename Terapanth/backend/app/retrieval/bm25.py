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
