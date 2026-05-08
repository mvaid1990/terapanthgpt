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
