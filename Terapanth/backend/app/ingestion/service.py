# app/ingestion/service.py
# Orchestrates the document ingestion pipeline: extract -> chunk -> embed -> store.
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
