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
