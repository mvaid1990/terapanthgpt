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
