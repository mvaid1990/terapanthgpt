# app/chat/models.py
# Pydantic request/response schemas for the chat API.
# Imported by: app/chat/router.py, app/chat/service.py

import uuid
from pydantic import BaseModel


class Citation(BaseModel):
    title: str
    author: str | None
    chunk_ref: str  # chunk_id as string


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    citations: list[Citation] | None = None
    follow_ups: list[str] | None = None

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: uuid.UUID
    title: str

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    message: str
    conversation_id: uuid.UUID | None = None


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    message: MessageOut
