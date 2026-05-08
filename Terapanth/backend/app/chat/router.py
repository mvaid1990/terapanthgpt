# app/chat/router.py
# User-facing HTTP endpoints: send messages, list/delete conversations.
# Imported by: app/main.py (included as router with prefix /api/v1)

import uuid

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.models import ChatRequest, ChatResponse, ConversationOut, MessageOut
from app.chat.service import answer_question
from app.config import get_settings
from app.db.base import get_db
from app.db.models import Conversation, Message
from app.llm.factory import get_llm_provider

router = APIRouter(prefix="/api/v1", tags=["chat"])


def _parse_session_id(x_session_id: str = Header(...)) -> uuid.UUID:
    try:
        return uuid.UUID(x_session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid X-Session-ID header")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    session_id: uuid.UUID = Depends(_parse_session_id),
    db: AsyncSession = Depends(get_db),
):
    llm = get_llm_provider()
    settings = get_settings()
    return await answer_question(
        session_id=session_id,
        message=body.message,
        conversation_id=body.conversation_id,
        db=db,
        llm=llm,
        jina_api_key=settings.jina_api_key,
    )


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(
    session_id: uuid.UUID = Depends(_parse_session_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.session_id == session_id)
        .order_by(Conversation.last_active.desc())
    )
    return [ConversationOut.model_validate(c) for c in result.scalars().all()]


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
async def get_messages(
    conversation_id: uuid.UUID,
    session_id: uuid.UUID = Depends(_parse_session_id),
    db: AsyncSession = Depends(get_db),
):
    conv = await db.get(Conversation, conversation_id)
    if not conv or conv.session_id != session_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    return [MessageOut.model_validate(m) for m in result.scalars().all()]


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: uuid.UUID,
    session_id: uuid.UUID = Depends(_parse_session_id),
    db: AsyncSession = Depends(get_db),
):
    conv = await db.get(Conversation, conversation_id)
    if not conv or conv.session_id != session_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await db.delete(conv)
    await db.commit()
