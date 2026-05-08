# app/chat/session_service.py
# Session CRUD and sliding window message history for guest sessions.
# Called by: app/chat/service.py

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, Message
from app.db.models import Session as SessionModel

_WINDOW_SIZE = 10


async def get_or_create_session(
    session_id: uuid.UUID, db: AsyncSession
) -> SessionModel:
    existing = await db.get(SessionModel, session_id)
    if existing:
        return existing

    session = SessionModel(id=session_id)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get_or_create_conversation(
    session_id: uuid.UUID,
    conversation_id: uuid.UUID | None,
    first_message: str,
    db: AsyncSession,
) -> Conversation:
    if conversation_id:
        conv = await db.get(Conversation, conversation_id)
        if conv and conv.session_id == session_id:
            return conv

    title = first_message[:60]
    conv = Conversation(session_id=session_id, title=title)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


async def get_sliding_window(
    conversation_id: uuid.UUID, db: AsyncSession
) -> list[Message]:
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(_WINDOW_SIZE)
    )
    messages = result.scalars().all()
    return list(reversed(messages))


async def save_message(
    conversation_id: uuid.UUID,
    role: str,
    content: str,
    db: AsyncSession,
    citations: list | None = None,
    follow_ups: list[str] | None = None,
) -> Message:
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        citations=citations,
        follow_ups=follow_ups,
    )
    db.add(msg)

    conv = await db.get(Conversation, conversation_id)
    if conv:
        conv.last_active = datetime.utcnow()

    await db.commit()
    await db.refresh(msg)
    return msg
