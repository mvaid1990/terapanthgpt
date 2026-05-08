# app/db/models.py
# SQLAlchemy ORM models for all tables.
# Imported by: all service modules and Alembic migrations.

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Integer,
    Numeric,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    last_active: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )

    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    last_active: Mapped[datetime] = mapped_column(server_default=func.now())

    session: Mapped["Session"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (CheckConstraint("role IN ('user', 'assistant')", name="ck_role"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[Any] | None] = mapped_column(JSONB, nullable=True)
    follow_ups: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'approved', 'rejected')", name="ck_doc_status"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str | None] = mapped_column(Text, nullable=True)
    sect: Mapped[str | None] = mapped_column(Text, nullable=True)
    topic: Mapped[str | None] = mapped_column(Text, nullable=True)
    trust_score: Mapped[float] = mapped_column(Numeric(3, 2), default=0.8)
    language: Mapped[str] = mapped_column(Text, default="en")
    status: Mapped[str] = mapped_column(Text, default="pending")
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)
    ts_vector: Mapped[Any | None] = mapped_column(TSVECTOR, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    document: Mapped["Document"] = relationship(back_populates="chunks")
