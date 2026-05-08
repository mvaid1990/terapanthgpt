"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-05-06

"""
from typing import Sequence, Union
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Must run after extension is enabled so vector type is available
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("last_active", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("last_active", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.Text, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("citations", postgresql.JSONB, nullable=True),
        sa.Column("follow_ups", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="ck_role"),
    )

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("author", sa.Text, nullable=True),
        sa.Column("sect", sa.Text, nullable=True),
        sa.Column("topic", sa.Text, nullable=True),
        sa.Column("trust_score", sa.Numeric(3, 2), nullable=False, server_default="0.8"),
        sa.Column("language", sa.Text, nullable=False, server_default="en"),
        sa.Column("status", sa.Text, nullable=False, server_default="pending"),
        sa.Column("file_path", sa.Text, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("status IN ('pending', 'approved', 'rejected')", name="ck_doc_status"),
    )

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("context", sa.Text, nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("embedding", sa.Text, nullable=True),  # created as Text, altered to vector(768) below
        sa.Column("ts_vector", postgresql.TSVECTOR, nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    # Convert embedding column from Text to vector(768) now that pgvector is enabled
    op.execute(
        "ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(768) USING NULL"
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
        "ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS chunks_ts_vector_idx ON chunks USING GIN (ts_vector)"
    )

    op.execute("""
        CREATE OR REPLACE FUNCTION update_chunk_ts_vector()
        RETURNS trigger AS $$
        BEGIN
            NEW.ts_vector := to_tsvector('english', COALESCE(NEW.content, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)

    op.execute("""
        CREATE TRIGGER chunk_ts_vector_update
        BEFORE INSERT OR UPDATE OF content ON chunks
        FOR EACH ROW EXECUTE FUNCTION update_chunk_ts_vector()
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS chunk_ts_vector_update ON chunks")
    op.execute("DROP FUNCTION IF EXISTS update_chunk_ts_vector()")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("sessions")
