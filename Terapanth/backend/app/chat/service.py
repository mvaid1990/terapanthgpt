# app/chat/service.py
# Orchestrates the full query pipeline: session → expand → retrieve → generate → store.
# Called by: app/chat/router.py
# Phase 2: replace _build_system_prompt() with a router that selects expert prompts.

import json
import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.models import ChatResponse, Citation, MessageOut
from app.chat.query_expander import expand_query
from app.chat.session_service import (
    get_or_create_conversation,
    get_or_create_session,
    get_sliding_window,
    save_message,
)
from app.llm.provider import LLMProvider
from app.retrieval.hybrid import hybrid_retrieve
from app.retrieval.models import RerankResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a knowledgeable and respectful guide to Jain (Terapanth) philosophy.
Answer questions accurately based only on the provided context passages.
Always cite the source document for each claim. Never speculate or add information
not present in the context. If the context does not contain enough information, say so.

After your answer, provide a JSON structure in this exact format:
{"answer": "your answer here", "follow_ups": ["question 1", "question 2"]}
"""


async def answer_question(
    session_id: uuid.UUID,
    message: str,
    conversation_id: uuid.UUID | None,
    db: AsyncSession,
    llm: LLMProvider,
    jina_api_key: str,
) -> ChatResponse:
    session = await get_or_create_session(session_id, db)
    conversation = await get_or_create_conversation(
        session_id=session.id,
        conversation_id=conversation_id,
        first_message=message,
        db=db,
    )

    history = await get_sliding_window(conversation.id, db)

    query_variants = await expand_query(message, llm)
    primary_query = query_variants[0]

    query_embedding = await llm.embed(primary_query)

    retrieval_results = await hybrid_retrieve(
        query=primary_query,
        query_embedding=query_embedding,
        db=db,
        jina_api_key=jina_api_key,
        top_k=5,
    )

    context_text = _format_context(retrieval_results)
    chat_messages = _build_messages(history, message, context_text)

    raw_response = await llm.chat(chat_messages, temperature=0.2, max_tokens=1024)
    answer, follow_ups = _parse_response(raw_response)
    citations = _build_citations(retrieval_results)

    saved = await save_message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        db=db,
        citations=[c.model_dump() for c in citations],
        follow_ups=follow_ups,
    )

    return ChatResponse(
        conversation_id=conversation.id,
        message=MessageOut(
            id=saved.id,
            role="assistant",
            content=answer,
            citations=citations,
            follow_ups=follow_ups,
        ),
    )


def _format_context(results: list[RerankResult]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] Source: {r.chunk.doc_title}"
            + (f" by {r.chunk.doc_author}" if r.chunk.doc_author else "")
            + f"\n{r.chunk.content}"
        )
    return "\n\n".join(parts)


def _build_messages(
    history: list, current_message: str, context: str
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {current_message}",
    })
    return messages


def _parse_response(raw: str) -> tuple[str, list[str]]:
    try:
        data = json.loads(raw)
        return data.get("answer", raw), data.get("follow_ups", [])
    except json.JSONDecodeError:
        return raw, []


def _build_citations(results: list[RerankResult]) -> list[Citation]:
    return [
        Citation(
            title=r.chunk.doc_title,
            author=r.chunk.doc_author,
            chunk_ref=str(r.chunk.chunk_id),
        )
        for r in results
    ]
