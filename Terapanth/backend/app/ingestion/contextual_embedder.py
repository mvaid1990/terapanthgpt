# app/ingestion/contextual_embedder.py
# Generates a contextual summary for each chunk and embeds the combined text.
# This is Anthropic's "contextual retrieval" technique — summarising the chunk
# within its document context before embedding significantly improves recall.
# Called by: app/ingestion/service.py

from app.llm.provider import LLMProvider

_CONTEXT_PROMPT = """\
Document: {title} by {author}. Topic: {topic}.

Summarize in 1-2 sentences what the following passage covers, so it can be \
retrieved in isolation without the rest of the document:

{chunk}
"""


async def build_contextual_embedding(
    chunk: str,
    doc_title: str,
    doc_author: str | None,
    doc_topic: str | None,
    llm: LLMProvider,
) -> tuple[str, list[float]]:
    """Return (context_summary, embedding) for a single chunk.

    The embedding is computed on "summary + chunk" so the vector captures
    both location context and content.
    """
    prompt = _CONTEXT_PROMPT.format(
        title=doc_title,
        author=doc_author or "Unknown",
        topic=doc_topic or "general",
        chunk=chunk,
    )
    context = await llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150,
    )
    combined = f"{context}\n\n{chunk}"
    embedding = await llm.embed(combined)
    return context, embedding
