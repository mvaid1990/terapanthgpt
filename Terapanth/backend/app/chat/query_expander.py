# app/chat/query_expander.py
# Generates 2-3 alternative phrasings of a user query for broader retrieval.
# Called by: app/chat/service.py

import json
import logging

from app.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

_EXPAND_PROMPT = """\
Generate 2-3 alternative phrasings of the following question about Jain philosophy.
Return ONLY a valid JSON array of strings. No explanation.

Question: {query}

Example output: ["alternate phrasing 1", "alternate phrasing 2", "alternate phrasing 3"]
"""


async def expand_query(query: str, llm: LLMProvider) -> list[str]:
    """Return list of query variants including fallback to original on failure."""
    try:
        raw = await llm.chat(
            [{"role": "user", "content": _EXPAND_PROMPT.format(query=query)}],
            temperature=0.3,
            max_tokens=200,
        )
        variants = json.loads(raw)
        if isinstance(variants, list) and all(isinstance(v, str) for v in variants):
            return variants
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Query expansion failed, using original. Error: %s", exc)
    return [query]
