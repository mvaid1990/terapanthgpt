# app/llm/provider.py
# Protocol (interface) for LLM providers.
# Imported by: app/llm/groq_provider.py, app/llm/ollama_provider.py,
#              app/chat/service.py, app/ingestion/contextual_embedder.py,
#              app/chat/query_expander.py

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str: ...

    async def embed(self, text: str) -> list[float]: ...
