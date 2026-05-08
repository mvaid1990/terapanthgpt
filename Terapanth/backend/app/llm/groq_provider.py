# app/llm/groq_provider.py
# Groq LLM provider using the OpenAI-compatible API.
# Returned by app/llm/factory.py when LLM_PROVIDER=groq.

from openai import AsyncOpenAI

from app.core.exceptions import LLMError


class GroqProvider:
    """Chat via Groq API; embeddings via OpenAI API."""

    def __init__(
        self,
        llm_model: str,
        llm_base_url: str,
        groq_api_key: str,
        embedding_model: str,
        embedding_base_url: str,
        openai_api_key: str,
    ) -> None:
        self._chat_client = AsyncOpenAI(
            api_key=groq_api_key, base_url=llm_base_url
        )
        self._embed_client = AsyncOpenAI(
            api_key=openai_api_key, base_url=embedding_base_url
        )
        self._llm_model = llm_model
        self._embedding_model = embedding_model

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        try:
            response = await self._chat_client.chat.completions.create(
                model=self._llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMError(f"Groq chat failed: {exc}") from exc

    async def embed(self, text: str) -> list[float]:
        try:
            response = await self._embed_client.embeddings.create(
                model=self._embedding_model, input=text
            )
            return response.data[0].embedding
        except Exception as exc:
            raise LLMError(f"Embedding failed: {exc}") from exc
