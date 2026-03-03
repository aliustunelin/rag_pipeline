import logging
import os
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from ..utils.prompts import SYSTEM_PROMPT, USER_PROMPT

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")


class LLMClient:
    """Async OpenAI openai/gpt-4o-mini client with streaming support."""

    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        self.model = model

    async def generate(self, query: str, context: str) -> str:
        """Non-streaming generation for simple responses."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": USER_PROMPT.format(query=query)},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content

    async def generate_stream(self, query: str, context: str) -> AsyncGenerator[str, None]:
        """Streaming generation - yields chunks as they arrive."""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": USER_PROMPT.format(query=query)},
            ],
            temperature=0.1,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
