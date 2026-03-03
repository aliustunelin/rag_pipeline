import os

from openai import OpenAI

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")


class Embedder:
    """Wrapper around OpenAI openai/text-embedding-3-small model."""

    def __init__(self, api_key: str, model: str = "openai/text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of texts and returns their vector representations."""
        if not texts:
            return []

        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
        )

        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        """Embeds a single query string."""
        response = self.client.embeddings.create(
            input=[query],
            model=self.model,
        )
        return response.data[0].embedding
