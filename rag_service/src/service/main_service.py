import logging
from collections.abc import AsyncGenerator

from .retriever import Retriever
from .context_builder import ContextBuilder
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class MainService:
    """Orchestrator: coordinates retrieval → context building → LLM generation.

    Flow:
    1. User query comes in
    2. Retriever searches FAISS for relevant chunks
    3. ContextBuilder fetches original content and formats context
    4. LLMClient generates response using context + query
    """

    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        llm_client: LLMClient,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.llm_client = llm_client
        self.top_k = top_k

    async def query(self, user_query: str) -> dict:
        """Full RAG pipeline: search → context → LLM → response."""
        search_results = self.retriever.search(user_query, top_k=self.top_k)

        if not search_results:
            return {
                "answer": "Üzgünüm, sorunuzla ilgili bilgi bulunamadı. Lütfen verilerin indexlendiğinden emin olun.",
                "sources": [],
            }

        context = self.context_builder.build_context(search_results)

        answer = await self.llm_client.generate(user_query, context)

        sources = self._extract_sources(search_results)

        return {
            "answer": answer,
            "sources": sources,
        }

    async def query_stream(self, user_query: str) -> AsyncGenerator[str, None]:
        """Streaming RAG pipeline: search → context → LLM stream."""
        search_results = self.retriever.search(user_query, top_k=self.top_k)

        if not search_results:
            yield "Üzgünüm, sorunuzla ilgili bilgi bulunamadı. Lütfen verilerin indexlendiğinden emin olun."
            return

        context = self.context_builder.build_context(search_results)

        async for chunk in self.llm_client.generate_stream(user_query, context):
            yield chunk

    def _extract_sources(self, search_results: list[dict]) -> list[dict]:
        """Extracts source metadata from search results for reference."""
        sources = []
        seen = set()
        for result in search_results:
            meta = result["metadata"]
            source_key = meta.get("source", "")
            date = meta.get("date", "")
            key = f"{source_key}:{date}"
            if key not in seen:
                seen.add(key)
                source_info = {
                    "file": source_key,
                    "type": meta.get("source_type", ""),
                    "score": result.get("score", 0),
                }
                if date:
                    source_info["date"] = date
                if "affected_package" in meta:
                    source_info["affected_package"] = meta["affected_package"]
                if "affected_clause" in meta:
                    source_info["affected_clause"] = meta["affected_clause"]
                sources.append(source_info)
        return sources

    def reload_index(self) -> bool:
        """Reloads FAISS index from disk (after re-indexing)."""
        return self.retriever.load_index()
