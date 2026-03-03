import json
import os
import logging

import numpy as np
import faiss
from openai import OpenAI

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")


class Retriever:
    """Searches FAISS index and returns relevant chunks with metadata."""

    def __init__(
        self,
        store_dir: str,
        OPENROUTER_API_KEY: str,
        embedding_model: str = "openai/text-embedding-3-small",
    ):
        self.store_dir = store_dir
        self.client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        self.embedding_model = embedding_model
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []
        self._last_modified: float = 0.0

    def load_index(self) -> bool:
        """Loads FAISS index and metadata from disk."""
        index_path = os.path.join(self.store_dir, INDEX_FILENAME)
        metadata_path = os.path.join(self.store_dir, METADATA_FILENAME)

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.warning("FAISS index not found. Run indexer first.")
            return False

        self.index = faiss.read_index(index_path)
        with open(metadata_path, encoding="utf-8") as f:
            self.metadata = json.load(f)

        self._last_modified = os.path.getmtime(index_path)
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        return True

    def _reload_if_changed(self) -> None:
        """Reloads FAISS index from disk if the index file has been modified."""
        index_path = os.path.join(self.store_dir, INDEX_FILENAME)
        try:
            current_mtime = os.path.getmtime(index_path)
        except OSError:
            return
        if current_mtime > self._last_modified:
            logger.info("FAISS index changed on disk, reloading...")
            self.load_index()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Embeds query and searches FAISS index for nearest neighbors."""
        self._reload_if_changed()

        if self.index is None or self.index.ntotal == 0:
            self.load_index()

        if self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = self._embed_query(query)
        vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(vector)

        scores, indices = self.index.search(vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                {
                    "score": float(score),
                    "metadata": self.metadata[idx],
                }
            )
        return results

    def _embed_query(self, query: str) -> list[float]:
        """Embeds query text using OpenAI."""
        response = self.client.embeddings.create(
            input=[query],
            model=self.embedding_model,
        )
        return response.data[0].embedding
