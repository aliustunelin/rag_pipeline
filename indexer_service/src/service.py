import logging
import os
from pathlib import Path

from .parsers import TxtParser, CsvParser, JsonParser
from .chunking import ChunkingStrategy
from .embedder import Embedder
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

PARSER_MAP = {
    ".txt": TxtParser,
    ".csv": CsvParser,
    ".json": JsonParser,
}


class IndexerService:
    """Orchestrates the full indexing pipeline: parse → chunk → embed → store."""

    def __init__(
        self,
        data_dir: str,
        store_dir: str,
        OPENROUTER_API_KEY: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedding_model: str = "openai/text-embedding-3-small",
    ):
        self.data_dir = data_dir
        self.chunker = ChunkingStrategy(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = Embedder(api_key=OPENROUTER_API_KEY, model=embedding_model)
        self.vector_store = VectorStore(store_dir=store_dir)
        self._indexing = False

    def index_all(self) -> dict:
        """Parses all supported files in data_dir, chunks, embeds, and stores them."""
        if self._indexing:
            return {"status": "already_indexing"}

        self._indexing = True
        try:
            all_documents = self._parse_all_files()

            chunked_documents = self.chunker.chunk(all_documents)
            logger.info(f"Total chunks after chunking: {len(chunked_documents)}")

            texts = [doc["content"] for doc in chunked_documents]
            metadata_list = []
            for doc in chunked_documents:
                meta = {**doc["metadata"], "content": doc["content"]}
                metadata_list.append(meta)

            logger.info("Generating embeddings...")
            embeddings = self._batch_embed(texts)

            self.vector_store.initialize()
            self.vector_store.add(embeddings, metadata_list)
            self.vector_store.save()

            result = {
                "status": "success",
                "total_files": len(self._get_supported_files()),
                "total_chunks": len(chunked_documents),
                "total_vectors": self.vector_store.total_vectors,
            }
            logger.info(f"Indexing complete: {result}")
            return result

        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            return {"status": "error", "detail": str(e)}
        finally:
            self._indexing = False

    def _parse_all_files(self) -> list[dict]:
        """Parses all supported files in the data directory."""
        documents = []
        for file_path in self._get_supported_files():
            ext = file_path.suffix.lower()
            parser = PARSER_MAP.get(ext)
            if parser:
                logger.info(f"Parsing {file_path.name} with {parser.__name__}")
                docs = parser.parse(str(file_path))
                documents.extend(docs)
        return documents

    def _get_supported_files(self) -> list[Path]:
        """Returns list of supported files in data directory."""
        data_path = Path(self.data_dir)
        files = []
        for ext in PARSER_MAP:
            files.extend(data_path.glob(f"*{ext}"))
        return sorted(files)

    def _batch_embed(self, texts: list[str], batch_size: int = 50) -> list[list[float]]:
        """Embeds texts in batches to avoid API limits."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.embedder.embed_texts(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def get_status(self) -> dict:
        """Returns current index status."""
        loaded = self.vector_store.load()
        return {
            "indexed": loaded,
            "total_vectors": self.vector_store.total_vectors if loaded else 0,
            "indexing_in_progress": self._indexing,
            "data_files": [f.name for f in self._get_supported_files()],
        }
