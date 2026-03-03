from langchain.text_splitter import RecursiveCharacterTextSplitter


class ChunkingStrategy:
    """Applies file-type-specific chunking strategies.

    - TXT: Recursive Character Splitting with overlap (for long legal text)
    - CSV: No chunking needed (each row is already a document)
    - JSON: No chunking needed (each log entry is already a document)
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, documents: list[dict]) -> list[dict]:
        """Takes parsed documents and applies appropriate chunking."""
        chunked = []

        for doc in documents:
            source_type = doc["metadata"].get("source_type", "")

            if source_type == "txt":
                chunked.extend(self._chunk_text(doc))
            else:
                chunked.append(doc)

        return chunked

    def _chunk_text(self, doc: dict) -> list[dict]:
        """Splits long text documents using recursive character splitting."""
        texts = self.text_splitter.split_text(doc["content"])
        chunks = []
        for i, text in enumerate(texts):
            chunk_metadata = {**doc["metadata"], "chunk_index": i}
            chunks.append({"content": text, "metadata": chunk_metadata})
        return chunks
