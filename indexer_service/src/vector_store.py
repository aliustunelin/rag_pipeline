import json
import os
import numpy as np
import faiss


class VectorStore:
    """FAISS-based vector store with metadata support.

    Stores embeddings in a FAISS index and metadata in a sidecar JSON file.
    """

    def __init__(self, store_dir: str, dimension: int = 1536):
        self.store_dir = store_dir
        self.dimension = dimension
        self.index_path = os.path.join(store_dir, "index.faiss")
        self.metadata_path = os.path.join(store_dir, "metadata.json")
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []

    def initialize(self):
        """Creates a new empty FAISS index."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []

    def add(self, embeddings: list[list[float]], metadata_list: list[dict]):
        """Adds embeddings and their metadata to the store."""
        if not embeddings:
            return

        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadata.extend(metadata_list)

    def save(self):
        """Persists index and metadata to disk."""
        os.makedirs(self.store_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        """Loads index and metadata from disk. Returns True if successful."""
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            return False

        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, encoding="utf-8") as f:
            self.metadata = json.load(f)
        return True

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        """Searches the index and returns top_k results with metadata."""
        if self.index is None or self.index.ntotal == 0:
            return []

        vector = np.array([query_vector], dtype=np.float32)
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

    @property
    def total_vectors(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal
