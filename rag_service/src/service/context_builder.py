import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds LLM context from retrieval results by fetching original content
    from source documents and formatting with metadata."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def build_context(self, search_results: list[dict]) -> str:
        """Takes FAISS search results and builds a formatted context string
        with source attribution for the LLM."""
        if not search_results:
            return "Bağlam bilgisi bulunamadı."

        context_parts = []
        for i, result in enumerate(search_results, 1):
            metadata = result["metadata"]
            source = metadata.get("source", "bilinmeyen")
            source_type = metadata.get("source_type", "")
            score = result.get("score", 0)

            content = self._fetch_original_content(metadata)

            header = f"[Kaynak {i}: {source}"
            if "date" in metadata:
                header += f" | Tarih: {metadata['date']}"
            header += f" | Benzerlik: {score:.3f}]"

            context_parts.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(context_parts)

    def _fetch_original_content(self, metadata: dict) -> str:
        """Fetches content for context. Uses stored chunk content as primary source,
        enriches with original file data for CSV/JSON when possible."""
        source_type = metadata.get("source_type", "")
        stored_content = metadata.get("content", "")

        path = self._resolve_file_path(metadata)

        try:
            if source_type == "csv" and path and path.exists():
                return self._fetch_csv_content(path, metadata)
            elif source_type == "json" and path and path.exists():
                return self._fetch_json_content(path, metadata)
        except Exception as e:
            logger.error(f"Error fetching content from {path}: {e}")

        if stored_content:
            return stored_content

        return self._reconstruct_from_metadata(metadata)

    def _resolve_file_path(self, metadata: dict) -> Path | None:
        """Resolves the file path from metadata, trying container path then data_dir fallback."""
        file_path = metadata.get("file_path", "")
        if file_path:
            path = Path(file_path)
            if path.exists():
                return path
        source = metadata.get("source", "")
        if source:
            path = Path(self.data_dir) / source
            if path.exists():
                return path
        return None

    def _fetch_csv_content(self, path: Path, metadata: dict) -> str:
        """Fetches the relevant row from a CSV file with full column context."""
        row_index = metadata.get("row_index")

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if row_index is not None and row_index < len(rows):
            row = rows[row_index]
            parts = []
            for key, value in row.items():
                readable_key = key.replace("_", " ").title()
                parts.append(f"  - {readable_key}: {value}")
            return "\n".join(parts)

        return str(rows)

    def _fetch_json_content(self, path: Path, metadata: dict) -> str:
        """Fetches the relevant entry from a JSON file."""
        entry_index = metadata.get("entry_index")
        data = json.loads(path.read_text(encoding="utf-8"))

        if entry_index is not None and entry_index < len(data):
            entry = data[entry_index]
            parts = []
            if "tarih" in entry:
                parts.append(f"  Tarih: {entry['tarih']}")
            if "degisiklik" in entry:
                parts.append(f"  Değişiklik: {entry['degisiklik']}")
            if "onceki_deger" in entry:
                parts.append(f"  Önceki Değer: {entry['onceki_deger']}")
            if "yeni_deger" in entry:
                parts.append(f"  Yeni Değer: {entry['yeni_deger']}")
            if "etkilenen_paket" in entry and entry["etkilenen_paket"]:
                parts.append(f"  Etkilenen Paket: {entry['etkilenen_paket']}")
            if "etkilenen_madde" in entry and entry["etkilenen_madde"]:
                parts.append(f"  Etkilenen Madde: {entry['etkilenen_madde']}")
            return "\n".join(parts)

        return json.dumps(data, ensure_ascii=False, indent=2)

    def _reconstruct_from_metadata(self, metadata: dict) -> str:
        """Fallback: reconstructs content from metadata if original file is unavailable."""
        parts = []
        if "raw_row" in metadata:
            for k, v in metadata["raw_row"].items():
                parts.append(f"  - {k}: {v}")
            return "\n".join(parts)
        return str(metadata)
