import json
from pathlib import Path


class JsonParser:
    """Parses .json update log files. Each entry becomes a separate document
    with date metadata for temporal conflict resolution."""

    @staticmethod
    def parse(file_path: str) -> list[dict]:
        path = Path(file_path)
        raw = json.loads(path.read_text(encoding="utf-8"))

        documents = []
        for idx, entry in enumerate(raw):
            natural_text = JsonParser._entry_to_natural_language(entry)
            metadata = {
                "source": path.name,
                "source_type": "json",
                "file_path": str(path),
                "entry_index": idx,
            }
            if "tarih" in entry:
                metadata["date"] = entry["tarih"]
            if "etkilenen_paket" in entry and entry["etkilenen_paket"]:
                metadata["affected_package"] = entry["etkilenen_paket"]
            if "etkilenen_madde" in entry and entry["etkilenen_madde"]:
                metadata["affected_clause"] = entry["etkilenen_madde"]

            documents.append({"content": natural_text, "metadata": metadata})

        return documents

    @staticmethod
    def _entry_to_natural_language(entry: dict) -> str:
        """Converts a JSON log entry to natural language with date context."""
        parts = []

        if "tarih" in entry:
            parts.append(f"{entry['tarih']} tarihli güncelleme:")

        if "degisiklik" in entry:
            parts.append(entry["degisiklik"])

        if "onceki_deger" in entry and "yeni_deger" in entry:
            parts.append(
                f"(Önceki değer: {entry['onceki_deger']}, Yeni değer: {entry['yeni_deger']})"
            )

        return " ".join(parts)
