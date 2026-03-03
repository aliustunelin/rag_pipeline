import csv
from pathlib import Path


class CsvParser:
    """Parses .csv files row-by-row, converting each row to natural language
    to preserve column context for semantic search."""

    @staticmethod
    def parse(file_path: str) -> list[dict]:
        path = Path(file_path)
        documents = []

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                natural_text = CsvParser._row_to_natural_language(row)
                documents.append(
                    {
                        "content": natural_text,
                        "metadata": {
                            "source": path.name,
                            "source_type": "csv",
                            "file_path": str(path),
                            "row_index": row_idx,
                            "raw_row": dict(row),
                        },
                    }
                )

        return documents

    @staticmethod
    def _row_to_natural_language(row: dict) -> str:
        """Converts a CSV row dict to a natural language sentence
        preserving all column context."""
        parts = []
        for key, value in row.items():
            readable_key = key.replace("_", " ").title()
            parts.append(f"{readable_key}: {value}")
        return " | ".join(parts)
