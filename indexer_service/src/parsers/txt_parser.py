from pathlib import Path


class TxtParser:
    """Parses .txt files and returns raw text with metadata."""

    @staticmethod
    def parse(file_path: str) -> list[dict]:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")

        return [
            {
                "content": text,
                "metadata": {
                    "source": path.name,
                    "source_type": "txt",
                    "file_path": str(path),
                },
            }
        ]
