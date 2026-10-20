"""Incrementally index supported files stored in data/documents/."""

from pathlib import Path

from aether_core import DOCUMENT_DIR
from document_manager import index_uploaded_document

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def main() -> None:
    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path for path in DOCUMENT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        print("Put a PDF, TXT, MD, or DOCX file in data/documents/")
        return

    for path in files:
        try:
            result = index_uploaded_document(path.name, path.read_bytes())
            print(f"{path.name}: {result['message']}")
        except Exception as error:
            print(f"Failed to index {path.name}: {error}")


if __name__ == "__main__":
    main()
