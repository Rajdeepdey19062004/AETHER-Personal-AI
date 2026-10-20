"""Document upload, hashing, indexing, listing, and deletion operations."""

import hashlib
import re
from pathlib import Path

from aether_core import DOCUMENT_DIR, extract_text, split_text
from document_store import delete_document_from_index, list_documents, upsert_document
from logger_config import get_logger
from user_context import get_user_data_dir

logger = get_logger("document_manager")


def calculate_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def make_document_id(filename: str) -> str:
    """Use a stable, case-insensitive filename identity for file updates."""
    normalized = Path(filename.replace("\\", "/")).name.strip().casefold()
    if not normalized or normalized in {".", ".."}:
        raise ValueError("A valid filename is required.")
    return normalized


def get_safe_storage_path(filename: str) -> Path:
    safe_name = Path(filename.replace("\\", "/")).name.strip()
    if not safe_name or safe_name in {".", ".."}:
        raise ValueError("A valid filename is required.")
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", safe_name)
    data_dir = get_user_data_dir()
    document_dir = DOCUMENT_DIR if data_dir == DOCUMENT_DIR.parent else data_dir / "documents"
    return document_dir / safe_name


def index_uploaded_document(filename: str, content: bytes) -> dict:
    path = get_safe_storage_path(filename)
    digest = calculate_sha256(content)
    # Avoid touching the stored source when exact content is already indexed.
    known = list_documents()
    duplicate = next((doc for doc in known if doc.get("sha256") == digest), None)
    if duplicate:
        logger.info("Duplicate document detected: %s", filename)
        return {"status": "duplicate", "filename": filename,
                "document_id": duplicate.get("document_id"),
                "message": f"Duplicate content already indexed as {duplicate['filename']}."}

    text = extract_text(path.name, content)
    chunks = split_text(text)
    if not chunks:
        raise ValueError("No readable text could be extracted from this document.")

    result = upsert_document({
        "document_id": make_document_id(path.name),
        "filename": path.name,
        "sha256": digest,
        "chunks": chunks,
    })
    if result["status"] != "duplicate":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        logger.info("Document indexed: %s (%s)", path.name, result["status"])
    return result


def get_all_documents() -> list[dict]:
    return list_documents()


def remove_document(document_id: str, delete_file: bool = True) -> int:
    docs = list_documents()
    selected = next((doc for doc in docs if doc.get("document_id") == document_id), None)
    removed = delete_document_from_index(document_id)
    if delete_file and selected:
        path = get_safe_storage_path(selected["filename"])
        path.unlink(missing_ok=True)
    logger.info("Document removed from index: %s", document_id)
    return removed
