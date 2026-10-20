"""Persistent, incremental document and embedding store for AETHER."""

import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from embedding_engine import create_embeddings, get_embedding_backend
from user_context import get_user_data_dir

DATA_DIR = Path(__file__).resolve().parent / "data"
INDEX_DIR = DATA_DIR / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
METADATA_PATH = INDEX_DIR / "index_meta.json"
DOCUMENTS_PATH = INDEX_DIR / "documents.json"


def _index_paths() -> tuple[Path, Path, Path, Path, Path]:
    user_root = get_user_data_dir()
    if user_root == DATA_DIR:
        return INDEX_DIR, CHUNKS_PATH, EMBEDDINGS_PATH, METADATA_PATH, DOCUMENTS_PATH
    index_dir = user_root / "index"
    return (index_dir, index_dir / "chunks.json", index_dir / "embeddings.npy",
            index_dir / "index_meta.json", index_dir / "documents.json")


def index_files_exist() -> bool:
    _, chunks_path, embeddings_path, _, _ = _index_paths()
    return chunks_path.is_file() and embeddings_path.is_file()


def _read_json(path: Path, default):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _atomic_write(path: Path, writer) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _legacy_documents(chunks: list[dict]) -> list[dict]:
    """Build basic metadata when upgrading an older chunks-only index."""
    docs = {}
    for chunk in chunks:
        source = str(chunk.get("filename") or chunk.get("source") or "").strip()
        if not source:
            continue
        doc_id = str(chunk.get("document_id") or source.casefold())
        doc = docs.setdefault(doc_id, {
            "document_id": doc_id,
            "filename": source,
            "sha256": None,
            "chunk_count": 0,
        })
        doc["chunk_count"] += 1
    return list(docs.values())


def load_index() -> tuple[list[dict], np.ndarray, list[dict]]:
    """Load chunks, aligned vectors, and document metadata."""
    _, chunks_path, embeddings_path, _, documents_path = _index_paths()
    chunks = _read_json(chunks_path, [])
    if not isinstance(chunks, list):
        chunks = []
    documents = _read_json(documents_path, None)
    if not isinstance(documents, list):
        documents = _legacy_documents(chunks)

    try:
        embeddings = np.asarray(np.load(embeddings_path), dtype=np.float32)
    except (OSError, ValueError):
        embeddings = np.empty((0, 0), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError(
            "Document index is inconsistent; restore a backup or rebuild it from data/documents/."
        )
    return chunks, embeddings, documents


def load_documents() -> tuple[list[dict], np.ndarray]:
    """Compatibility API used by semantic_search and older callers."""
    chunks, embeddings, _ = load_index()
    return chunks, embeddings


def load_embedding_backend() -> str:
    _, _, _, metadata_path, _ = _index_paths()
    metadata = _read_json(metadata_path, {})
    backend = metadata.get("embedding_backend") if isinstance(metadata, dict) else None
    return str(backend or get_embedding_backend())


def save_index(chunks: list[dict], embeddings: np.ndarray, documents: list[dict]) -> None:
    """Persist a complete index, validating row alignment before replacing files."""
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("Number of embedding rows must match number of chunks.")
    index_dir, chunks_path, embeddings_path, metadata_path, documents_path = _index_paths()
    index_dir.mkdir(parents=True, exist_ok=True)

    def write_json(path: Path, value) -> None:
        _atomic_write(path, lambda temp: temp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        ))

    # The metadata file is committed last so it describes the newly saved index.
    write_json(chunks_path, chunks)
    write_json(documents_path, documents)
    _atomic_write(embeddings_path, lambda temp: _save_numpy(temp, vectors))
    write_json(metadata_path, {"embedding_backend": load_embedding_backend()})


def _save_numpy(path: Path, vectors: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.save(stream, vectors)


def save_documents(chunks: list[dict[str, str]]) -> None:
    """Legacy replace-index API retained for compatibility."""
    normalized = [
        {"source": str(item.get("source", "")).strip(),
         "text": str(item.get("text", "")).strip()}
        for item in chunks
        if str(item.get("source", "")).strip() and str(item.get("text", "")).strip()
    ]
    backend = load_embedding_backend()
    vectors = create_embeddings([item["text"] for item in normalized], backend=backend)
    save_index(normalized, vectors, _legacy_documents(normalized))


def upsert_document(document: dict) -> dict:
    """Add a file, skip content duplicates, or replace its prior version."""
    document_id = str(document["document_id"])
    filename = str(document["filename"])
    sha256 = str(document["sha256"])
    text_chunks = [str(text).strip() for text in document["chunks"] if str(text).strip()]
    if not text_chunks:
        raise ValueError("This document contains no extractable text.")

    chunks, embeddings, documents = load_index()
    for existing in documents:
        if existing.get("sha256") == sha256:
            return {"status": "duplicate", "filename": filename,
                    "message": f"{filename} has the same content as an indexed document."}

    # Keep the current backend so new and retained vectors remain comparable.
    backend = load_embedding_backend()
    replacing = any(doc.get("document_id") == document_id for doc in documents)
    keep = [i for i, chunk in enumerate(chunks)
            if chunk.get("document_id", chunk.get("source", "").casefold()) != document_id]
    kept_chunks = [chunks[i] for i in keep]
    kept_vectors = embeddings[keep] if len(keep) else np.empty((0, embeddings.shape[1]), dtype=np.float32)
    new_vectors = create_embeddings(text_chunks, backend=backend)
    if new_vectors.ndim != 2 or new_vectors.shape[0] != len(text_chunks):
        raise ValueError("Embedding model returned invalid data.")
    if len(kept_vectors) and kept_vectors.shape[1] != new_vectors.shape[1]:
        raise ValueError("Embedding dimensions differ; rebuild with the same embedding model.")

    new_chunks = [{"document_id": document_id, "filename": filename,
                   "source": filename, "chunk_index": i, "text": value}
                  for i, value in enumerate(text_chunks)]
    all_chunks = kept_chunks + new_chunks
    all_vectors = np.vstack((kept_vectors, new_vectors)) if len(kept_vectors) else new_vectors
    updated_docs = [doc for doc in documents if doc.get("document_id") != document_id]
    updated_docs.append({"document_id": document_id, "filename": filename,
                         "sha256": sha256, "chunk_count": len(new_chunks)})
    save_index(all_chunks, all_vectors, updated_docs)
    return {"status": "updated" if replacing else "indexed", "filename": filename,
            "chunk_count": len(new_chunks),
            "message": f"Document {'updated' if replacing else 'indexed'} successfully."}


def delete_document_from_index(document_id: str) -> int:
    chunks, embeddings, documents = load_index()
    keep = [i for i, chunk in enumerate(chunks)
            if chunk.get("document_id", chunk.get("source", "").casefold()) != document_id]
    removed = len(chunks) - len(keep)
    remaining_vectors = embeddings[keep] if keep else np.empty((0, embeddings.shape[1]), dtype=np.float32)
    remaining_docs = [doc for doc in documents if doc.get("document_id") != document_id]
    save_index([chunks[i] for i in keep], remaining_vectors, remaining_docs)
    return removed


def list_documents() -> list[dict]:
    return load_index()[2]
