import json

import numpy as np

from document_store import save_index


def test_save_index_creates_files(tmp_path, monkeypatch):
    import document_store

    monkeypatch.setattr(document_store, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(document_store, "CHUNKS_PATH", tmp_path / "chunks.json")
    monkeypatch.setattr(document_store, "EMBEDDINGS_PATH", tmp_path / "embeddings.npy")
    monkeypatch.setattr(document_store, "DOCUMENTS_PATH", tmp_path / "documents.json")
    monkeypatch.setattr(document_store, "METADATA_PATH", tmp_path / "index_meta.json")
    monkeypatch.setattr(document_store, "load_embedding_backend", lambda: "local-hash-v1")
    chunks = [{"document_id": "python.pdf", "filename": "python.pdf", "text": "Python."}]
    embeddings = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    documents = [{"document_id": "python.pdf", "filename": "python.pdf", "sha256": "testhash", "chunk_count": 1}]

    save_index(chunks, embeddings, documents)

    assert (tmp_path / "chunks.json").exists()
    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "documents.json").exists()
    assert len(json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))) == 1
