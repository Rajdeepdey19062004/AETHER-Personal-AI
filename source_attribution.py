"""Build compact, de-duplicated source metadata for retrieved passages."""


def build_sources(documents: list[dict]) -> list[dict]:
    sources, seen = [], set()
    for document in documents:
        document_id = document.get("document_id", document.get("source", "Unknown"))
        chunk_id = document.get("id", document.get("chunk_index"))
        key = (document_id, chunk_id)
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "document_id": document_id,
            "chunk_id": chunk_id,
            "source": document.get("source", document.get("filename", document_id)),
            "text": document.get("text", ""),
            "score": document.get("rerank_score", document.get("rrf_score", document.get("score"))),
        })
    return sources
