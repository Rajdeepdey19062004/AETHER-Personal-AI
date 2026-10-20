"""Find document passages semantically related to a query."""

import re

import numpy as np

from document_store import load_documents, load_embedding_backend
from embedding_engine import create_embeddings
from hybrid_search import hybrid_search_from_results, keyword_search


OVERVIEW_PATTERNS = (
    r"\b(what is|what does|what's|tell me about|explain|explian|explin|summari[sz]e|describe)\b.*\b(this|it|file|document|paper|pdf|docx)\b",
    r"\b(this|it|file|document|paper|pdf|docx)\b.*\b(about|meaning|summary|explain)\b",
    r"(এটা কী|এটি কী|এই ফাইল|এই ডকুমেন্ট|এই পেপার|সারাংশ|ব্যাখ্যা কর)",
)


def search_documents(
    query: str, top_k: int = 3, document_ids: set[str] | list[str] | None = None
) -> list[dict[str, str | float]]:
    """Return relevant passages, optionally limited to selected chat attachments."""
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if not query.strip() or top_k == 0:
        return []

    chunks, embeddings = load_documents()
    if not chunks:
        return []

    allowed_ids = set(document_ids) if document_ids is not None else None
    eligible = [
        index for index, chunk in enumerate(chunks)
        if allowed_ids is None or chunk.get("document_id") in allowed_ids
    ]
    if not eligible:
        return []

    # Vague document pointers usually mean "explain the attached file". A
    # similarity search for words like "what is this" is weak, so lead with
    # each attached document's opening passages, which usually contain its title
    # and abstract/introduction.
    overview_request = any(re.search(pattern, query, re.IGNORECASE) for pattern in OVERVIEW_PATTERNS)
    if overview_request:
        by_document = {}
        for index in eligible:
            doc_id = str(chunks[index].get("document_id", ""))
            by_document.setdefault(doc_id, []).append(index)
        ranked_indices = []
        for doc_id in sorted(by_document):
            ordered = sorted(
                by_document[doc_id],
                key=lambda index: int(chunks[index].get("chunk_index", index)),
            )
            count = len(ordered)
            sample_positions = sorted({
                *range(min(3, count)),
                max(0, count // 2 - 1), count // 2,
                max(0, count - 3), max(0, count - 2), count - 1,
            })
            ranked_indices.extend(ordered[position] for position in sample_positions)
        ranked_indices = ranked_indices[:16]
    else:
        query_embedding = create_embeddings(
            [query], backend=load_embedding_backend()
        )[0]
        scores = embeddings @ query_embedding
        ranked_indices = sorted(eligible, key=lambda index: scores[index], reverse=True)[:top_k]

        # Fuse fast lexical matches with the existing semantic ranking. Stable
        # document/chunk IDs ensure the same passage is merged across both lists.
        candidates = []
        for index in eligible:
            chunk = chunks[index]
            candidates.append({**chunk, "id": f"{chunk.get('document_id', chunk.get('source', 'doc'))}:{chunk.get('chunk_index', index)}",
                               "_index": index, "score": float(scores[index])})
        semantic_ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)[:max(top_k * 3, top_k)]
        lexical_ranked = keyword_search(query, candidates, top_k=max(top_k * 3, top_k))
        fused = hybrid_search_from_results(lexical_ranked, semantic_ranked, top_k=top_k)
        ranked_indices = [item["_index"] for item in fused]

    if overview_request:
        query_embedding = create_embeddings([query], backend=load_embedding_backend())[0]
        scores = embeddings @ query_embedding

    return [
        {
            "document_id": chunks[int(index)].get("document_id"),
            "chunk_index": chunks[int(index)].get("chunk_index", int(index)),
            "id": f"{chunks[int(index)].get('document_id', chunks[int(index)].get('source', 'doc'))}:{chunks[int(index)].get('chunk_index', int(index))}",
            "source": chunks[int(index)].get("source", chunks[int(index)].get("filename", "Unknown")),
            "text": chunks[int(index)]["text"],
            "score": float(scores[int(index)]),
        }
        for index in ranked_indices
    ]


if __name__ == "__main__":
    question = input("Enter your question: ")
    results = search_documents(question)

    if not results:
        print("No indexed documents found.")

    for item in results:
        print("\nSource:", item["source"])
        print("Similarity:", round(item["score"], 3))
        print("Text:", item["text"])
