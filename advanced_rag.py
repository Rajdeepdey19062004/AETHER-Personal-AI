"""Fast hybrid retrieval adapter for AETHER's existing semantic index."""

from context_filter import filter_context
from hybrid_search import hybrid_search_from_results, keyword_search
from source_attribution import build_sources


def advanced_retrieve(query: str, chunks: list[dict], semantic_search_fn,
                      reranker=None, candidate_k: int = 10, final_k: int = 5) -> dict:
    if not query.strip():
        return {"documents": [], "sources": []}
    keyword_results = keyword_search(query, chunks, candidate_k)
    semantic_results = semantic_search_fn(query, candidate_k)
    # Normalize stable IDs across both retrieval paths using document and chunk identity.
    for result in keyword_results + semantic_results:
        result.setdefault("id", f"{result.get('document_id', result.get('source', 'doc'))}:{result.get('chunk_index', result.get('text', '')[:80])}")
    fused = hybrid_search_from_results(keyword_results, semantic_results, candidate_k)
    if reranker is not None and fused:
        fused = reranker.rerank(query, fused, top_k=final_k)
    documents = filter_context(fused[:final_k], max_chunks=final_k, max_chars=6000)
    return {"documents": documents, "sources": build_sources(documents)}
