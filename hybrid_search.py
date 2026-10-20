"""Dependency-free keyword retrieval and reciprocal-rank fusion helpers."""

from collections import defaultdict
import re


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", str(text).casefold(), flags=re.UNICODE)


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = defaultdict(float)
    documents: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            item_id = item.get("id")
            if item_id is None:
                raise ValueError("Every search result must contain an 'id'.")
            key = str(item_id)
            scores[key] += 1.0 / (k + rank)
            documents[key] = {**documents.get(key, {}), **item}
    return [
        {**documents[key], "rrf_score": score}
        for key, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    ]


def keyword_search(query: str, chunks: list[dict], top_k: int = 10) -> list[dict]:
    query_tokens = set(tokenize(query))
    if not query_tokens or top_k <= 0:
        return []
    results = []
    for index, chunk in enumerate(chunks):
        tokens = set(tokenize(chunk.get("text", "")))
        overlap = query_tokens & tokens
        if overlap:
            item = dict(chunk)
            item.setdefault("id", f"{item.get('document_id', item.get('source', 'doc'))}:{item.get('chunk_index', index)}")
            item["keyword_score"] = len(overlap) / len(query_tokens)
            results.append(item)
    return sorted(results, key=lambda item: item["keyword_score"], reverse=True)[:top_k]


def hybrid_search_from_results(keyword_results: list[dict], semantic_results: list[dict], top_k: int = 5) -> list[dict]:
    return reciprocal_rank_fusion([keyword_results, semantic_results])[:top_k]
