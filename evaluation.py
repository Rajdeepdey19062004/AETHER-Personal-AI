"""Retrieval metrics and basic answer structure checks."""

from logger_config import get_logger

logger = get_logger("evaluation")


def recall_at_k(retrieved_ids, relevant_ids, k=5):
    relevant = set(relevant_ids)
    if not relevant or k <= 0:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)
    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(result_sets):
    if not result_sets:
        return 0.0
    return sum(reciprocal_rank(item["retrieved"], item["relevant"])
               for item in result_sets) / len(result_sets)


def evaluate_search_results(retrieved_ids, relevant_ids, k=5):
    result = {
        "recall_at_k": round(recall_at_k(retrieved_ids, relevant_ids, k), 4),
        "reciprocal_rank": round(reciprocal_rank(retrieved_ids, relevant_ids), 4),
        "k": k,
    }
    logger.info("Search evaluation completed: %s", result)
    return result


def check_answer_quality(answer, sources):
    """Check answer/source structure; this does not establish factual accuracy."""
    problems = []
    if not isinstance(answer, str):
        problems.append("Answer must be a string.")
        answer = ""
    if not answer.strip():
        problems.append("Answer is empty.")
    if not sources:
        problems.append("No source passages were provided.")
    else:
        for source in sources:
            if not isinstance(source, dict) or not source.get("source", source.get("filename")):
                problems.append("A source is missing its filename.")
            if not isinstance(source, dict) or not source.get("text"):
                problems.append("A source is missing its text.")
    result = {"passed": not problems, "problems": problems}
    (logger.info if result["passed"] else logger.warning)(
        "Answer quality check %s: %s", "passed" if result["passed"] else "failed", problems
    )
    return result


def evaluate_aether_queries(test_cases, search_function, k=5):
    results = []
    for case in test_cases:
        relevant_ids = set(case["relevant_ids"])
        search_results = search_function(case["query"], top_k=k)
        retrieved_ids = [item.get("document_id") for item in search_results]
        metrics = evaluate_search_results(retrieved_ids, relevant_ids, k)
        results.append({"query": case["query"], "retrieved_ids": retrieved_ids,
                        "relevant_ids": sorted(relevant_ids), **metrics})
    count = len(results)
    summary = {
        "total_queries": count,
        "mean_recall_at_k": round(sum(item["recall_at_k"] for item in results) / count, 4) if count else 0.0,
        "mean_reciprocal_rank": round(sum(item["reciprocal_rank"] for item in results) / count, 4) if count else 0.0,
        "results": results,
    }
    logger.info("AETHER evaluation summary: %s", {key: summary[key] for key in (
        "total_queries", "mean_recall_at_k", "mean_reciprocal_rank")})
    return summary
