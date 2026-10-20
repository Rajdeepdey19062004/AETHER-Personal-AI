from evaluation import (
    check_answer_quality,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k():
    assert recall_at_k(["doc_a", "doc_b", "doc_c"], ["doc_b", "doc_c"], k=2) == 0.5


def test_reciprocal_rank():
    assert abs(reciprocal_rank(["doc_a", "doc_b", "doc_c"], ["doc_c"]) - 1 / 3) < 1e-9


def test_reciprocal_rank_no_match():
    assert reciprocal_rank(["doc_a", "doc_b"], ["doc_c"]) == 0.0


def test_mean_reciprocal_rank():
    cases = [{"retrieved": ["doc_a", "doc_b"], "relevant": ["doc_a"]},
             {"retrieved": ["doc_a", "doc_b"], "relevant": ["doc_b"]}]
    assert abs(mean_reciprocal_rank(cases) - 0.75) < 1e-9


def test_answer_quality_passes():
    result = check_answer_quality("Evidence based answer.", [{"source": "file.pdf", "text": "Evidence."}])
    assert result["passed"] is True


def test_answer_quality_fails_without_sources():
    result = check_answer_quality("This is an answer.", [])
    assert result["passed"] is False
