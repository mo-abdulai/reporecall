"""Transparent rank-based metrics; absent judgments always have relevance zero."""

from collections.abc import Sequence
from math import fsum, log2

from reporecall.evaluation.adapters import validate_ranked_items
from reporecall.models.evaluation import (
    RankedRetrievedItem,
    RetrievalEvaluationCase,
    RetrievalMetricsAtK,
)


def dcg_at_k(relevance: Sequence[int], k: int) -> float:
    """Sum (2**grade - 1)/log2(rank + 1), using grades 0, 1, or 2."""
    if type(k) is not int or k <= 0:
        raise ValueError("K must be a positive integer.")
    if any(type(grade) is not int or not 0 <= grade <= 2 for grade in relevance):
        raise ValueError("Relevance grades must be integers from 0 to 2.")
    return fsum(
        (2**grade - 1) / log2(rank + 1) for rank, grade in enumerate(relevance[:k], 1)
    )


def metrics_at_k(
    case: RetrievalEvaluationCase,
    items: Sequence[RankedRetrievedItem],
    k: int,
) -> RetrievalMetricsAtK:
    """Measure a validated ranking; precision divides by K, including empty slots.

    Binary metrics use grade > 0. Recall divides by all relevant judgments;
    hit rate indicates any hit; reciprocal rank uses the first relevant rank
    within K. NDCG divides graded DCG by ideal descending judgment DCG.
    """
    ranked = validate_ranked_items(items)
    judgments = {j.chunk_id: j.relevance for j in case.judgments}
    grades = [judgments.get(item.chunk_id, 0) for item in ranked]
    dcg = dcg_at_k(grades, k)
    ideal = dcg_at_k(sorted(judgments.values(), reverse=True), k)
    if ideal <= 0:
        raise ValueError("Evaluation requires positive ideal DCG.")
    top = grades[:k]
    relevant = sum(grade > 0 for grade in top)
    first = next((rank for rank, grade in enumerate(top, 1) if grade > 0), None)
    return RetrievalMetricsAtK(
        k=k,
        precision=relevant / k,
        recall=relevant / sum(grade > 0 for grade in judgments.values()),
        hit_rate=float(relevant > 0),
        reciprocal_rank=0.0 if first is None else 1 / first,
        ndcg=min(
            1.0, dcg / ideal
        ),  # Bound floating-point overshoot at the ideal order.
    )
