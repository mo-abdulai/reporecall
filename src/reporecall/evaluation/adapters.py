"""Normalize existing search hits without executing retrieval or changing ranks."""

from collections.abc import Sequence

from reporecall.models.evaluation import RankedRetrievedItem
from reporecall.models.hybrid_ranking import HybridSearchHit
from reporecall.models.keyword_retrieval import KeywordSearchHit
from reporecall.models.reranking import RerankedSearchHit
from reporecall.models.retrieval import VectorSearchHit


def validate_ranked_items(
    items: Sequence[RankedRetrievedItem],
) -> tuple[RankedRetrievedItem, ...]:
    """Require unique chunk IDs and contiguous ranks in the supplied order."""
    records = tuple(items)
    if len({item.chunk_id for item in records}) != len(records):
        raise ValueError("Retrieved chunk IDs must be unique.")
    if tuple(item.rank for item in records) != tuple(range(1, len(records) + 1)):
        raise ValueError(
            "Retrieved ranks must be contiguous and one-based in input order."
        )
    return records


def normalize_ranked_hits(
    hits: Sequence[
        VectorSearchHit | KeywordSearchHit | HybridSearchHit | RerankedSearchHit
    ],
) -> tuple[RankedRetrievedItem, ...]:
    """Adapt dense, BM25, RRF, or reranked hits using identities and ranks only.

    Malformed order is rejected, never repaired by sorting scores. Unranked
    hybrid candidates and relationship-expanded context are not valid inputs.
    """
    items = []
    for hit in hits:
        if not isinstance(
            hit, (VectorSearchHit, KeywordSearchHit, HybridSearchHit, RerankedSearchHit)
        ):
            raise TypeError("Evaluation requires ranked retrieval hits.")
        items.append(RankedRetrievedItem(chunk_id=hit.chunk.chunk_id, rank=hit.rank))
    return validate_ranked_items(items)
