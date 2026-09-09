import math
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from reporecall.models import (
    HybridCandidate,
    HybridRetrievalResult,
    KeywordSearchHit,
    MetadataFilter,
    RetrievalBranch,
    RetrievalChunk,
    VectorSearchHit,
)
from reporecall.retrieval.exceptions import HybridRetrievalError
from reporecall.retrieval.keyword_retriever import KeywordRetriever
from reporecall.retrieval.vector_retriever import VectorRetriever


class HybridRetrievalConfig(BaseModel):
    """Independent retrieval depths for hybrid candidate collection."""

    dense_k: int = Field(default=20, gt=0)
    keyword_k: int = Field(default=20, gt=0)

    model_config = ConfigDict(frozen=True)


class HybridRetriever:
    """Collect and merge dense and keyword candidates without rank fusion."""

    def __init__(
        self,
        *,
        vector_retriever: VectorRetriever,
        keyword_retriever: KeywordRetriever,
        config: HybridRetrievalConfig | None = None,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.config = config or HybridRetrievalConfig()

    def search(
        self,
        query: str,
        *,
        metadata_filter: MetadataFilter | None = None,
    ) -> HybridRetrievalResult:
        """Run both branches and return their neutral union by chunk ID."""

        if not query.strip():
            raise HybridRetrievalError("Hybrid retrieval query must not be blank.")

        dense_hits = self.vector_retriever.search(
            query,
            k=self.config.dense_k,
            metadata_filter=metadata_filter,
        )
        keyword_hits = self.keyword_retriever.search(
            query,
            k=self.config.keyword_k,
            metadata_filter=metadata_filter,
        )
        _validate_branch_hits(dense_hits, branch="dense")
        _validate_branch_hits(keyword_hits, branch="keyword")
        candidates = _merge_candidates(dense_hits, keyword_hits)

        return HybridRetrievalResult(
            query=query,
            dense_hits=tuple(dense_hits),
            keyword_hits=tuple(keyword_hits),
            candidates=candidates,
            dense_retrieved_count=len(dense_hits),
            keyword_retrieved_count=len(keyword_hits),
            unique_candidate_count=len(candidates),
        )


class _RankedHit(Protocol):
    chunk: RetrievalChunk
    rank: int
    score: float


def _validate_branch_hits(
    hits: Sequence[_RankedHit],
    *,
    branch: str,
) -> None:
    chunk_ids: set[str] = set()
    ranks: set[int] = set()
    for hit in hits:
        chunk_id = hit.chunk.chunk_id
        if chunk_id in chunk_ids:
            raise HybridRetrievalError(
                f"{branch.capitalize()} retrieval returned duplicate chunk ID "
                f"{chunk_id!r}."
            )
        if hit.rank <= 0 or hit.rank in ranks:
            raise HybridRetrievalError(
                f"{branch.capitalize()} retrieval returned invalid or duplicate ranks."
            )
        if not math.isfinite(hit.score):
            raise HybridRetrievalError(
                f"{branch.capitalize()} retrieval returned a non-finite score."
            )
        chunk_ids.add(chunk_id)
        ranks.add(hit.rank)


def _merge_candidates(
    dense_hits: Sequence[VectorSearchHit],
    keyword_hits: Sequence[KeywordSearchHit],
) -> tuple[HybridCandidate, ...]:
    dense_by_id = {hit.chunk.chunk_id: hit for hit in dense_hits}
    keyword_by_id = {hit.chunk.chunk_id: hit for hit in keyword_hits}
    candidates: list[HybridCandidate] = []

    for chunk_id in sorted(set(dense_by_id) | set(keyword_by_id)):
        dense_hit = dense_by_id.get(chunk_id)
        keyword_hit = keyword_by_id.get(chunk_id)
        if (
            dense_hit is not None
            and keyword_hit is not None
            and dense_hit.chunk != keyword_hit.chunk
        ):
            raise HybridRetrievalError(
                f"Retrieval branches returned conflicting data for chunk ID {chunk_id!r}."
            )

        if dense_hit is not None:
            chunk = dense_hit.chunk
        elif keyword_hit is not None:
            chunk = keyword_hit.chunk
        else:
            raise HybridRetrievalError(
                f"Hybrid candidate {chunk_id!r} has no retrieval source."
            )
        sources = tuple(
            branch
            for branch, hit in (
                (RetrievalBranch.DENSE, dense_hit),
                (RetrievalBranch.KEYWORD, keyword_hit),
            )
            if hit is not None
        )
        candidates.append(
            HybridCandidate(
                chunk=chunk,
                dense_rank=None if dense_hit is None else dense_hit.rank,
                dense_score=None if dense_hit is None else dense_hit.score,
                keyword_rank=None if keyword_hit is None else keyword_hit.rank,
                keyword_score=None if keyword_hit is None else keyword_hit.score,
                sources=sources,
            )
        )
    return tuple(candidates)
