import math
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reporecall.models.keyword_retrieval import KeywordSearchHit
from reporecall.models.retrieval import VectorSearchHit
from reporecall.models.retrieval_chunks import RetrievalChunk


class RetrievalBranch(str, Enum):
    """Independent retrieval signals that discovered a hybrid candidate."""

    DENSE = "dense"
    KEYWORD = "keyword"


class HybridCandidate(BaseModel):
    """One neutral candidate with unmodified branch-specific retrieval signals."""

    chunk: RetrievalChunk
    dense_rank: int | None = Field(default=None, gt=0)
    dense_score: float | None = Field(default=None, allow_inf_nan=False)
    keyword_rank: int | None = Field(default=None, gt=0)
    keyword_score: float | None = Field(default=None, allow_inf_nan=False)
    sources: tuple[RetrievalBranch, ...] = Field(min_length=1, max_length=2)

    model_config = ConfigDict(frozen=True)

    @field_validator("dense_score", "keyword_score")
    @classmethod
    def require_finite_scores(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Hybrid candidate branch scores must be finite.")
        return value

    @field_validator("sources")
    @classmethod
    def normalize_sources(
        cls,
        value: tuple[RetrievalBranch, ...],
    ) -> tuple[RetrievalBranch, ...]:
        if len(set(value)) != len(value):
            raise ValueError("Hybrid candidate retrieval sources must be unique.")
        selected = set(value)
        return tuple(branch for branch in RetrievalBranch if branch in selected)

    @model_validator(mode="after")
    def validate_branch_signals(self) -> "HybridCandidate":
        dense_present = self.dense_rank is not None and self.dense_score is not None
        keyword_present = (
            self.keyword_rank is not None and self.keyword_score is not None
        )
        if (self.dense_rank is None) != (self.dense_score is None):
            raise ValueError("Dense rank and score must appear together.")
        if (self.keyword_rank is None) != (self.keyword_score is None):
            raise ValueError("Keyword rank and score must appear together.")

        expected_sources = tuple(
            branch
            for branch, present in (
                (RetrievalBranch.DENSE, dense_present),
                (RetrievalBranch.KEYWORD, keyword_present),
            )
            if present
        )
        if not expected_sources:
            raise ValueError("Hybrid candidate must contain a retrieval signal.")
        if self.sources != expected_sources:
            raise ValueError(
                "Hybrid candidate sources must agree with its branch signals."
            )
        return self


class HybridRetrievalResult(BaseModel):
    """Original branch rankings plus their neutral union by chunk identity."""

    query: str
    dense_hits: tuple[VectorSearchHit, ...]
    keyword_hits: tuple[KeywordSearchHit, ...]
    candidates: tuple[HybridCandidate, ...]
    dense_retrieved_count: int = Field(ge=0)
    keyword_retrieved_count: int = Field(ge=0)
    unique_candidate_count: int = Field(ge=0)

    model_config = ConfigDict(frozen=True)

    @field_validator("query")
    @classmethod
    def require_nonblank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Hybrid retrieval query must not be blank.")
        return value

    @model_validator(mode="after")
    def validate_collections(self) -> "HybridRetrievalResult":
        if self.dense_retrieved_count != len(self.dense_hits):
            raise ValueError("Dense retrieved count must match dense hits.")
        if self.keyword_retrieved_count != len(self.keyword_hits):
            raise ValueError("Keyword retrieved count must match keyword hits.")
        if self.unique_candidate_count != len(self.candidates):
            raise ValueError("Unique candidate count must match candidates.")

        dense_by_id = _hits_by_chunk_id(self.dense_hits, branch="dense")
        keyword_by_id = _hits_by_chunk_id(self.keyword_hits, branch="keyword")
        candidate_ids = tuple(candidate.chunk.chunk_id for candidate in self.candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Hybrid candidate chunk IDs must be unique.")
        if candidate_ids != tuple(sorted(candidate_ids)):
            raise ValueError("Hybrid candidates must use neutral chunk-ID ordering.")

        expected_ids = set(dense_by_id) | set(keyword_by_id)
        if set(candidate_ids) != expected_ids:
            raise ValueError("Hybrid candidates must equal the union of branch hits.")

        for candidate in self.candidates:
            chunk_id = candidate.chunk.chunk_id
            dense_hit = dense_by_id.get(chunk_id)
            keyword_hit = keyword_by_id.get(chunk_id)
            if dense_hit is not None and (
                candidate.chunk != dense_hit.chunk
                or candidate.dense_rank != dense_hit.rank
                or candidate.dense_score != dense_hit.score
            ):
                raise ValueError("Hybrid candidate does not preserve its dense hit.")
            if dense_hit is None and candidate.dense_rank is not None:
                raise ValueError("Hybrid candidate has no corresponding dense hit.")
            if keyword_hit is not None and (
                candidate.chunk != keyword_hit.chunk
                or candidate.keyword_rank != keyword_hit.rank
                or candidate.keyword_score != keyword_hit.score
            ):
                raise ValueError("Hybrid candidate does not preserve its keyword hit.")
            if keyword_hit is None and candidate.keyword_rank is not None:
                raise ValueError("Hybrid candidate has no corresponding keyword hit.")
        return self


def _hits_by_chunk_id(
    hits: tuple[VectorSearchHit, ...] | tuple[KeywordSearchHit, ...],
    *,
    branch: str,
) -> dict[str, VectorSearchHit | KeywordSearchHit]:
    by_id: dict[str, VectorSearchHit | KeywordSearchHit] = {}
    ranks: set[int] = set()
    for hit in hits:
        chunk_id = hit.chunk.chunk_id
        if chunk_id in by_id:
            raise ValueError(f"{branch.capitalize()} hits contain a duplicate chunk ID.")
        if hit.rank in ranks:
            raise ValueError(f"{branch.capitalize()} hits contain a duplicate rank.")
        by_id[chunk_id] = hit
        ranks.add(hit.rank)
    return by_id
