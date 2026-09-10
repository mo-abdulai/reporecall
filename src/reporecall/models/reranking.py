import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reporecall.models.hybrid_ranking import HybridSearchHit
from reporecall.models.hybrid_retrieval import RetrievalBranch
from reporecall.models.retrieval_chunks import RetrievalChunk


class RerankedSearchHit(BaseModel):
    """One cross-encoder-ranked hit with its full first-stage provenance."""

    chunk: RetrievalChunk
    rank: int = Field(gt=0)
    reranker_score: float = Field(allow_inf_nan=False)
    reranker_model: str

    previous_hybrid_rank: int = Field(gt=0)
    rrf_score: float = Field(gt=0, allow_inf_nan=False)

    dense_rank: int | None = Field(default=None, gt=0)
    dense_score: float | None = Field(default=None, allow_inf_nan=False)
    dense_rrf_contribution: float = Field(default=0.0, ge=0, allow_inf_nan=False)

    keyword_rank: int | None = Field(default=None, gt=0)
    keyword_score: float | None = Field(default=None, allow_inf_nan=False)
    keyword_rrf_contribution: float = Field(
        default=0.0,
        ge=0,
        allow_inf_nan=False,
    )

    sources: tuple[RetrievalBranch, ...] = Field(min_length=1, max_length=2)

    model_config = ConfigDict(frozen=True)

    @field_validator("reranker_score")
    @classmethod
    def require_finite_reranker_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Reranker score must be finite.")
        return value

    @field_validator("reranker_model")
    @classmethod
    def require_nonblank_reranker_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Reranker model name must not be blank.")
        return value

    @field_validator("sources")
    @classmethod
    def normalize_sources(
        cls,
        value: tuple[RetrievalBranch, ...],
    ) -> tuple[RetrievalBranch, ...]:
        if len(set(value)) != len(value):
            raise ValueError("Reranked retrieval branches must be unique.")
        selected = set(value)
        return tuple(branch for branch in RetrievalBranch if branch in selected)

    @model_validator(mode="after")
    def validate_hybrid_provenance(self) -> "RerankedSearchHit":
        HybridSearchHit(
            chunk=self.chunk,
            rank=self.previous_hybrid_rank,
            rrf_score=self.rrf_score,
            dense_rank=self.dense_rank,
            dense_score=self.dense_score,
            dense_rrf_contribution=self.dense_rrf_contribution,
            keyword_rank=self.keyword_rank,
            keyword_score=self.keyword_score,
            keyword_rrf_contribution=self.keyword_rrf_contribution,
            sources=self.sources,
        )
        return self


class RerankedRetrievalResult(BaseModel):
    """Cross-encoder output with explicit candidate and return counts."""

    query: str
    model_name: str
    input_candidate_count: int = Field(ge=0)
    reranked_candidate_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    candidate_k: int = Field(gt=0)
    top_k: int = Field(gt=0)
    hits: tuple[RerankedSearchHit, ...]

    model_config = ConfigDict(frozen=True)

    @field_validator("query")
    @classmethod
    def require_nonblank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Reranked retrieval query must not be blank.")
        return value

    @field_validator("model_name")
    @classmethod
    def require_nonblank_model_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Reranked retrieval model name must not be blank.")
        return value

    @model_validator(mode="after")
    def validate_reranked_hits(self) -> "RerankedRetrievalResult":
        if self.top_k > self.candidate_k:
            raise ValueError("Reranker top-k cannot exceed candidate-k.")
        expected_reranked = min(self.input_candidate_count, self.candidate_k)
        if self.reranked_candidate_count != expected_reranked:
            raise ValueError(
                "Reranked candidate count must match the bounded input count."
            )
        expected_returned = min(self.reranked_candidate_count, self.top_k)
        if self.returned_count != expected_returned:
            raise ValueError("Returned count must match the reranker top-k result.")
        if self.returned_count != len(self.hits):
            raise ValueError("Returned count must match reranked hits.")

        chunk_ids = tuple(hit.chunk.chunk_id for hit in self.hits)
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Reranked hit chunk IDs must be unique.")
        expected_ranks = tuple(range(1, len(self.hits) + 1))
        if tuple(hit.rank for hit in self.hits) != expected_ranks:
            raise ValueError("Reranker ranks must be contiguous and one-based.")
        if self.hits != tuple(
            sorted(
                self.hits,
                key=lambda hit: (-hit.reranker_score, hit.chunk.chunk_id),
            )
        ):
            raise ValueError(
                "Reranked hits must use reranker score and chunk-ID ordering."
            )
        if any(hit.reranker_model != self.model_name for hit in self.hits):
            raise ValueError("Reranked hit models must match the result model.")
        return self
