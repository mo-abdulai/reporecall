import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reporecall.models.hybrid_retrieval import RetrievalBranch
from reporecall.models.retrieval_chunks import RetrievalChunk

_REL_TOLERANCE = 1e-12
_ABS_TOLERANCE = 1e-15


class HybridSearchHit(BaseModel):
    """One final RRF-ranked chunk with explainable branch contributions."""

    chunk: RetrievalChunk
    rank: int = Field(gt=0)
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

    @field_validator("rrf_score", "dense_score", "keyword_score")
    @classmethod
    def require_finite_scores(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Hybrid search scores must be finite.")
        return value

    @field_validator("dense_rrf_contribution", "keyword_rrf_contribution")
    @classmethod
    def require_finite_contributions(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("RRF contributions must be finite.")
        return value

    @field_validator("sources")
    @classmethod
    def normalize_sources(
        cls,
        value: tuple[RetrievalBranch, ...],
    ) -> tuple[RetrievalBranch, ...]:
        if len(set(value)) != len(value):
            raise ValueError("Hybrid search retrieval branches must be unique.")
        selected = set(value)
        return tuple(branch for branch in RetrievalBranch if branch in selected)

    @model_validator(mode="after")
    def validate_branch_signals(self) -> "HybridSearchHit":
        _validate_branch_signal(
            label="Dense",
            rank=self.dense_rank,
            score=self.dense_score,
            contribution=self.dense_rrf_contribution,
            source_present=RetrievalBranch.DENSE in self.sources,
        )
        _validate_branch_signal(
            label="Keyword",
            rank=self.keyword_rank,
            score=self.keyword_score,
            contribution=self.keyword_rrf_contribution,
            source_present=RetrievalBranch.KEYWORD in self.sources,
        )
        contribution_sum = (
            self.dense_rrf_contribution + self.keyword_rrf_contribution
        )
        if not math.isclose(
            self.rrf_score,
            contribution_sum,
            rel_tol=_REL_TOLERANCE,
            abs_tol=_ABS_TOLERANCE,
        ):
            raise ValueError("RRF score must equal the sum of branch contributions.")
        return self


class RankedHybridRetrievalResult(BaseModel):
    """Final ranked hybrid hits with fusion diagnostics and query provenance."""

    query: str
    hits: tuple[HybridSearchHit, ...]
    candidate_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    rank_constant: int = Field(gt=0)
    top_k: int = Field(gt=0)

    model_config = ConfigDict(frozen=True)

    @field_validator("query")
    @classmethod
    def require_nonblank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Ranked hybrid retrieval query must not be blank.")
        return value

    @model_validator(mode="after")
    def validate_ranked_hits(self) -> "RankedHybridRetrievalResult":
        if self.returned_count != len(self.hits):
            raise ValueError("Returned count must match ranked hybrid hits.")
        if self.returned_count > self.candidate_count:
            raise ValueError("Returned count cannot exceed candidate count.")
        if self.returned_count > self.top_k:
            raise ValueError("Returned count cannot exceed fusion top-k.")
        if self.candidate_count > 0 and self.returned_count == 0:
            raise ValueError("Non-empty candidates must produce ranked hybrid hits.")

        chunk_ids = tuple(hit.chunk.chunk_id for hit in self.hits)
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Ranked hybrid hit chunk IDs must be unique.")
        expected_ranks = tuple(range(1, len(self.hits) + 1))
        if tuple(hit.rank for hit in self.hits) != expected_ranks:
            raise ValueError("Final hybrid ranks must be contiguous and one-based.")
        if self.hits != tuple(
            sorted(
                self.hits,
                key=lambda hit: (-hit.rrf_score, hit.chunk.chunk_id),
            )
        ):
            raise ValueError(
                "Ranked hybrid hits must use RRF score and chunk-ID ordering."
            )

        for hit in self.hits:
            expected_dense = _expected_contribution(
                self.rank_constant,
                hit.dense_rank,
            )
            expected_keyword = _expected_contribution(
                self.rank_constant,
                hit.keyword_rank,
            )
            if not math.isclose(
                hit.dense_rrf_contribution,
                expected_dense,
                rel_tol=_REL_TOLERANCE,
                abs_tol=_ABS_TOLERANCE,
            ):
                raise ValueError("Dense contribution does not match the RRF formula.")
            if not math.isclose(
                hit.keyword_rrf_contribution,
                expected_keyword,
                rel_tol=_REL_TOLERANCE,
                abs_tol=_ABS_TOLERANCE,
            ):
                raise ValueError("Keyword contribution does not match the RRF formula.")
        return self


def _validate_branch_signal(
    *,
    label: str,
    rank: int | None,
    score: float | None,
    contribution: float,
    source_present: bool,
) -> None:
    if (rank is None) != (score is None):
        raise ValueError(f"{label} rank and score must appear together.")
    signal_present = rank is not None and score is not None
    if signal_present != source_present:
        raise ValueError(
            f"{label} retrieval branch must agree with its rank and score."
        )
    if signal_present and contribution <= 0:
        raise ValueError(f"Present {label.lower()} branch contribution must be positive.")
    if not signal_present and contribution != 0:
        raise ValueError(f"Missing {label.lower()} branch contribution must be zero.")


def _expected_contribution(rank_constant: int, rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / (rank_constant + rank)
