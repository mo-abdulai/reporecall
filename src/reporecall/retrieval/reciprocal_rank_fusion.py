from pydantic import BaseModel, ConfigDict, Field

from reporecall.models import (
    HybridCandidate,
    HybridRetrievalResult,
    HybridSearchHit,
    RankedHybridRetrievalResult,
)


class ReciprocalRankFusionConfig(BaseModel):
    """Standard symmetric RRF configuration without branch weighting."""

    rank_constant: int = Field(default=60, gt=0)
    top_k: int = Field(default=10, gt=0)

    model_config = ConfigDict(frozen=True)


class ReciprocalRankFusion:
    """Rank Phase 13 hybrid candidates using their branch ranks only."""

    def __init__(
        self,
        config: ReciprocalRankFusionConfig | None = None,
    ) -> None:
        self.config = config or ReciprocalRankFusionConfig()

    def fuse(
        self,
        result: HybridRetrievalResult,
    ) -> RankedHybridRetrievalResult:
        """Score every candidate, sort deterministically, then apply final top-k."""

        scored = [self._score(candidate) for candidate in result.candidates]
        scored.sort(key=lambda item: (-item[3], item[0].chunk.chunk_id))
        hits = tuple(
            HybridSearchHit(
                chunk=candidate.chunk,
                rank=rank,
                rrf_score=rrf_score,
                dense_rank=candidate.dense_rank,
                dense_score=candidate.dense_score,
                dense_rrf_contribution=dense_contribution,
                keyword_rank=candidate.keyword_rank,
                keyword_score=candidate.keyword_score,
                keyword_rrf_contribution=keyword_contribution,
                sources=candidate.sources,
            )
            for rank, (
                candidate,
                dense_contribution,
                keyword_contribution,
                rrf_score,
            ) in enumerate(scored[: self.config.top_k], start=1)
        )
        return RankedHybridRetrievalResult(
            query=result.query,
            hits=hits,
            candidate_count=len(result.candidates),
            returned_count=len(hits),
            rank_constant=self.config.rank_constant,
            top_k=self.config.top_k,
        )

    def _score(
        self,
        candidate: HybridCandidate,
    ) -> tuple[HybridCandidate, float, float, float]:
        dense_contribution = self._contribution(candidate.dense_rank)
        keyword_contribution = self._contribution(candidate.keyword_rank)
        return (
            candidate,
            dense_contribution,
            keyword_contribution,
            dense_contribution + keyword_contribution,
        )

    def _contribution(self, rank: int | None) -> float:
        if rank is None:
            return 0.0
        return 1.0 / (self.config.rank_constant + rank)
