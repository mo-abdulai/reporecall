import math
from collections.abc import Sequence

from reporecall.models import (
    HybridSearchHit,
    RankedHybridRetrievalResult,
    RerankedRetrievalResult,
    RerankedSearchHit,
)
from reporecall.retrieval.exceptions import RerankerOutputError, RerankingError
from reporecall.retrieval.reranker_backend import (
    CrossEncoderRerankerConfig,
    RerankerBackend,
)


class CrossEncoderReranker:
    """Rerank a bounded prefix of fused hybrid hits using raw model scores."""

    def __init__(
        self,
        *,
        backend: RerankerBackend,
        config: CrossEncoderRerankerConfig | None = None,
    ) -> None:
        backend_model = backend.model_name
        if not backend_model.strip():
            raise RerankingError("Reranker backend model name must not be blank.")

        self.backend = backend
        self.config = config or CrossEncoderRerankerConfig(model_name=backend_model)
        if self.config.model_name != backend_model:
            raise RerankingError(
                "Reranker configuration model does not match the backend model."
            )

    def rerank(
        self,
        result: RankedHybridRetrievalResult,
    ) -> RerankedRetrievalResult:
        """Score the RRF prefix, order by relevance, and apply final top-k."""

        selected = result.hits[: self.config.candidate_k]
        if not selected:
            return self._result(result, reranked=0, hits=())

        passages = [hit.chunk.text for hit in selected]
        raw_scores = self.backend.score(result.query, passages)
        scores = _validated_scores(raw_scores, expected_count=len(selected))

        scored_hits = [
            self._reranked_hit(hit, reranker_score=score, rank=1)
            for hit, score in zip(selected, scores, strict=True)
        ]
        scored_hits.sort(
            key=lambda hit: (-hit.reranker_score, hit.chunk.chunk_id)
        )
        hits = tuple(
            hit.model_copy(update={"rank": rank})
            for rank, hit in enumerate(scored_hits[: self.config.top_k], start=1)
        )
        return self._result(result, reranked=len(selected), hits=hits)

    def _reranked_hit(
        self,
        hit: HybridSearchHit,
        *,
        reranker_score: float,
        rank: int,
    ) -> RerankedSearchHit:
        return RerankedSearchHit(
            chunk=hit.chunk,
            rank=rank,
            reranker_score=reranker_score,
            reranker_model=self.backend.model_name,
            previous_hybrid_rank=hit.rank,
            rrf_score=hit.rrf_score,
            dense_rank=hit.dense_rank,
            dense_score=hit.dense_score,
            dense_rrf_contribution=hit.dense_rrf_contribution,
            keyword_rank=hit.keyword_rank,
            keyword_score=hit.keyword_score,
            keyword_rrf_contribution=hit.keyword_rrf_contribution,
            sources=hit.sources,
        )

    def _result(
        self,
        result: RankedHybridRetrievalResult,
        *,
        reranked: int,
        hits: tuple[RerankedSearchHit, ...],
    ) -> RerankedRetrievalResult:
        return RerankedRetrievalResult(
            query=result.query,
            model_name=self.backend.model_name,
            input_candidate_count=len(result.hits),
            reranked_candidate_count=reranked,
            returned_count=len(hits),
            candidate_k=self.config.candidate_k,
            top_k=self.config.top_k,
            hits=hits,
        )


def _validated_scores(
    scores: Sequence[float],
    *,
    expected_count: int,
) -> list[float]:
    if len(scores) != expected_count:
        raise RerankerOutputError(
            "Reranker backend returned a different number of scores than candidates."
        )

    validated: list[float] = []
    for value in scores:
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise RerankerOutputError(
                "Reranker backend scores must be numeric."
            ) from exc
        if not math.isfinite(score):
            raise RerankerOutputError("Reranker backend scores must be finite.")
        validated.append(score)
    return validated
