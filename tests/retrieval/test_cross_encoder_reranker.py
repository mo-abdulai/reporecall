import inspect
from collections.abc import Sequence

import pytest

import reporecall.retrieval.cross_encoder_reranker as reranker_module
from reporecall.github import GitHubRepository
from reporecall.models import (
    EventMetadata,
    HybridSearchHit,
    RankedHybridRetrievalResult,
    RetrievalBranch,
    RetrievalChunk,
    RetrievalSectionType,
)
from reporecall.retrieval import (
    CrossEncoderReranker,
    CrossEncoderRerankerConfig,
    RerankerModelError,
    RerankerOutputError,
    RerankingError,
)


class FakeRerankerBackend:
    def __init__(
        self,
        scores: dict[str, float] | None = None,
        *,
        model_name: str = "fake/reranker",
        output: Sequence[float] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._model_name = model_name
        self.scores = scores or {}
        self.output = output
        self.error = error
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        self.calls.append((query, tuple(passages)))
        if self.error is not None:
            raise self.error
        if self.output is not None:
            return self.output
        return [self.scores[passage] for passage in passages]


def test_reranker_score_alone_controls_final_order_and_preserves_previous_rank():
    ranked = _ranked(["a", "b", "c"])
    backend = _backend_for(ranked, [0.2, 7.0, 3.0])

    result = CrossEncoderReranker(backend=backend).rerank(ranked)

    assert [hit.chunk.chunk_id for hit in result.hits] == ["b", "c", "a"]
    assert [hit.rank for hit in result.hits] == [1, 2, 3]
    assert [hit.reranker_score for hit in result.hits] == [7.0, 3.0, 0.2]
    assert [hit.previous_hybrid_rank for hit in result.hits] == [2, 3, 1]


def test_reranker_preserves_exact_query_and_chunk_text_in_rrf_order():
    query = "ConnectionResetError after retry_worker() crashes"
    texts = [
        "src/database/session.py\nretry_worker()\nConnectionResetError\n@@ -10,4 +10,8 @@",
        "unrelated exact text",
    ]
    ranked = _ranked(["a", "b"], query=query, texts=texts)
    backend = _backend_for(ranked, [2.0, 1.0])

    CrossEncoderReranker(backend=backend).rerank(ranked)

    assert backend.calls == [(query, tuple(texts))]


def test_equal_scores_use_chunk_id_not_previous_rank_as_tiebreaker():
    ranked = _ranked(["z", "a"], dense_ranks=[1, 2])
    backend = _backend_for(ranked, [5.0, 5.0])

    result = CrossEncoderReranker(backend=backend).rerank(ranked)

    assert [hit.chunk.chunk_id for hit in result.hits] == ["a", "z"]
    assert [hit.previous_hybrid_rank for hit in result.hits] == [2, 1]


def test_raw_retrieval_scores_do_not_change_reranker_order():
    first = _ranked(["a", "b"], dense_scores=[0.99, 0.01])
    second = _ranked(["a", "b"], dense_scores=[-50.0, 900.0])
    first_backend = _backend_for(first, [1.0, 8.0])
    second_backend = _backend_for(second, [1.0, 8.0])

    first_result = CrossEncoderReranker(backend=first_backend).rerank(first)
    second_result = CrossEncoderReranker(backend=second_backend).rerank(second)

    assert [hit.chunk.chunk_id for hit in first_result.hits] == ["b", "a"]
    assert [hit.chunk.chunk_id for hit in second_result.hits] == ["b", "a"]


def test_rrf_scores_do_not_change_reranker_order():
    first = _ranked(["a", "b"])
    second = _ranked(
        ["a", "b"],
        keyword_ranks=[5, None],
        keyword_scores=[3.5, None],
    )
    first_backend = _backend_for(first, [1.0, 8.0])
    second_backend = _backend_for(second, [1.0, 8.0])

    first_result = CrossEncoderReranker(backend=first_backend).rerank(first)
    second_result = CrossEncoderReranker(backend=second_backend).rerank(second)

    assert first.hits[0].rrf_score != second.hits[0].rrf_score
    assert [hit.chunk.chunk_id for hit in first_result.hits] == ["b", "a"]
    assert [hit.chunk.chunk_id for hit in second_result.hits] == ["b", "a"]


def test_candidate_limit_scores_exactly_the_first_rrf_hits():
    ranked = _ranked([f"chunk-{index:02d}" for index in range(1, 31)])
    backend = _backend_for(ranked, [float(index) for index in range(30)])
    reranker = CrossEncoderReranker(
        backend=backend,
        config=CrossEncoderRerankerConfig(
            model_name=backend.model_name,
            candidate_k=20,
            top_k=5,
        ),
    )

    result = reranker.rerank(ranked)

    assert backend.calls[0][1] == tuple(hit.chunk.text for hit in ranked.hits[:20])
    assert len(backend.calls[0][1]) == 20
    assert result.input_candidate_count == 30
    assert result.reranked_candidate_count == 20
    assert result.returned_count == 5
    assert [hit.chunk.chunk_id for hit in result.hits] == [
        "chunk-20",
        "chunk-19",
        "chunk-18",
        "chunk-17",
        "chunk-16",
    ]


def test_candidate_k_larger_than_available_scores_every_available_hit():
    ranked = _ranked(["a", "b", "c"])
    backend = _backend_for(ranked, [1.0, 2.0, 3.0])

    result = CrossEncoderReranker(backend=backend).rerank(ranked)

    assert len(backend.calls[0][1]) == 3
    assert result.reranked_candidate_count == 3
    assert result.returned_count == 3


def test_empty_input_returns_empty_result_without_calling_backend():
    ranked = _ranked([])
    backend = FakeRerankerBackend()

    result = CrossEncoderReranker(backend=backend).rerank(ranked)

    assert backend.calls == []
    assert result.query == ranked.query
    assert result.model_name == backend.model_name
    assert result.input_candidate_count == 0
    assert result.reranked_candidate_count == 0
    assert result.returned_count == 0
    assert result.hits == ()


@pytest.mark.parametrize("output", [[1.0], [1.0, 2.0, 3.0]])
def test_reranker_rejects_backend_score_count_mismatch(output: list[float]):
    ranked = _ranked(["a", "b"])
    backend = FakeRerankerBackend(output=output)

    with pytest.raises(RerankerOutputError, match="different number"):
        CrossEncoderReranker(backend=backend).rerank(ranked)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_reranker_rejects_nonfinite_backend_scores(score: float):
    ranked = _ranked(["a"])
    backend = FakeRerankerBackend(output=[score])

    with pytest.raises(RerankerOutputError, match="finite"):
        CrossEncoderReranker(backend=backend).rerank(ranked)


def test_reranker_accepts_negative_finite_scores():
    ranked = _ranked(["a", "b"])
    backend = _backend_for(ranked, [-5.0, -1.0])

    result = CrossEncoderReranker(backend=backend).rerank(ranked)

    assert [hit.chunk.chunk_id for hit in result.hits] == ["b", "a"]


def test_backend_failure_is_not_silently_replaced_with_rrf_results():
    ranked = _ranked(["a"])
    backend = FakeRerankerBackend(error=RerankerModelError("model unavailable"))

    with pytest.raises(RerankerModelError, match="model unavailable"):
        CrossEncoderReranker(backend=backend).rerank(ranked)


def test_explicit_configuration_must_match_backend_model():
    backend = FakeRerankerBackend(model_name="actual/model")

    with pytest.raises(RerankingError, match="does not match"):
        CrossEncoderReranker(
            backend=backend,
            config=CrossEncoderRerankerConfig(model_name="configured/model"),
        )


def test_backend_model_is_the_default_config_and_result_source_of_truth():
    ranked = _ranked(["a"])
    backend = _backend_for(ranked, [1.0], model_name="fake/actual-model")
    reranker = CrossEncoderReranker(backend=backend)

    result = reranker.rerank(ranked)

    assert reranker.config.model_name == "fake/actual-model"
    assert result.model_name == "fake/actual-model"
    assert result.hits[0].reranker_model == "fake/actual-model"


def test_reranking_preserves_complete_chunk_and_branch_provenance():
    ranked = _ranked(
        ["a"],
        dense_ranks=[2],
        dense_scores=[0.8123],
        keyword_ranks=[3],
        keyword_scores=[7.456],
    )
    source = ranked.hits[0]
    result = CrossEncoderReranker(
        backend=_backend_for(ranked, [9.0])
    ).rerank(ranked)
    hit = result.hits[0]

    assert hit.chunk is source.chunk
    assert hit.previous_hybrid_rank == source.rank
    assert hit.rrf_score == source.rrf_score
    assert hit.dense_rank == source.dense_rank
    assert hit.dense_score == source.dense_score
    assert hit.dense_rrf_contribution == source.dense_rrf_contribution
    assert hit.keyword_rank == source.keyword_rank
    assert hit.keyword_score == source.keyword_score
    assert hit.keyword_rrf_contribution == source.keyword_rrf_contribution
    assert hit.sources == source.sources


def test_reranking_is_deterministic_and_does_not_mutate_input():
    ranked = _ranked(["a", "b", "c"])
    before = ranked.model_dump()
    backend = _backend_for(ranked, [2.0, 2.0, 1.0])
    reranker = CrossEncoderReranker(backend=backend)

    first = reranker.rerank(ranked)
    second = reranker.rerank(ranked)

    assert first == second
    assert ranked.model_dump() == before
    assert first.hits[0].chunk is ranked.hits[0].chunk


def test_reranker_does_not_access_retrieval_fusion_or_query_internals():
    source = inspect.getsource(reranker_module)

    for forbidden in (
        "VectorRetriever",
        "KeywordRetriever",
        "HybridRetriever",
        "ReciprocalRankFusion",
        "EmbeddingBackend",
        "SentenceTransformer",
        "FaissVectorIndex",
        "BM25Index",
        "MetadataFilter",
        ".search(",
        ".embed(",
        ".fuse(",
    ):
        assert forbidden not in source


def _backend_for(
    ranked: RankedHybridRetrievalResult,
    scores: Sequence[float],
    *,
    model_name: str = "fake/reranker",
) -> FakeRerankerBackend:
    return FakeRerankerBackend(
        {
            hit.chunk.text: score
            for hit, score in zip(ranked.hits, scores, strict=True)
        },
        model_name=model_name,
    )


def _ranked(
    chunk_ids: Sequence[str],
    *,
    query: str = "query",
    texts: Sequence[str] | None = None,
    dense_ranks: Sequence[int | None] | None = None,
    dense_scores: Sequence[float | None] | None = None,
    keyword_ranks: Sequence[int | None] | None = None,
    keyword_scores: Sequence[float | None] | None = None,
) -> RankedHybridRetrievalResult:
    count = len(chunk_ids)
    selected_texts = texts or [f"retrieval text for {chunk_id}" for chunk_id in chunk_ids]
    selected_dense_ranks = dense_ranks or list(range(1, count + 1))
    selected_dense_scores = dense_scores or [0.8 for _ in chunk_ids]
    selected_keyword_ranks = keyword_ranks or [None for _ in chunk_ids]
    selected_keyword_scores = keyword_scores or [None for _ in chunk_ids]
    hits = tuple(
        _hybrid_hit(
            chunk_id,
            text=text,
            rank=rank,
            dense_rank=dense_rank,
            dense_score=dense_score,
            keyword_rank=keyword_rank,
            keyword_score=keyword_score,
        )
        for chunk_id, text, rank, dense_rank, dense_score, keyword_rank, keyword_score in zip(
            chunk_ids,
            selected_texts,
            range(1, count + 1),
            selected_dense_ranks,
            selected_dense_scores,
            selected_keyword_ranks,
            selected_keyword_scores,
            strict=True,
        )
    )
    return RankedHybridRetrievalResult(
        query=query,
        hits=hits,
        candidate_count=count,
        returned_count=count,
        rank_constant=60,
        top_k=max(count, 1),
    )


def _hybrid_hit(
    chunk_id: str,
    *,
    text: str,
    rank: int,
    dense_rank: int | None,
    dense_score: float | None,
    keyword_rank: int | None,
    keyword_score: float | None,
) -> HybridSearchHit:
    dense_contribution = 0.0 if dense_rank is None else 1.0 / (60 + dense_rank)
    keyword_contribution = (
        0.0 if keyword_rank is None else 1.0 / (60 + keyword_rank)
    )
    sources = tuple(
        branch
        for branch, branch_rank in (
            (RetrievalBranch.DENSE, dense_rank),
            (RetrievalBranch.KEYWORD, keyword_rank),
        )
        if branch_rank is not None
    )
    return HybridSearchHit(
        chunk=_chunk(chunk_id, text=text),
        rank=rank,
        rrf_score=dense_contribution + keyword_contribution,
        dense_rank=dense_rank,
        dense_score=dense_score,
        dense_rrf_contribution=dense_contribution,
        keyword_rank=keyword_rank,
        keyword_score=keyword_score,
        keyword_rrf_contribution=keyword_contribution,
        sources=sources,
    )


def _chunk(chunk_id: str, *, text: str) -> RetrievalChunk:
    repository = GitHubRepository(owner="owner", name="repo")
    event_id = f"event-{chunk_id}"
    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        event_id=event_id,
        repository=repository,
        section_id=f"section-{chunk_id}",
        section_type=RetrievalSectionType.PATCH,
        chunk_index=0,
        content="source content",
        text=text,
        metadata=EventMetadata(event_id=event_id, repository=repository),
    )
