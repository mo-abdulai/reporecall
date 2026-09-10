import pytest
from pydantic import ValidationError

from reporecall.github import GitHubRepository
from reporecall.models import (
    EventMetadata,
    RerankedRetrievalResult,
    RerankedSearchHit,
    RetrievalBranch,
    RetrievalChunk,
    RetrievalSectionType,
)


def test_reranked_hit_preserves_all_hybrid_signals_and_negative_raw_score():
    hit = _hit(
        "a",
        rank=1,
        reranker_score=-2.4,
        previous_hybrid_rank=7,
        dense_rank=2,
        dense_score=0.81,
        keyword_rank=3,
        keyword_score=5.2,
    )

    assert hit.rank == 1
    assert hit.reranker_score == -2.4
    assert hit.reranker_model == "test/reranker"
    assert hit.previous_hybrid_rank == 7
    assert hit.rrf_score == pytest.approx(1 / 62 + 1 / 63)
    assert hit.dense_rank == 2
    assert hit.dense_score == 0.81
    assert hit.dense_rrf_contribution == pytest.approx(1 / 62)
    assert hit.keyword_rank == 3
    assert hit.keyword_score == 5.2
    assert hit.keyword_rrf_contribution == pytest.approx(1 / 63)
    assert hit.sources == (RetrievalBranch.DENSE, RetrievalBranch.KEYWORD)


def test_reranked_hit_normalizes_sources_and_is_immutable():
    hit = _hit(
        "a",
        rank=1,
        reranker_score=2.0,
        previous_hybrid_rank=1,
        dense_rank=1,
        dense_score=0.8,
        keyword_rank=1,
        keyword_score=4.2,
        sources=(RetrievalBranch.KEYWORD, RetrievalBranch.DENSE),
    )

    assert hit.sources == (RetrievalBranch.DENSE, RetrievalBranch.KEYWORD)
    with pytest.raises(ValidationError, match="frozen"):
        hit.rank = 2


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_reranked_hit_rejects_nonfinite_scores(score: float):
    with pytest.raises(ValidationError):
        _hit(
            "a",
            rank=1,
            reranker_score=score,
            previous_hybrid_rank=1,
            dense_rank=1,
            dense_score=0.8,
        )


@pytest.mark.parametrize("model_name", ["", "   "])
def test_reranked_hit_rejects_blank_model_name(model_name: str):
    with pytest.raises(ValidationError, match="must not be blank"):
        _hit(
            "a",
            rank=1,
            reranker_score=2.0,
            previous_hybrid_rank=1,
            dense_rank=1,
            dense_score=0.8,
            reranker_model=model_name,
        )


def test_reranked_hit_reuses_hybrid_provenance_validation():
    with pytest.raises(ValidationError, match="sum of branch contributions"):
        RerankedSearchHit(
            chunk=_chunk("a"),
            rank=1,
            reranker_score=2.0,
            reranker_model="test/reranker",
            previous_hybrid_rank=1,
            rrf_score=0.5,
            dense_rank=1,
            dense_score=0.8,
            dense_rrf_contribution=1 / 61,
            sources=(RetrievalBranch.DENSE,),
        )


def test_result_preserves_counts_order_model_and_query():
    first = _hit(
        "b",
        rank=1,
        reranker_score=8.0,
        previous_hybrid_rank=2,
        dense_rank=2,
        dense_score=0.7,
    )
    second = _hit(
        "a",
        rank=2,
        reranker_score=2.0,
        previous_hybrid_rank=1,
        dense_rank=1,
        dense_score=0.8,
    )

    result = RerankedRetrievalResult(
        query="  ConnectionResetError  ",
        model_name="test/reranker",
        input_candidate_count=7,
        reranked_candidate_count=5,
        returned_count=2,
        candidate_k=5,
        top_k=2,
        hits=(first, second),
    )

    assert result.query == "  ConnectionResetError  "
    assert result.hits == (first, second)
    assert result.input_candidate_count == 7
    assert result.reranked_candidate_count == 5
    assert result.returned_count == 2
    with pytest.raises(ValidationError, match="frozen"):
        result.returned_count = 1


def test_empty_result_is_valid_when_no_candidates_were_provided():
    result = RerankedRetrievalResult(
        query="query",
        model_name="test/reranker",
        input_candidate_count=0,
        reranked_candidate_count=0,
        returned_count=0,
        candidate_k=20,
        top_k=5,
        hits=(),
    )

    assert result.hits == ()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"top_k": 6}, "cannot exceed candidate-k"),
        ({"reranked_candidate_count": 4}, "bounded input count"),
        ({"returned_count": 1}, "reranker top-k result"),
        ({"model_name": " "}, "must not be blank"),
        ({"query": " "}, "must not be blank"),
    ],
)
def test_result_rejects_inconsistent_diagnostics(
    changes: dict[str, object],
    message: str,
):
    values = _result_values()
    values.update(changes)

    with pytest.raises(ValidationError, match=message):
        RerankedRetrievalResult(**values)


def test_result_requires_score_then_chunk_id_order_and_contiguous_ranks():
    lower = _hit(
        "a",
        rank=1,
        reranker_score=2.0,
        previous_hybrid_rank=1,
        dense_rank=1,
        dense_score=0.8,
    )
    higher = _hit(
        "b",
        rank=2,
        reranker_score=8.0,
        previous_hybrid_rank=2,
        dense_rank=2,
        dense_score=0.7,
    )
    values = _result_values()
    values["hits"] = (lower, higher)

    with pytest.raises(ValidationError, match="score and chunk-ID ordering"):
        RerankedRetrievalResult(**values)

    values["hits"] = (
        higher.model_copy(update={"rank": 1}),
        lower.model_copy(update={"rank": 3}),
    )
    with pytest.raises(ValidationError, match="contiguous and one-based"):
        RerankedRetrievalResult(**values)


def test_result_requires_unique_chunks_and_matching_models():
    first = _result_values()["hits"][0]
    assert isinstance(first, RerankedSearchHit)
    duplicate = first.model_copy(update={"rank": 2, "reranker_score": 1.0})
    values = _result_values()
    values["hits"] = (first, duplicate)

    with pytest.raises(ValidationError, match="chunk IDs must be unique"):
        RerankedRetrievalResult(**values)

    other_model = _hit(
        "b",
        rank=2,
        reranker_score=1.0,
        previous_hybrid_rank=2,
        dense_rank=2,
        dense_score=0.7,
        reranker_model="other/model",
    )
    values["hits"] = (first, other_model)
    with pytest.raises(ValidationError, match="models must match"):
        RerankedRetrievalResult(**values)


def _result_values() -> dict[str, object]:
    return {
        "query": "query",
        "model_name": "test/reranker",
        "input_candidate_count": 2,
        "reranked_candidate_count": 2,
        "returned_count": 2,
        "candidate_k": 2,
        "top_k": 2,
        "hits": (
            _hit(
                "a",
                rank=1,
                reranker_score=2.0,
                previous_hybrid_rank=1,
                dense_rank=1,
                dense_score=0.8,
            ),
            _hit(
                "b",
                rank=2,
                reranker_score=1.0,
                previous_hybrid_rank=2,
                dense_rank=2,
                dense_score=0.7,
            ),
        ),
    }


def _hit(
    chunk_id: str,
    *,
    rank: int,
    reranker_score: float,
    previous_hybrid_rank: int,
    dense_rank: int | None = None,
    dense_score: float | None = None,
    keyword_rank: int | None = None,
    keyword_score: float | None = None,
    reranker_model: str = "test/reranker",
    sources: tuple[RetrievalBranch, ...] | None = None,
) -> RerankedSearchHit:
    dense_contribution = 0.0 if dense_rank is None else 1.0 / (60 + dense_rank)
    keyword_contribution = (
        0.0 if keyword_rank is None else 1.0 / (60 + keyword_rank)
    )
    active_sources = sources or tuple(
        branch
        for branch, branch_rank in (
            (RetrievalBranch.DENSE, dense_rank),
            (RetrievalBranch.KEYWORD, keyword_rank),
        )
        if branch_rank is not None
    )
    return RerankedSearchHit(
        chunk=_chunk(chunk_id),
        rank=rank,
        reranker_score=reranker_score,
        reranker_model=reranker_model,
        previous_hybrid_rank=previous_hybrid_rank,
        rrf_score=dense_contribution + keyword_contribution,
        dense_rank=dense_rank,
        dense_score=dense_score,
        dense_rrf_contribution=dense_contribution,
        keyword_rank=keyword_rank,
        keyword_score=keyword_score,
        keyword_rrf_contribution=keyword_contribution,
        sources=active_sources,
    )


def _chunk(chunk_id: str) -> RetrievalChunk:
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
        content="Connection cleanup",
        text=f"Repository: owner/repo\nConnection cleanup for {chunk_id}",
        metadata=EventMetadata(event_id=event_id, repository=repository),
    )
