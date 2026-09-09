import pytest
from pydantic import ValidationError

from reporecall.github import GitHubRepository
from reporecall.models import (
    EventMetadata,
    HybridCandidate,
    HybridRetrievalResult,
    KeywordSearchHit,
    RetrievalBranch,
    RetrievalChunk,
    RetrievalSectionType,
    VectorSearchHit,
)


def test_dense_only_candidate_preserves_signal_and_provenance():
    chunk = _chunk("chunk-a")

    candidate = HybridCandidate(
        chunk=chunk,
        dense_rank=2,
        dense_score=0.81,
        sources=(RetrievalBranch.DENSE,),
    )

    assert candidate.chunk is chunk
    assert candidate.dense_rank == 2
    assert candidate.dense_score == 0.81
    assert candidate.keyword_rank is None
    assert candidate.keyword_score is None
    assert candidate.sources == (RetrievalBranch.DENSE,)


def test_keyword_only_candidate_preserves_signal():
    candidate = HybridCandidate(
        chunk=_chunk("chunk-a"),
        keyword_rank=3,
        keyword_score=4.27,
        sources=(RetrievalBranch.KEYWORD,),
    )

    assert candidate.dense_rank is None
    assert candidate.dense_score is None
    assert candidate.keyword_rank == 3
    assert candidate.keyword_score == 4.27
    assert candidate.sources == (RetrievalBranch.KEYWORD,)


def test_dual_candidate_preserves_different_rank_and_score_scales():
    candidate = HybridCandidate(
        chunk=_chunk("chunk-a"),
        dense_rank=4,
        dense_score=0.91,
        keyword_rank=1,
        keyword_score=8.73,
        sources=(RetrievalBranch.DENSE, RetrievalBranch.KEYWORD),
    )

    assert candidate.dense_rank == 4
    assert candidate.dense_score == 0.91
    assert candidate.keyword_rank == 1
    assert candidate.keyword_score == 8.73


def test_candidate_normalizes_source_order_deterministically():
    candidate = HybridCandidate(
        chunk=_chunk("chunk-a"),
        dense_rank=1,
        dense_score=0.8,
        keyword_rank=2,
        keyword_score=3.4,
        sources=(RetrievalBranch.KEYWORD, RetrievalBranch.DENSE),
    )

    assert candidate.sources == (
        RetrievalBranch.DENSE,
        RetrievalBranch.KEYWORD,
    )


def test_candidate_rejects_duplicate_sources():
    with pytest.raises(ValidationError, match="must be unique"):
        HybridCandidate(
            chunk=_chunk("chunk-a"),
            dense_rank=1,
            dense_score=0.8,
            sources=(RetrievalBranch.DENSE, RetrievalBranch.DENSE),
        )


@pytest.mark.parametrize(
    "values",
    [
        {"dense_rank": 1, "sources": (RetrievalBranch.DENSE,)},
        {"dense_score": 0.8, "sources": (RetrievalBranch.DENSE,)},
        {"keyword_rank": 1, "sources": (RetrievalBranch.KEYWORD,)},
        {"keyword_score": 4.2, "sources": (RetrievalBranch.KEYWORD,)},
    ],
)
def test_candidate_rejects_unpaired_rank_and_score(values: dict[str, object]):
    with pytest.raises(ValidationError, match="must appear together"):
        HybridCandidate(chunk=_chunk("chunk-a"), **values)


@pytest.mark.parametrize(
    "values",
    [
        {
            "dense_rank": 1,
            "dense_score": 0.8,
            "sources": (RetrievalBranch.KEYWORD,),
        },
        {
            "keyword_rank": 1,
            "keyword_score": 4.2,
            "sources": (RetrievalBranch.DENSE,),
        },
    ],
)
def test_candidate_rejects_sources_that_disagree_with_signals(
    values: dict[str, object],
):
    with pytest.raises(ValidationError, match="sources must agree"):
        HybridCandidate(chunk=_chunk("chunk-a"), **values)


def test_candidate_rejects_no_retrieval_signal():
    with pytest.raises(ValidationError, match="retrieval signal"):
        HybridCandidate(
            chunk=_chunk("chunk-a"),
            sources=(RetrievalBranch.DENSE,),
        )


@pytest.mark.parametrize("rank", [0, -1])
def test_candidate_rejects_nonpositive_branch_ranks(rank: int):
    with pytest.raises(ValidationError):
        HybridCandidate(
            chunk=_chunk("chunk-a"),
            dense_rank=rank,
            dense_score=0.8,
            sources=(RetrievalBranch.DENSE,),
        )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_candidate_rejects_nonfinite_branch_scores(score: float):
    with pytest.raises(ValidationError):
        HybridCandidate(
            chunk=_chunk("chunk-a"),
            keyword_rank=1,
            keyword_score=score,
            sources=(RetrievalBranch.KEYWORD,),
        )


def test_candidate_has_no_fused_score_or_hybrid_rank_and_is_immutable():
    candidate = _dense_candidate(_chunk("chunk-a"), rank=1, score=0.8)

    assert "hybrid_score" not in HybridCandidate.model_fields
    assert "hybrid_rank" not in HybridCandidate.model_fields
    with pytest.raises(ValidationError, match="frozen"):
        candidate.dense_rank = 2


def test_result_preserves_query_hits_candidates_and_counts():
    chunk_a = _chunk("chunk-a")
    chunk_b = _chunk("chunk-b")
    dense_hits = (
        VectorSearchHit(chunk=chunk_a, score=0.91, rank=1),
        VectorSearchHit(chunk=chunk_b, score=0.82, rank=2),
    )
    keyword_hits = (KeywordSearchHit(chunk=chunk_a, score=4.5, rank=1),)
    candidates = (
        HybridCandidate(
            chunk=chunk_a,
            dense_rank=1,
            dense_score=0.91,
            keyword_rank=1,
            keyword_score=4.5,
            sources=(RetrievalBranch.DENSE, RetrievalBranch.KEYWORD),
        ),
        _dense_candidate(chunk_b, rank=2, score=0.82),
    )

    result = HybridRetrievalResult(
        query="  ConnectionResetError  ",
        dense_hits=dense_hits,
        keyword_hits=keyword_hits,
        candidates=candidates,
        dense_retrieved_count=2,
        keyword_retrieved_count=1,
        unique_candidate_count=2,
    )

    assert result.query == "  ConnectionResetError  "
    assert result.dense_hits == dense_hits
    assert result.keyword_hits == keyword_hits
    assert result.candidates == candidates
    assert result.dense_retrieved_count == 2
    assert result.keyword_retrieved_count == 1
    assert result.unique_candidate_count == 2
    with pytest.raises(ValidationError, match="frozen"):
        result.unique_candidate_count = 3


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_result_rejects_blank_query(query: str):
    with pytest.raises(ValidationError, match="must not be blank"):
        HybridRetrievalResult(
            query=query,
            dense_hits=(),
            keyword_hits=(),
            candidates=(),
            dense_retrieved_count=0,
            keyword_retrieved_count=0,
            unique_candidate_count=0,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("dense_retrieved_count", "Dense retrieved count"),
        ("keyword_retrieved_count", "Keyword retrieved count"),
        ("unique_candidate_count", "Unique candidate count"),
    ],
)
def test_result_rejects_incorrect_counts(field: str, message: str):
    values = _valid_result_values()
    values[field] = 99

    with pytest.raises(ValidationError, match=message):
        HybridRetrievalResult(**values)


def test_result_rejects_duplicate_candidate_identities():
    values = _valid_result_values()
    candidate = values["candidates"][0]
    values["candidates"] = (candidate, candidate)
    values["unique_candidate_count"] = 2

    with pytest.raises(ValidationError, match="chunk IDs must be unique"):
        HybridRetrievalResult(**values)


def test_result_rejects_non_neutral_candidate_order():
    chunk_a = _chunk("chunk-a")
    chunk_b = _chunk("chunk-b")
    dense_hits = (
        VectorSearchHit(chunk=chunk_a, score=0.9, rank=1),
        VectorSearchHit(chunk=chunk_b, score=0.8, rank=2),
    )

    with pytest.raises(ValidationError, match="chunk-ID ordering"):
        HybridRetrievalResult(
            query="query",
            dense_hits=dense_hits,
            keyword_hits=(),
            candidates=(
                _dense_candidate(chunk_b, rank=2, score=0.8),
                _dense_candidate(chunk_a, rank=1, score=0.9),
            ),
            dense_retrieved_count=2,
            keyword_retrieved_count=0,
            unique_candidate_count=2,
        )


def test_result_rejects_candidates_that_do_not_equal_branch_union():
    values = _valid_result_values()
    values["candidates"] = ()
    values["unique_candidate_count"] = 0

    with pytest.raises(ValidationError, match="union of branch hits"):
        HybridRetrievalResult(**values)


def test_result_rejects_duplicate_branch_identities_and_ranks():
    chunk_a = _chunk("chunk-a")
    chunk_b = _chunk("chunk-b")
    duplicate_id_values = _valid_result_values()
    duplicate_id_values["dense_hits"] = (
        VectorSearchHit(chunk=chunk_a, score=0.9, rank=1),
        VectorSearchHit(chunk=chunk_a, score=0.8, rank=2),
    )
    duplicate_id_values["dense_retrieved_count"] = 2

    with pytest.raises(ValidationError, match="duplicate chunk ID"):
        HybridRetrievalResult(**duplicate_id_values)

    duplicate_rank_values = _valid_result_values()
    duplicate_rank_values["dense_hits"] = (
        VectorSearchHit(chunk=chunk_a, score=0.9, rank=1),
        VectorSearchHit(chunk=chunk_b, score=0.8, rank=1),
    )
    duplicate_rank_values["dense_retrieved_count"] = 2

    with pytest.raises(ValidationError, match="duplicate rank"):
        HybridRetrievalResult(**duplicate_rank_values)


def test_result_rejects_candidate_that_changes_branch_signal():
    values = _valid_result_values()
    chunk = values["dense_hits"][0].chunk
    values["candidates"] = (
        _dense_candidate(chunk, rank=1, score=0.1),
    )

    with pytest.raises(ValidationError, match="preserve its dense hit"):
        HybridRetrievalResult(**values)


def _valid_result_values() -> dict[str, object]:
    chunk = _chunk("chunk-a")
    dense_hit = VectorSearchHit(chunk=chunk, score=0.9, rank=1)
    return {
        "query": "query",
        "dense_hits": (dense_hit,),
        "keyword_hits": (),
        "candidates": (_dense_candidate(chunk, rank=1, score=0.9),),
        "dense_retrieved_count": 1,
        "keyword_retrieved_count": 0,
        "unique_candidate_count": 1,
    }


def _dense_candidate(
    chunk: RetrievalChunk,
    *,
    rank: int,
    score: float,
) -> HybridCandidate:
    return HybridCandidate(
        chunk=chunk,
        dense_rank=rank,
        dense_score=score,
        sources=(RetrievalBranch.DENSE,),
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
