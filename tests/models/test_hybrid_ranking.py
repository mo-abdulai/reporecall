import pytest
from pydantic import ValidationError

from reporecall.github import GitHubRepository
from reporecall.models import (
    EventMetadata,
    HybridSearchHit,
    RankedHybridRetrievalResult,
    RetrievalBranch,
    RetrievalChunk,
    RetrievalSectionType,
)


def test_dense_only_hit_preserves_raw_signal_and_zero_missing_contribution():
    hit = _hit("a", rank=1, dense_rank=2, dense_score=0.81)

    assert hit.dense_rank == 2
    assert hit.dense_score == 0.81
    assert hit.dense_rrf_contribution == pytest.approx(1 / 62)
    assert hit.keyword_rank is None
    assert hit.keyword_score is None
    assert hit.keyword_rrf_contribution == 0
    assert hit.sources == (RetrievalBranch.DENSE,)


def test_keyword_only_hit_treats_keyword_branch_symmetrically():
    hit = _hit("a", rank=1, keyword_rank=2, keyword_score=8.4)

    assert hit.dense_rrf_contribution == 0
    assert hit.keyword_rrf_contribution == pytest.approx(1 / 62)
    assert hit.rrf_score == pytest.approx(1 / 62)
    assert hit.sources == (RetrievalBranch.KEYWORD,)


def test_dual_hit_preserves_scores_ranks_sources_and_contributions():
    hit = _hit(
        "a",
        rank=1,
        dense_rank=1,
        dense_score=0.74482,
        keyword_rank=3,
        keyword_score=5.318,
    )

    assert hit.dense_rank == 1
    assert hit.dense_score == 0.74482
    assert hit.dense_rrf_contribution == pytest.approx(1 / 61)
    assert hit.keyword_rank == 3
    assert hit.keyword_score == 5.318
    assert hit.keyword_rrf_contribution == pytest.approx(1 / 63)
    assert hit.rrf_score == pytest.approx(1 / 61 + 1 / 63)
    assert hit.sources == (RetrievalBranch.DENSE, RetrievalBranch.KEYWORD)


def test_hit_normalizes_branch_order_and_is_immutable():
    hit = _hit(
        "a",
        rank=1,
        dense_rank=1,
        dense_score=0.8,
        keyword_rank=1,
        keyword_score=4.2,
        sources=(RetrievalBranch.KEYWORD, RetrievalBranch.DENSE),
    )

    assert hit.sources == (RetrievalBranch.DENSE, RetrievalBranch.KEYWORD)
    with pytest.raises(ValidationError, match="frozen"):
        hit.rank = 2


def test_hit_rejects_duplicate_branches():
    with pytest.raises(ValidationError, match="must be unique"):
        _hit(
            "a",
            rank=1,
            dense_rank=1,
            dense_score=0.8,
            sources=(RetrievalBranch.DENSE, RetrievalBranch.DENSE),
        )


@pytest.mark.parametrize("rank", [0, -1])
def test_hit_rejects_nonpositive_final_rank(rank: int):
    with pytest.raises(ValidationError):
        _hit("a", rank=rank, dense_rank=1, dense_score=0.8)


@pytest.mark.parametrize("rank", [0, -1])
def test_hit_rejects_nonpositive_branch_rank(rank: int):
    with pytest.raises(ValidationError):
        _hit("a", rank=1, dense_rank=rank, dense_score=0.8)


@pytest.mark.parametrize("score", [0.0, -1.0, float("nan"), float("inf")])
def test_hit_rejects_nonpositive_or_nonfinite_rrf_score(score: float):
    with pytest.raises(ValidationError):
        HybridSearchHit(
            chunk=_chunk("a"),
            rank=1,
            rrf_score=score,
            dense_rank=1,
            dense_score=0.8,
            dense_rrf_contribution=1 / 61,
            sources=(RetrievalBranch.DENSE,),
        )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_hit_rejects_nonfinite_raw_branch_scores(score: float):
    with pytest.raises(ValidationError):
        _hit("a", rank=1, dense_rank=1, dense_score=score)


@pytest.mark.parametrize(
    "contribution",
    [-0.1, float("nan"), float("inf"), float("-inf")],
)
def test_hit_rejects_negative_or_nonfinite_contributions(contribution: float):
    with pytest.raises(ValidationError):
        HybridSearchHit(
            chunk=_chunk("a"),
            rank=1,
            rrf_score=1 / 61,
            dense_rank=1,
            dense_score=0.8,
            dense_rrf_contribution=contribution,
            sources=(RetrievalBranch.DENSE,),
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
def test_hit_rejects_unpaired_branch_rank_and_score(values: dict[str, object]):
    with pytest.raises(ValidationError, match="must appear together"):
        HybridSearchHit(
            chunk=_chunk("a"),
            rank=1,
            rrf_score=1 / 61,
            dense_rrf_contribution=(
                1 / 61 if RetrievalBranch.DENSE in values["sources"] else 0
            ),
            keyword_rrf_contribution=(
                1 / 61 if RetrievalBranch.KEYWORD in values["sources"] else 0
            ),
            **values,
        )


@pytest.mark.parametrize(
    "values",
    [
        {
            "dense_rank": 1,
            "dense_score": 0.8,
            "dense_rrf_contribution": 1 / 61,
            "sources": (RetrievalBranch.KEYWORD,),
        },
        {
            "keyword_rank": 1,
            "keyword_score": 4.2,
            "keyword_rrf_contribution": 1 / 61,
            "sources": (RetrievalBranch.DENSE,),
        },
    ],
)
def test_hit_rejects_branch_membership_that_disagrees_with_signal(
    values: dict[str, object],
):
    with pytest.raises(ValidationError, match="branch must agree"):
        HybridSearchHit(
            chunk=_chunk("a"),
            rank=1,
            rrf_score=1 / 61,
            **values,
        )


def test_hit_requires_positive_contribution_for_present_branch():
    with pytest.raises(ValidationError, match="must be positive"):
        HybridSearchHit(
            chunk=_chunk("a"),
            rank=1,
            rrf_score=1 / 61,
            dense_rank=1,
            dense_score=0.8,
            dense_rrf_contribution=0,
            sources=(RetrievalBranch.DENSE,),
        )


def test_hit_requires_zero_contribution_for_missing_branch():
    with pytest.raises(ValidationError, match="must be zero"):
        HybridSearchHit(
            chunk=_chunk("a"),
            rank=1,
            rrf_score=2 / 61,
            dense_rank=1,
            dense_score=0.8,
            dense_rrf_contribution=1 / 61,
            keyword_rrf_contribution=1 / 61,
            sources=(RetrievalBranch.DENSE,),
        )


def test_hit_requires_rrf_score_to_equal_contribution_sum():
    with pytest.raises(ValidationError, match="sum of branch contributions"):
        HybridSearchHit(
            chunk=_chunk("a"),
            rank=1,
            rrf_score=0.5,
            dense_rank=1,
            dense_score=0.8,
            dense_rrf_contribution=1 / 61,
            sources=(RetrievalBranch.DENSE,),
        )


def test_ranked_result_preserves_query_diagnostics_and_hits():
    first = _hit(
        "a",
        rank=1,
        dense_rank=1,
        dense_score=0.8,
        keyword_rank=3,
        keyword_score=4.2,
    )
    second = _hit("b", rank=2, dense_rank=2, dense_score=0.7)

    result = RankedHybridRetrievalResult(
        query="  ConnectionResetError  ",
        hits=(first, second),
        candidate_count=5,
        returned_count=2,
        rank_constant=60,
        top_k=2,
    )

    assert result.query == "  ConnectionResetError  "
    assert result.hits == (first, second)
    assert result.candidate_count == 5
    assert result.returned_count == 2
    assert result.rank_constant == 60
    assert result.top_k == 2
    with pytest.raises(ValidationError, match="frozen"):
        result.returned_count = 1


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_ranked_result_rejects_blank_query(query: str):
    with pytest.raises(ValidationError, match="must not be blank"):
        RankedHybridRetrievalResult(
            query=query,
            hits=(),
            candidate_count=0,
            returned_count=0,
            rank_constant=60,
            top_k=10,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"returned_count": 0}, "match ranked hybrid hits"),
        ({"candidate_count": 0}, "cannot exceed candidate count"),
        ({"top_k": 0}, "greater than 0"),
        ({"rank_constant": 0}, "greater than 0"),
    ],
)
def test_ranked_result_rejects_invalid_diagnostics(
    changes: dict[str, int],
    message: str,
):
    values = _ranked_result_values()
    values.update(changes)

    with pytest.raises(ValidationError, match=message):
        RankedHybridRetrievalResult(**values)


def test_ranked_result_rejects_nonempty_candidates_without_hits():
    with pytest.raises(ValidationError, match="must produce"):
        RankedHybridRetrievalResult(
            query="query",
            hits=(),
            candidate_count=1,
            returned_count=0,
            rank_constant=60,
            top_k=10,
        )


def test_ranked_result_requires_unique_chunks_and_contiguous_ranks():
    duplicate = _hit("a", rank=2, dense_rank=2, dense_score=0.7)
    values = _ranked_result_values()
    values["hits"] = (values["hits"][0], duplicate)
    values["returned_count"] = 2
    values["candidate_count"] = 2
    values["top_k"] = 2

    with pytest.raises(ValidationError, match="chunk IDs must be unique"):
        RankedHybridRetrievalResult(**values)

    skipped_rank = _hit("b", rank=3, dense_rank=2, dense_score=0.7)
    values["hits"] = (values["hits"][0], skipped_rank)

    with pytest.raises(ValidationError, match="contiguous and one-based"):
        RankedHybridRetrievalResult(**values)


def test_ranked_result_requires_score_then_chunk_identity_ordering():
    higher = _hit("b", rank=2, dense_rank=1, dense_score=0.8)
    lower = _hit("a", rank=1, dense_rank=2, dense_score=0.7)

    with pytest.raises(ValidationError, match="RRF score and chunk-ID ordering"):
        RankedHybridRetrievalResult(
            query="query",
            hits=(lower, higher),
            candidate_count=2,
            returned_count=2,
            rank_constant=60,
            top_k=2,
        )


def test_ranked_result_validates_contributions_against_rank_constant():
    hit_using_sixty = _hit("a", rank=1, dense_rank=1, dense_score=0.8)

    with pytest.raises(ValidationError, match="RRF formula"):
        RankedHybridRetrievalResult(
            query="query",
            hits=(hit_using_sixty,),
            candidate_count=1,
            returned_count=1,
            rank_constant=10,
            top_k=1,
        )


def _ranked_result_values() -> dict[str, object]:
    return {
        "query": "query",
        "hits": (_hit("a", rank=1, dense_rank=1, dense_score=0.8),),
        "candidate_count": 1,
        "returned_count": 1,
        "rank_constant": 60,
        "top_k": 1,
    }


def _hit(
    chunk_id: str,
    *,
    rank: int,
    dense_rank: int | None = None,
    dense_score: float | None = None,
    keyword_rank: int | None = None,
    keyword_score: float | None = None,
    rank_constant: int = 60,
    sources: tuple[RetrievalBranch, ...] | None = None,
) -> HybridSearchHit:
    dense_contribution = (
        0.0 if dense_rank is None else 1.0 / (rank_constant + dense_rank)
    )
    keyword_contribution = (
        0.0 if keyword_rank is None else 1.0 / (rank_constant + keyword_rank)
    )
    active_sources = sources or tuple(
        branch
        for branch, branch_rank in (
            (RetrievalBranch.DENSE, dense_rank),
            (RetrievalBranch.KEYWORD, keyword_rank),
        )
        if branch_rank is not None
    )
    return HybridSearchHit(
        chunk=_chunk(chunk_id),
        rank=rank,
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
