import inspect
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

import reporecall.retrieval.reciprocal_rank_fusion as fusion_module
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
from reporecall.retrieval import ReciprocalRankFusion, ReciprocalRankFusionConfig


def test_config_has_conventional_immutable_defaults_and_custom_values():
    defaults = ReciprocalRankFusionConfig()
    custom = ReciprocalRankFusionConfig(rank_constant=10, top_k=3)

    assert defaults.rank_constant == 60
    assert defaults.top_k == 10
    assert custom.rank_constant == 10
    assert custom.top_k == 3
    with pytest.raises(ValidationError, match="frozen"):
        custom.top_k = 5


@pytest.mark.parametrize(
    "values",
    [
        {"rank_constant": 0},
        {"rank_constant": -1},
        {"top_k": 0},
        {"top_k": -1},
    ],
)
def test_config_rejects_nonpositive_values(values: dict[str, int]):
    with pytest.raises(ValidationError):
        ReciprocalRankFusionConfig(**values)


def test_default_dense_only_formula_uses_existing_one_based_rank():
    candidate = _candidate("a", dense_rank=1, dense_score=0.8)

    hit = _fuse([candidate]).hits[0]

    assert hit.dense_rrf_contribution == pytest.approx(1 / 61)
    assert hit.keyword_rrf_contribution == 0
    assert hit.rrf_score == pytest.approx(1 / 61)
    assert hit.rank == 1


def test_default_keyword_only_formula_is_symmetric():
    candidate = _candidate("a", keyword_rank=1, keyword_score=9.2)

    hit = _fuse([candidate]).hits[0]

    assert hit.dense_rrf_contribution == 0
    assert hit.keyword_rrf_contribution == pytest.approx(1 / 61)
    assert hit.rrf_score == pytest.approx(1 / 61)


def test_dual_rank_one_candidate_receives_sum_of_branch_contributions():
    candidate = _candidate(
        "a",
        dense_rank=1,
        dense_score=0.8,
        keyword_rank=1,
        keyword_score=5.2,
    )

    hit = _fuse([candidate]).hits[0]

    assert hit.dense_rrf_contribution == pytest.approx(1 / 61)
    assert hit.keyword_rrf_contribution == pytest.approx(1 / 61)
    assert hit.rrf_score == pytest.approx(2 / 61)


def test_multi_branch_example_uses_rrf_only_and_neutral_ties():
    candidates = [
        _candidate(
            "a",
            dense_rank=1,
            dense_score=0.82,
            keyword_rank=3,
            keyword_score=2.95,
        ),
        _candidate("b", dense_rank=2, dense_score=0.74),
        _candidate(
            "c",
            dense_rank=3,
            dense_score=0.69,
            keyword_rank=1,
            keyword_score=4.81,
        ),
        _candidate("d", keyword_rank=2, keyword_score=3.72),
    ]

    ranked = _fuse(candidates)

    assert [hit.chunk.chunk_id for hit in ranked.hits] == ["a", "c", "b", "d"]
    assert [hit.rank for hit in ranked.hits] == [1, 2, 3, 4]
    by_id = {hit.chunk.chunk_id: hit for hit in ranked.hits}
    assert by_id["a"].rrf_score == pytest.approx(1 / 61 + 1 / 63)
    assert by_id["b"].rrf_score == pytest.approx(1 / 62)
    assert by_id["c"].rrf_score == pytest.approx(1 / 63 + 1 / 61)
    assert by_id["d"].rrf_score == pytest.approx(1 / 62)


def test_branch_symmetry_uses_chunk_id_to_break_equal_rrf_scores():
    dense_first = _candidate(
        "a",
        dense_rank=1,
        dense_score=0.8,
        keyword_rank=3,
        keyword_score=2.0,
    )
    keyword_first = _candidate(
        "c",
        dense_rank=3,
        dense_score=0.6,
        keyword_rank=1,
        keyword_score=8.0,
    )

    hits = _fuse([keyword_first, dense_first]).hits

    assert hits[0].rrf_score == hits[1].rrf_score
    assert [hit.chunk.chunk_id for hit in hits] == ["a", "c"]


def test_dense_raw_score_does_not_override_branch_rank():
    candidates = [
        _candidate("a", dense_rank=1, dense_score=0.51),
        _candidate("b", dense_rank=2, dense_score=0.99),
    ]

    assert [hit.chunk.chunk_id for hit in _fuse(candidates).hits] == ["a", "b"]


def test_keyword_raw_score_does_not_override_branch_rank():
    candidates = [
        _candidate("a", keyword_rank=1, keyword_score=1.0),
        _candidate("b", keyword_rank=2, keyword_score=100.0),
    ]

    assert [hit.chunk.chunk_id for hit in _fuse(candidates).hits] == ["a", "b"]


def test_cross_scale_raw_scores_are_preserved_but_not_combined():
    candidate = _candidate(
        "a",
        dense_rank=2,
        dense_score=0.8,
        keyword_rank=1,
        keyword_score=12.7,
    )

    hit = _fuse([candidate]).hits[0]

    assert hit.dense_score == 0.8
    assert hit.keyword_score == 12.7
    assert hit.rrf_score == pytest.approx(1 / 62 + 1 / 61)


def test_custom_rank_constant_changes_formula_without_hard_coded_sixty():
    fusion = ReciprocalRankFusion(
        ReciprocalRankFusionConfig(rank_constant=10, top_k=5)
    )
    result = _result(
        [_candidate("a", dense_rank=1, dense_score=0.8, keyword_rank=3, keyword_score=4.2)]
    )

    ranked = fusion.fuse(result)

    assert ranked.hits[0].dense_rrf_contribution == pytest.approx(1 / 11)
    assert ranked.hits[0].keyword_rrf_contribution == pytest.approx(1 / 13)
    assert ranked.hits[0].rrf_score == pytest.approx(1 / 11 + 1 / 13)
    assert ranked.rank_constant == 10


def test_all_candidates_are_scored_before_final_top_k():
    candidates = [
        _candidate("a", dense_rank=1, dense_score=0.9),
        _candidate("b", dense_rank=2, dense_score=0.8),
        _candidate("c", dense_rank=3, dense_score=0.7),
        _candidate("d", keyword_rank=2, keyword_score=4.0),
        _candidate(
            "z",
            dense_rank=4,
            dense_score=0.6,
            keyword_rank=1,
            keyword_score=5.0,
        ),
    ]
    fusion = ReciprocalRankFusion(ReciprocalRankFusionConfig(top_k=3))

    ranked = fusion.fuse(_result(candidates))

    assert ranked.candidate_count == 5
    assert ranked.returned_count == 3
    assert [hit.chunk.chunk_id for hit in ranked.hits] == ["z", "a", "b"]


def test_top_k_does_not_change_scores_or_ranks_of_retained_prefix():
    candidates = [
        _candidate("a", dense_rank=1, dense_score=0.9),
        _candidate("b", dense_rank=2, dense_score=0.8),
        _candidate("c", keyword_rank=1, keyword_score=5.0),
        _candidate("d", keyword_rank=2, keyword_score=4.0),
        _candidate("e", dense_rank=3, dense_score=0.7),
    ]
    result = _result(candidates)

    top_three = ReciprocalRankFusion(
        ReciprocalRankFusionConfig(top_k=3)
    ).fuse(result)
    top_ten = ReciprocalRankFusion(
        ReciprocalRankFusionConfig(top_k=10)
    ).fuse(result)

    assert top_three.hits == top_ten.hits[:3]


def test_top_k_larger_than_candidates_returns_only_available_hits():
    ranked = _fuse(
        [
            _candidate("a", dense_rank=1, dense_score=0.9),
            _candidate("b", keyword_rank=1, keyword_score=5.0),
        ],
        top_k=10,
    )

    assert ranked.candidate_count == 2
    assert ranked.returned_count == 2


def test_empty_input_returns_valid_empty_ranked_result():
    result = _result([], query="ConnectionResetError")

    ranked = ReciprocalRankFusion().fuse(result)

    assert ranked.query == "ConnectionResetError"
    assert ranked.hits == ()
    assert ranked.candidate_count == 0
    assert ranked.returned_count == 0
    assert ranked.rank_constant == 60
    assert ranked.top_k == 10


def test_fusion_is_deterministic_and_input_order_independent():
    candidates = [
        _candidate("a", dense_rank=1, dense_score=0.9),
        _candidate("b", keyword_rank=1, keyword_score=5.0),
        _candidate("c", dense_rank=2, dense_score=0.8),
    ]
    result = _result(candidates)
    reordered = result.model_copy(update={"candidates": tuple(reversed(result.candidates))})
    fusion = ReciprocalRankFusion()

    first = fusion.fuse(result)
    second = fusion.fuse(result)
    from_reordered = fusion.fuse(reordered)

    assert first == second
    assert from_reordered == first


def test_fusion_does_not_mutate_result_candidates_or_chunks():
    candidate = _candidate(
        "a",
        dense_rank=1,
        dense_score=0.8,
        keyword_rank=2,
        keyword_score=4.2,
    )
    result = _result([candidate])
    original_result = result.model_dump()
    original_candidate = candidate.model_dump()
    original_chunk = candidate.chunk.model_dump()

    ranked = ReciprocalRankFusion().fuse(result)

    assert result.model_dump() == original_result
    assert candidate.model_dump() == original_candidate
    assert candidate.chunk.model_dump() == original_chunk
    assert ranked.hits[0].chunk is candidate.chunk


def test_final_hit_preserves_complete_chunk_provenance_and_section_type():
    repository = GitHubRepository(owner="other", name="service")
    chunk = _chunk(
        "review",
        repository=repository,
        section_type=RetrievalSectionType.REVIEW_COMMENT,
    )
    candidate = _candidate_for_chunk(chunk, keyword_rank=1, keyword_score=4.2)

    hit = _fuse([candidate]).hits[0]

    assert hit.chunk is chunk
    assert hit.chunk.chunk_id == chunk.chunk_id
    assert hit.chunk.document_id == chunk.document_id
    assert hit.chunk.event_id == chunk.event_id
    assert hit.chunk.repository == repository
    assert hit.chunk.section_id == chunk.section_id
    assert hit.chunk.section_type is RetrievalSectionType.REVIEW_COMMENT
    assert hit.chunk.artifact == chunk.artifact
    assert hit.chunk.metadata == chunk.metadata
    assert hit.chunk.content == chunk.content
    assert hit.chunk.text == chunk.text


def test_section_repository_and_metadata_do_not_boost_rrf_score():
    issue = _candidate_for_chunk(
        _chunk("a", section_type=RetrievalSectionType.ISSUE),
        dense_rank=1,
        dense_score=0.8,
    )
    patch = _candidate_for_chunk(
        _chunk(
            "b",
            repository=GitHubRepository(owner="other", name="repo"),
            section_type=RetrievalSectionType.PATCH,
        ),
        keyword_rank=1,
        keyword_score=5.0,
    )

    hits = _fuse([patch, issue]).hits

    assert hits[0].rrf_score == hits[1].rrf_score
    assert [hit.chunk.chunk_id for hit in hits] == ["a", "b"]


def test_fusion_layer_does_not_access_query_or_retrieval_internals():
    source = inspect.getsource(fusion_module)

    for forbidden in (
        "VectorRetriever",
        "KeywordRetriever",
        "HybridRetriever",
        "EmbeddingBackend",
        "SentenceTransformer",
        "FaissVectorIndex",
        "BM25Index",
        "EngineeringTokenizer",
        "bm25s",
        ".search(",
        ".embed(",
        ".tokenize(",
    ):
        assert forbidden not in source


def _fuse(
    candidates: Sequence[HybridCandidate],
    *,
    top_k: int = 10,
):
    return ReciprocalRankFusion(
        ReciprocalRankFusionConfig(top_k=top_k)
    ).fuse(_result(candidates))


def _result(
    candidates: Sequence[HybridCandidate],
    *,
    query: str = "query",
) -> HybridRetrievalResult:
    ordered_candidates = tuple(sorted(candidates, key=lambda item: item.chunk.chunk_id))
    dense_hits = tuple(
        sorted(
            (
                VectorSearchHit(
                    chunk=candidate.chunk,
                    rank=candidate.dense_rank,
                    score=candidate.dense_score,
                )
                for candidate in ordered_candidates
                if candidate.dense_rank is not None
                and candidate.dense_score is not None
            ),
            key=lambda hit: hit.rank,
        )
    )
    keyword_hits = tuple(
        sorted(
            (
                KeywordSearchHit(
                    chunk=candidate.chunk,
                    rank=candidate.keyword_rank,
                    score=candidate.keyword_score,
                )
                for candidate in ordered_candidates
                if candidate.keyword_rank is not None
                and candidate.keyword_score is not None
            ),
            key=lambda hit: hit.rank,
        )
    )
    return HybridRetrievalResult(
        query=query,
        dense_hits=dense_hits,
        keyword_hits=keyword_hits,
        candidates=ordered_candidates,
        dense_retrieved_count=len(dense_hits),
        keyword_retrieved_count=len(keyword_hits),
        unique_candidate_count=len(ordered_candidates),
    )


def _candidate(
    chunk_id: str,
    *,
    dense_rank: int | None = None,
    dense_score: float | None = None,
    keyword_rank: int | None = None,
    keyword_score: float | None = None,
) -> HybridCandidate:
    return _candidate_for_chunk(
        _chunk(chunk_id),
        dense_rank=dense_rank,
        dense_score=dense_score,
        keyword_rank=keyword_rank,
        keyword_score=keyword_score,
    )


def _candidate_for_chunk(
    chunk: RetrievalChunk,
    *,
    dense_rank: int | None = None,
    dense_score: float | None = None,
    keyword_rank: int | None = None,
    keyword_score: float | None = None,
) -> HybridCandidate:
    sources = tuple(
        branch
        for branch, rank in (
            (RetrievalBranch.DENSE, dense_rank),
            (RetrievalBranch.KEYWORD, keyword_rank),
        )
        if rank is not None
    )
    return HybridCandidate(
        chunk=chunk,
        dense_rank=dense_rank,
        dense_score=dense_score,
        keyword_rank=keyword_rank,
        keyword_score=keyword_score,
        sources=sources,
    )


def _chunk(
    chunk_id: str,
    *,
    repository: GitHubRepository | None = None,
    section_type: RetrievalSectionType = RetrievalSectionType.PATCH,
) -> RetrievalChunk:
    active_repository = repository or GitHubRepository(owner="owner", name="repo")
    event_id = f"event-{chunk_id}"
    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        event_id=event_id,
        repository=active_repository,
        section_id=f"section-{chunk_id}",
        section_type=section_type,
        chunk_index=0,
        content="Connection cleanup",
        text=f"Repository: {active_repository.owner}/{active_repository.name}\nConnection cleanup",
        metadata=EventMetadata(event_id=event_id, repository=active_repository),
    )
