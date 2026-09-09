import hashlib
import inspect
import math
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

import reporecall.retrieval.hybrid_retriever as hybrid_retriever_module
from reporecall.github import GitHubRepository
from reporecall.models import (
    ChunkEmbedding,
    EventMetadata,
    KeywordSearchHit,
    MetadataFilter,
    RetrievalBranch,
    RetrievalChunk,
    RetrievalSectionType,
    VectorSearchHit,
)
from reporecall.retrieval import (
    BM25Index,
    FaissVectorIndex,
    HybridRetrievalConfig,
    HybridRetrievalError,
    HybridRetriever,
    KeywordRetriever,
    VectorRetriever,
)


class FakeVectorRetriever:
    def __init__(
        self,
        hits: Sequence[VectorSearchHit] = (),
        *,
        error: Exception | None = None,
    ) -> None:
        self.hits = list(hits)
        self.error = error
        self.calls: list[tuple[str, int, MetadataFilter | None]] = []

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[VectorSearchHit]:
        self.calls.append((query, k, metadata_filter))
        if self.error is not None:
            raise self.error
        return self.hits[:k]


class FakeKeywordRetriever:
    def __init__(
        self,
        hits: Sequence[KeywordSearchHit] = (),
        *,
        error: Exception | None = None,
    ) -> None:
        self.hits = list(hits)
        self.error = error
        self.calls: list[tuple[str, int, MetadataFilter | None]] = []

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[KeywordSearchHit]:
        self.calls.append((query, k, metadata_filter))
        if self.error is not None:
            raise self.error
        return self.hits[:k]


def test_config_has_immutable_defaults_and_supports_custom_depths():
    defaults = HybridRetrievalConfig()
    custom = HybridRetrievalConfig(dense_k=12, keyword_k=7)

    assert defaults.dense_k == 20
    assert defaults.keyword_k == 20
    assert custom.dense_k == 12
    assert custom.keyword_k == 7
    with pytest.raises(ValidationError, match="frozen"):
        custom.dense_k = 5


@pytest.mark.parametrize(
    "values",
    [
        {"dense_k": 0},
        {"dense_k": -1},
        {"keyword_k": 0},
        {"keyword_k": -1},
    ],
)
def test_config_rejects_nonpositive_branch_depths(values: dict[str, int]):
    with pytest.raises(ValidationError):
        HybridRetrievalConfig(**values)


def test_overlap_merges_each_chunk_once_and_preserves_all_signals():
    chunks = {key: _chunk(key) for key in "abcd"}
    dense = FakeVectorRetriever(
        [
            _dense_hit(chunks["a"], 1, 0.82),
            _dense_hit(chunks["b"], 2, 0.74),
            _dense_hit(chunks["c"], 3, 0.69),
        ]
    )
    keyword = FakeKeywordRetriever(
        [
            _keyword_hit(chunks["c"], 1, 4.81),
            _keyword_hit(chunks["d"], 2, 3.72),
            _keyword_hit(chunks["a"], 3, 2.95),
        ]
    )

    result = _hybrid(dense, keyword).search("query")

    assert [candidate.chunk.chunk_id for candidate in result.candidates] == list(
        "abcd"
    )
    by_id = {candidate.chunk.chunk_id: candidate for candidate in result.candidates}
    assert (
        by_id["a"].dense_rank,
        by_id["a"].dense_score,
        by_id["a"].keyword_rank,
        by_id["a"].keyword_score,
    ) == (1, 0.82, 3, 2.95)
    assert by_id["a"].sources == (
        RetrievalBranch.DENSE,
        RetrievalBranch.KEYWORD,
    )
    assert by_id["b"].sources == (RetrievalBranch.DENSE,)
    assert by_id["c"].sources == (
        RetrievalBranch.DENSE,
        RetrievalBranch.KEYWORD,
    )
    assert by_id["d"].sources == (RetrievalBranch.KEYWORD,)
    assert result.dense_retrieved_count == 3
    assert result.keyword_retrieved_count == 3
    assert result.unique_candidate_count == 4


def test_no_overlap_returns_all_dense_only_and_keyword_only_candidates():
    chunks = {key: _chunk(key) for key in "abcd"}
    result = _hybrid(
        FakeVectorRetriever(
            [_dense_hit(chunks["a"], 1, 0.9), _dense_hit(chunks["b"], 2, 0.8)]
        ),
        FakeKeywordRetriever(
            [
                _keyword_hit(chunks["c"], 1, 5.0),
                _keyword_hit(chunks["d"], 2, 4.0),
            ]
        ),
    ).search("query")

    assert result.unique_candidate_count == 4
    assert [candidate.sources for candidate in result.candidates] == [
        (RetrievalBranch.DENSE,),
        (RetrievalBranch.DENSE,),
        (RetrievalBranch.KEYWORD,),
        (RetrievalBranch.KEYWORD,),
    ]


def test_complete_overlap_produces_only_dual_signal_candidates():
    chunks = {key: _chunk(key) for key in "abc"}
    result = _hybrid(
        FakeVectorRetriever(
            [_dense_hit(chunks[key], rank, 1.0 / rank) for rank, key in enumerate("abc", 1)]
        ),
        FakeKeywordRetriever(
            [
                _keyword_hit(chunks[key], rank, 5.0 / rank)
                for rank, key in enumerate("abc", 1)
            ]
        ),
    ).search("query")

    assert result.unique_candidate_count == 3
    assert all(
        candidate.sources
        == (RetrievalBranch.DENSE, RetrievalBranch.KEYWORD)
        for candidate in result.candidates
    )


def test_disagreeing_rankings_are_preserved_without_resolution():
    chunks = {key: _chunk(key) for key in "abc"}
    result = _hybrid(
        FakeVectorRetriever(
            [_dense_hit(chunks[key], rank, 1.0 / rank) for rank, key in enumerate("abc", 1)]
        ),
        FakeKeywordRetriever(
            [
                _keyword_hit(chunks[key], rank, 10.0 / rank)
                for rank, key in enumerate("cab", 1)
            ]
        ),
    ).search("query")
    by_id = {candidate.chunk.chunk_id: candidate for candidate in result.candidates}

    assert (by_id["a"].dense_rank, by_id["a"].keyword_rank) == (1, 2)
    assert (by_id["b"].dense_rank, by_id["b"].keyword_rank) == (2, 3)
    assert (by_id["c"].dense_rank, by_id["c"].keyword_rank) == (3, 1)


def test_exact_query_filter_and_branch_depths_are_passed_to_both_retrievers():
    dense = FakeVectorRetriever()
    keyword = FakeKeywordRetriever()
    config = HybridRetrievalConfig(dense_k=12, keyword_k=7)
    metadata_filter = MetadataFilter(
        languages=("Python",),
        section_types=(RetrievalSectionType.PATCH,),
    )
    query = "  ConnectionResetError in retry_worker()  "

    result = _hybrid(dense, keyword, config=config).search(
        query,
        metadata_filter=metadata_filter,
    )

    assert result.query == query
    assert dense.calls == [(query, 12, metadata_filter)]
    assert keyword.calls == [(query, 7, metadata_filter)]
    assert dense.calls[0][2] is metadata_filter
    assert keyword.calls[0][2] is metadata_filter


def test_empty_filter_is_passed_through_unchanged():
    dense = FakeVectorRetriever()
    keyword = FakeKeywordRetriever()
    metadata_filter = MetadataFilter()

    _hybrid(dense, keyword).search("query", metadata_filter=metadata_filter)

    assert dense.calls[0][2] is metadata_filter
    assert keyword.calls[0][2] is metadata_filter


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_fails_before_either_branch_runs(query: str):
    dense = FakeVectorRetriever()
    keyword = FakeKeywordRetriever()

    with pytest.raises(HybridRetrievalError, match="must not be blank"):
        _hybrid(dense, keyword).search(query)

    assert dense.calls == []
    assert keyword.calls == []


def test_both_empty_returns_valid_empty_result_without_fallback():
    dense = FakeVectorRetriever()
    keyword = FakeKeywordRetriever()

    result = _hybrid(dense, keyword).search("query")

    assert result.dense_hits == ()
    assert result.keyword_hits == ()
    assert result.candidates == ()
    assert result.dense_retrieved_count == 0
    assert result.keyword_retrieved_count == 0
    assert result.unique_candidate_count == 0
    assert len(dense.calls) == 1
    assert len(keyword.calls) == 1


@pytest.mark.parametrize("active_branch", ["dense", "keyword"])
def test_one_empty_branch_preserves_other_branch_candidates(active_branch: str):
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    dense_hits = [
        _dense_hit(chunk, rank, 1.0 / rank)
        for rank, chunk in enumerate(chunks, 1)
    ]
    keyword_hits = [
        _keyword_hit(chunk, rank, 4.0 / rank)
        for rank, chunk in enumerate(chunks, 1)
    ]
    dense = FakeVectorRetriever(dense_hits if active_branch == "dense" else ())
    keyword = FakeKeywordRetriever(
        keyword_hits if active_branch == "keyword" else ()
    )

    result = _hybrid(dense, keyword).search("query")

    expected_source = (
        RetrievalBranch.DENSE
        if active_branch == "dense"
        else RetrievalBranch.KEYWORD
    )
    assert result.unique_candidate_count == 3
    assert all(candidate.sources == (expected_source,) for candidate in result.candidates)


def test_conflicting_cross_branch_chunk_data_fails_clearly():
    dense_chunk = _chunk("same", content="original content")
    keyword_chunk = _chunk(
        "same",
        content="different content",
        event_id="different-event",
    )

    with pytest.raises(HybridRetrievalError, match="conflicting data"):
        _hybrid(
            FakeVectorRetriever([_dense_hit(dense_chunk, 1, 0.9)]),
            FakeKeywordRetriever([_keyword_hit(keyword_chunk, 1, 5.2)]),
        ).search("query")


@pytest.mark.parametrize("branch", ["dense", "keyword"])
def test_duplicate_chunk_id_within_branch_fails_clearly(branch: str):
    chunk = _chunk("duplicate")
    dense = FakeVectorRetriever(
        [_dense_hit(chunk, 1, 0.9), _dense_hit(chunk, 2, 0.8)]
        if branch == "dense"
        else ()
    )
    keyword = FakeKeywordRetriever(
        [_keyword_hit(chunk, 1, 5.0), _keyword_hit(chunk, 2, 4.0)]
        if branch == "keyword"
        else ()
    )

    with pytest.raises(HybridRetrievalError, match="duplicate chunk ID"):
        _hybrid(dense, keyword).search("query")


@pytest.mark.parametrize("branch", ["dense", "keyword"])
def test_duplicate_rank_within_branch_fails_clearly(branch: str):
    chunk_a = _chunk("a")
    chunk_b = _chunk("b")
    dense = FakeVectorRetriever(
        [_dense_hit(chunk_a, 1, 0.9), _dense_hit(chunk_b, 1, 0.8)]
        if branch == "dense"
        else ()
    )
    keyword = FakeKeywordRetriever(
        [_keyword_hit(chunk_a, 1, 5.0), _keyword_hit(chunk_b, 1, 4.0)]
        if branch == "keyword"
        else ()
    )

    with pytest.raises(HybridRetrievalError, match="invalid or duplicate ranks"):
        _hybrid(dense, keyword).search("query")


def test_branch_operational_failure_is_not_silently_ignored():
    error = RuntimeError("dense unavailable")
    dense = FakeVectorRetriever(error=error)
    keyword = FakeKeywordRetriever()

    with pytest.raises(RuntimeError) as exc_info:
        _hybrid(dense, keyword).search("query")

    assert exc_info.value is error
    assert keyword.calls == []


def test_candidate_order_is_neutral_deterministic_and_not_branch_order():
    chunks = {key: _chunk(key) for key in "abcd"}
    hybrid = _hybrid(
        FakeVectorRetriever(
            [_dense_hit(chunks["d"], 1, 0.9), _dense_hit(chunks["b"], 2, 0.8)]
        ),
        FakeKeywordRetriever(
            [
                _keyword_hit(chunks["c"], 1, 5.0),
                _keyword_hit(chunks["a"], 2, 4.0),
            ]
        ),
    )

    first = hybrid.search("query")
    second = hybrid.search("query")

    assert first == second
    assert [candidate.chunk.chunk_id for candidate in first.candidates] == list(
        "abcd"
    )
    assert "hybrid_rank" not in type(first.candidates[0]).model_fields


def test_hybrid_retrieval_does_not_mutate_hits_chunks_or_filter():
    chunk = _chunk("a", languages=("Python",))
    dense_hit = _dense_hit(chunk, 1, 0.9)
    keyword_hit = _keyword_hit(chunk, 1, 5.0)
    metadata_filter = MetadataFilter(languages=("Python",))
    original = {
        "chunk": chunk.model_dump(),
        "dense": dense_hit.model_dump(),
        "keyword": keyword_hit.model_dump(),
        "filter": metadata_filter.model_dump(),
    }

    result = _hybrid(
        FakeVectorRetriever([dense_hit]),
        FakeKeywordRetriever([keyword_hit]),
    ).search("query", metadata_filter=metadata_filter)

    assert result.candidates[0].chunk is chunk
    assert chunk.model_dump() == original["chunk"]
    assert dense_hit.model_dump() == original["dense"]
    assert keyword_hit.model_dump() == original["keyword"]
    assert metadata_filter.model_dump() == original["filter"]


def test_technical_scenario_preserves_dense_keyword_and_dual_candidates():
    semantic_issue = _chunk("a", content="semantic database leak issue")
    discussion = _chunk("b", content="cleanup pull request discussion")
    stack_trace = _chunk("c", content="ConnectionResetError stack trace")
    result = _hybrid(
        FakeVectorRetriever(
            [
                _dense_hit(semantic_issue, 1, 0.88),
                _dense_hit(discussion, 2, 0.74),
            ]
        ),
        FakeKeywordRetriever(
            [
                _keyword_hit(stack_trace, 1, 6.2),
                _keyword_hit(semantic_issue, 2, 2.1),
            ]
        ),
    ).search("ConnectionResetError after retry worker crash")

    by_id = {candidate.chunk.chunk_id: candidate for candidate in result.candidates}
    assert set(by_id) == {"a", "b", "c"}
    assert by_id["a"].sources == (
        RetrievalBranch.DENSE,
        RetrievalBranch.KEYWORD,
    )
    assert by_id["b"].sources == (RetrievalBranch.DENSE,)
    assert by_id["c"].sources == (RetrievalBranch.KEYWORD,)


def test_hybrid_orchestrator_does_not_access_retrieval_internals():
    source = inspect.getsource(hybrid_retriever_module)

    for forbidden in (
        "FaissVectorIndex",
        "BM25Index",
        "EmbeddingBackend",
        "EngineeringTokenizer",
        ".embed(",
        ".tokenize(",
    ):
        assert forbidden not in source


def test_real_tiny_faiss_and_bm25_branches_integrate_without_model_download():
    semantic = _chunk("a", content="database connection resource leak")
    exact = _chunk("b", content="ConnectionResetError stack trace")
    overlap = _chunk("c", content="ConnectionResetError cleanup fix")
    chunks = [semantic, exact, overlap]
    vector_index = _vector_index(
        [
            (semantic, (1.0, 0.0)),
            (exact, (0.0, 1.0)),
            (overlap, (math.sqrt(0.5), math.sqrt(0.5))),
        ]
    )
    vector_retriever = VectorRetriever(
        index=vector_index,
        backend=_QueryBackend((1.0, 0.0)),
        chunks={chunk.chunk_id: chunk for chunk in chunks},
    )
    keyword_index = BM25Index()
    keyword_index.build(chunks)
    keyword_retriever = KeywordRetriever(
        index=keyword_index,
        chunks={chunk.chunk_id: chunk for chunk in chunks},
    )

    result = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        config=HybridRetrievalConfig(dense_k=2, keyword_k=2),
    ).search("ConnectionResetError")

    assert [hit.chunk.chunk_id for hit in result.dense_hits] == ["a", "c"]
    assert {hit.chunk.chunk_id for hit in result.keyword_hits} == {"b", "c"}
    assert [candidate.chunk.chunk_id for candidate in result.candidates] == [
        "a",
        "b",
        "c",
    ]
    by_id = {candidate.chunk.chunk_id: candidate for candidate in result.candidates}
    assert by_id["a"].sources == (RetrievalBranch.DENSE,)
    assert by_id["b"].sources == (RetrievalBranch.KEYWORD,)
    assert by_id["c"].sources == (
        RetrievalBranch.DENSE,
        RetrievalBranch.KEYWORD,
    )


class _QueryBackend:
    def __init__(self, vector: Sequence[float]) -> None:
        self.vector = vector

    @property
    def model_name(self) -> str:
        return "test/model"

    @property
    def normalized(self) -> bool:
        return True

    def embed(self, texts: Sequence[str]) -> list[Sequence[float]]:
        return [self.vector for _text in texts]


def _hybrid(
    dense: FakeVectorRetriever,
    keyword: FakeKeywordRetriever,
    *,
    config: HybridRetrievalConfig | None = None,
) -> HybridRetriever:
    return HybridRetriever(
        vector_retriever=dense,
        keyword_retriever=keyword,
        config=config,
    )


def _dense_hit(
    chunk: RetrievalChunk,
    rank: int,
    score: float,
) -> VectorSearchHit:
    return VectorSearchHit(chunk=chunk, rank=rank, score=score)


def _keyword_hit(
    chunk: RetrievalChunk,
    rank: int,
    score: float,
) -> KeywordSearchHit:
    return KeywordSearchHit(chunk=chunk, rank=rank, score=score)


def _vector_index(
    values: Sequence[tuple[RetrievalChunk, tuple[float, ...]]],
) -> FaissVectorIndex:
    index = FaissVectorIndex()
    index.build(
        [
            ChunkEmbedding(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                event_id=chunk.event_id,
                repository=chunk.repository,
                section_id=chunk.section_id,
                section_type=chunk.section_type,
                model_name="test/model",
                dimension=len(vector),
                normalized=True,
                source_text_sha256=hashlib.sha256(
                    chunk.text.encode("utf-8")
                ).hexdigest(),
                vector=vector,
            )
            for chunk, vector in values
        ]
    )
    return index


def _chunk(
    chunk_id: str,
    *,
    content: str = "connection cleanup",
    event_id: str | None = None,
    languages: Sequence[str] = (),
) -> RetrievalChunk:
    repository = GitHubRepository(owner="owner", name="repo")
    active_event_id = event_id or f"event-{chunk_id}"
    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        event_id=active_event_id,
        repository=repository,
        section_id=f"section-{chunk_id}",
        section_type=RetrievalSectionType.PATCH,
        chunk_index=0,
        content=content,
        text=f"Repository: owner/repo\n{content}",
        metadata=EventMetadata(
            event_id=active_event_id,
            repository=repository,
            languages=tuple(languages),
        ),
    )
