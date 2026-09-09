import inspect
from collections.abc import Sequence

import pytest

import reporecall.retrieval.bm25_index as bm25_index_module
import reporecall.retrieval.keyword_retriever as keyword_retriever_module
from reporecall.github import GitHubRepository
from reporecall.models import (
    EventMetadata,
    MetadataFilter,
    RetrievalChunk,
    RetrievalSectionType,
)
from reporecall.retrieval import (
    BM25Index,
    BM25IndexError,
    EngineeringTokenizer,
    KeywordRetrievalError,
    KeywordRetriever,
)


@pytest.mark.parametrize(
    ("query", "expected_chunk_id"),
    [
        ("ConnectionResetError", "connection-error"),
        ("src/database/session.py", "database-path"),
        ("retry_worker", "retry-worker"),
        ("HTTP_429", "rate-limit"),
        ("9a0b27473cfb40769d1b06ac827241fb89025def", "commit-sha"),
    ],
)
def test_exact_technical_queries_rank_the_matching_chunk_first(
    query: str,
    expected_chunk_id: str,
):
    chunks = [
        _chunk(
            "connection-error",
            "ConnectionResetError occurred while receiving data",
        ),
        _chunk("database-path", "Changed src/database/session.py"),
        _chunk("retry-worker", "The retry_worker() function performs cleanup"),
        _chunk("rate-limit", "Handle HTTP_429 from the upstream API"),
        _chunk(
            "commit-sha",
            "Introduced by 9a0b27473cfb40769d1b06ac827241fb89025def",
        ),
        _chunk("unrelated", "Update frontend spacing and colors"),
    ]
    retriever = _retriever(chunks)

    hits = retriever.search(query, k=5)

    assert hits[0].chunk.chunk_id == expected_chunk_id
    assert hits[0].rank == 1
    assert hits[0].score > 0


def test_language_metadata_filter_restricts_eligible_chunks_before_ranking():
    chunks = [
        _chunk("python", "authentication authentication", languages=("Python",)),
        _chunk(
            "typescript",
            "authentication authentication authentication",
            languages=("TypeScript",),
        ),
    ]

    hits = _retriever(chunks).search(
        "authentication",
        k=2,
        metadata_filter=MetadataFilter(languages=("Python",)),
    )

    assert [hit.chunk.chunk_id for hit in hits] == ["python"]
    assert hits[0].rank == 1


def test_section_metadata_filter_restricts_eligible_chunks():
    chunks = [
        _chunk("issue", "connection cleanup", section=RetrievalSectionType.ISSUE),
        _chunk("patch", "connection cleanup", section=RetrievalSectionType.PATCH),
        _chunk(
            "review",
            "connection cleanup",
            section=RetrievalSectionType.REVIEW_COMMENT,
        ),
    ]

    hits = _retriever(chunks).search(
        "connection cleanup",
        k=3,
        metadata_filter=MetadataFilter(
            section_types=(RetrievalSectionType.PATCH,)
        ),
    )

    assert [hit.chunk.chunk_id for hit in hits] == ["patch"]


def test_filtered_top_k_uses_all_eligible_chunks_not_global_top_k():
    chunks = [
        _chunk("a", "needle needle needle needle needle", labels=("feature",)),
        _chunk("b", "needle needle needle needle", labels=("feature",)),
        _chunk("c", "needle needle", labels=("bug",)),
        _chunk("d", "needle", labels=("bug",)),
    ]

    hits = _retriever(chunks).search(
        "needle",
        k=2,
        metadata_filter=MetadataFilter(labels=("bug",)),
    )

    assert [hit.chunk.chunk_id for hit in hits] == ["c", "d"]
    assert [hit.rank for hit in hits] == [1, 2]


def test_empty_metadata_filter_matches_unfiltered_search():
    chunks = [_chunk("a", "database pool"), _chunk("b", "database worker")]
    retriever = _retriever(chunks)

    unfiltered = retriever.search("database", k=2)
    filtered = retriever.search("database", k=2, metadata_filter=MetadataFilter())

    assert filtered == unfiltered


def test_no_match_filter_returns_before_query_tokenization():
    tokenizer = _RecordingTokenizer()
    chunk = _chunk("python", "authentication", languages=("Python",))
    index = BM25Index(tokenizer=tokenizer)
    index.build([chunk])
    tokenizer.inputs.clear()
    retriever = KeywordRetriever(index=index, chunks={chunk.chunk_id: chunk})

    hits = retriever.search(
        "authentication",
        metadata_filter=MetadataFilter(languages=("TypeScript",)),
    )

    assert hits == []
    assert tokenizer.inputs == []


def test_raw_query_reaches_tokenizer_unchanged_without_filter_text():
    tokenizer = _RecordingTokenizer()
    chunk = _chunk("python", "ConnectionResetError retry_worker", languages=("Python",))
    index = BM25Index(tokenizer=tokenizer)
    index.build([chunk])
    tokenizer.inputs.clear()
    retriever = KeywordRetriever(index=index, chunks={chunk.chunk_id: chunk})
    raw_query = "ConnectionResetError retry_worker"

    retriever.search(
        raw_query,
        metadata_filter=MetadataFilter(languages=("Python",)),
    )

    assert tokenizer.inputs == [raw_query]


def test_punctuation_only_query_returns_no_matches():
    retriever = _retriever([_chunk("a", "database")])

    assert retriever.search("... () !!!") == []


@pytest.mark.parametrize("query", ["", "   "])
def test_blank_query_is_rejected_before_tokenization(query: str):
    tokenizer = _RecordingTokenizer()
    index = BM25Index(tokenizer=tokenizer)

    with pytest.raises(KeywordRetrievalError, match="must not be blank"):
        KeywordRetriever(index=index, chunks={}).search(query)

    assert tokenizer.inputs == []


@pytest.mark.parametrize("k", [0, -1])
def test_nonpositive_k_is_rejected(k: int):
    retriever = _retriever([_chunk("a", "database")])

    with pytest.raises(KeywordRetrievalError, match="greater than zero"):
        retriever.search("database", k=k)


def test_empty_index_returns_no_matches():
    assert KeywordRetriever(index=BM25Index(), chunks={}).search("database") == []


def test_hit_preserves_original_chunk_and_does_not_mutate_inputs():
    chunk = _chunk(
        "a",
        "ConnectionResetError in retry_worker()",
        languages=("Python",),
        labels=("Bug",),
    )
    metadata_filter = MetadataFilter(languages=("python",), labels=("bug",))
    original_chunk = chunk.model_dump()
    original_filter = metadata_filter.model_dump()

    hits = _retriever([chunk]).search(
        "ConnectionResetError",
        metadata_filter=metadata_filter,
    )

    assert hits[0].chunk is chunk
    assert hits[0].chunk.model_dump() == original_chunk
    assert metadata_filter.model_dump() == original_filter


def test_chunk_lookup_keys_must_match_chunk_identity():
    chunk = _chunk("actual", "database")

    with pytest.raises(BM25IndexError, match="lookup keys must match"):
        KeywordRetriever(index=BM25Index(), chunks={"wrong": chunk})


def test_all_indexed_chunks_must_exist_in_lookup():
    chunk = _chunk("indexed", "database")
    index = BM25Index()
    index.build([chunk])

    with pytest.raises(BM25IndexError, match="missing from the chunk lookup"):
        KeywordRetriever(index=index, chunks={}).search("database")


def test_retriever_rejects_stale_chunk_text():
    indexed = _chunk("same", "original database text")
    stale = _chunk("same", "changed database text")
    index = BM25Index()
    index.build([indexed])

    with pytest.raises(BM25IndexError, match="stale source text"):
        KeywordRetriever(index=index, chunks={"same": stale}).search("database")


def test_filter_cannot_select_a_chunk_missing_from_the_index():
    indexed = _chunk("indexed", "database", languages=("Python",))
    unindexed = _chunk("unindexed", "database", languages=("TypeScript",))
    index = BM25Index()
    index.build([indexed])
    retriever = KeywordRetriever(
        index=index,
        chunks={indexed.chunk_id: indexed, unindexed.chunk_id: unindexed},
    )

    with pytest.raises(BM25IndexError, match="missing from the BM25 index"):
        retriever.search(
            "database",
            metadata_filter=MetadataFilter(languages=("TypeScript",)),
        )


def test_keyword_modules_do_not_depend_on_dense_retrieval_or_embeddings():
    source = inspect.getsource(bm25_index_module) + inspect.getsource(
        keyword_retriever_module
    )

    for forbidden in (
        "EmbeddingBackend",
        "SentenceTransformer",
        "ChunkEmbedding",
        "FaissVectorIndex",
        "faiss_index",
        "reporecall.embeddings",
    ):
        assert forbidden not in source


class _RecordingTokenizer(EngineeringTokenizer):
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def tokenize(self, text: str) -> tuple[str, ...]:
        self.inputs.append(text)
        return super().tokenize(text)


def _retriever(chunks: Sequence[RetrievalChunk]) -> KeywordRetriever:
    index = BM25Index()
    index.build(chunks)
    return KeywordRetriever(
        index=index,
        chunks={chunk.chunk_id: chunk for chunk in chunks},
    )


def _chunk(
    chunk_id: str,
    content: str,
    *,
    languages: Sequence[str] = (),
    labels: Sequence[str] = (),
    section: RetrievalSectionType = RetrievalSectionType.ISSUE,
) -> RetrievalChunk:
    repository = GitHubRepository(owner="owner", name="repo")
    event_id = f"event-{chunk_id}"
    metadata = EventMetadata(
        event_id=event_id,
        repository=repository,
        languages=tuple(languages),
        labels=tuple(labels),
    )
    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        event_id=event_id,
        repository=repository,
        section_id=f"section-{chunk_id}",
        section_type=section,
        chunk_index=0,
        content=content,
        text=f"Repository: owner/repo\nSection: {section.value}\n{content}",
        metadata=metadata,
    )
