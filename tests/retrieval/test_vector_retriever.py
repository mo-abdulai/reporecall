import hashlib
from collections.abc import Sequence

import pytest

from reporecall.github import GitHubRepository
from reporecall.models import (
    ArtifactReference,
    ArtifactType,
    ChunkEmbedding,
    EventMetadata,
    RetrievalChunk,
    RetrievalSectionType,
)
from reporecall.retrieval import (
    FaissVectorIndex,
    RetrievalError,
    VectorIndexCompatibilityError,
    VectorIndexError,
    VectorRetriever,
)


class FakeQueryBackend:
    def __init__(
        self,
        vector: Sequence[float] = (1.0, 0.0),
        *,
        model_name: str = "test/model",
        normalized: bool = True,
        output_count: int = 1,
    ) -> None:
        self._vector = vector
        self._model_name = model_name
        self._normalized = normalized
        self.output_count = output_count
        self.calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def normalized(self) -> bool:
        return self._normalized

    def embed(self, texts: Sequence[str]) -> list[Sequence[float]]:
        self.calls.append(list(texts))
        return [self._vector for _ in range(self.output_count)]


def test_retriever_embeds_raw_query_and_returns_closest_complete_chunk():
    issue = _chunk("chunk-issue", "Database connection leak", artifact=True)
    docs = _chunk("chunk-docs", "Update installation documentation")
    index = _index_for(
        [(issue, (1.0, 0.0)), (docs, (0.0, 1.0))]
    )
    backend = FakeQueryBackend((1.0, 0.0))
    retriever = VectorRetriever(
        index=index,
        backend=backend,
        chunks={issue.chunk_id: issue, docs.chunk_id: docs},
    )
    query = "database connection leak"

    hits = retriever.search(query, k=2)

    assert backend.calls == [[query]]
    assert [hit.chunk.chunk_id for hit in hits] == ["chunk-issue", "chunk-docs"]
    assert [hit.rank for hit in hits] == [1, 2]
    assert [hit.score for hit in hits] == pytest.approx([1.0, 0.0])
    assert hits[0].chunk == issue
    assert hits[0].chunk.document_id == issue.document_id
    assert hits[0].chunk.event_id == issue.event_id
    assert hits[0].chunk.repository == issue.repository
    assert hits[0].chunk.section_id == issue.section_id
    assert hits[0].chunk.section_type == issue.section_type
    assert hits[0].chunk.artifact == issue.artifact
    assert hits[0].chunk.metadata == issue.metadata


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_retriever_rejects_blank_query_without_embedding(query: str):
    chunk = _chunk("chunk-a", "text")
    backend = FakeQueryBackend()
    retriever = VectorRetriever(
        index=_index_for([(chunk, (1.0, 0.0))]),
        backend=backend,
        chunks={chunk.chunk_id: chunk},
    )

    with pytest.raises(RetrievalError, match="must not be blank"):
        retriever.search(query)

    assert backend.calls == []


@pytest.mark.parametrize("k", [0, -1])
def test_retriever_rejects_nonpositive_k_without_embedding(k: int):
    chunk = _chunk("chunk-a", "text")
    backend = FakeQueryBackend()
    retriever = VectorRetriever(
        index=_index_for([(chunk, (1.0, 0.0))]),
        backend=backend,
        chunks={chunk.chunk_id: chunk},
    )

    with pytest.raises(RetrievalError, match="greater than zero"):
        retriever.search("query", k=k)

    assert backend.calls == []


def test_empty_index_returns_without_embedding_query():
    backend = FakeQueryBackend()
    retriever = VectorRetriever(
        index=FaissVectorIndex(),
        backend=backend,
        chunks={},
    )

    assert retriever.search("database connection leak", k=5) == []
    assert backend.calls == []


def test_retriever_rejects_query_model_mismatch_before_embedding():
    chunk = _chunk("chunk-a", "text")
    backend = FakeQueryBackend(model_name="different/model")
    retriever = VectorRetriever(
        index=_index_for([(chunk, (1.0, 0.0))]),
        backend=backend,
        chunks={chunk.chunk_id: chunk},
    )

    with pytest.raises(VectorIndexCompatibilityError, match="model does not match"):
        retriever.search("query")

    assert backend.calls == []


def test_retriever_rejects_non_normalized_query_backend_before_embedding():
    chunk = _chunk("chunk-a", "text")
    backend = FakeQueryBackend(normalized=False)
    retriever = VectorRetriever(
        index=_index_for([(chunk, (1.0, 0.0))]),
        backend=backend,
        chunks={chunk.chunk_id: chunk},
    )

    with pytest.raises(VectorIndexCompatibilityError, match="both be normalized"):
        retriever.search("query")

    assert backend.calls == []


def test_retriever_rejects_query_vector_dimension_mismatch():
    chunk = _chunk("chunk-a", "text")
    retriever = VectorRetriever(
        index=_index_for([(chunk, (1.0, 0.0))]),
        backend=FakeQueryBackend((1.0, 0.0, 0.0)),
        chunks={chunk.chunk_id: chunk},
    )

    with pytest.raises(VectorIndexCompatibilityError, match="dimension"):
        retriever.search("query")


@pytest.mark.parametrize("output_count", [0, 2])
def test_retriever_requires_exactly_one_query_vector(output_count: int):
    chunk = _chunk("chunk-a", "text")
    retriever = VectorRetriever(
        index=_index_for([(chunk, (1.0, 0.0))]),
        backend=FakeQueryBackend(output_count=output_count),
        chunks={chunk.chunk_id: chunk},
    )

    with pytest.raises(RetrievalError, match="exactly one vector"):
        retriever.search("query")


def test_missing_chunk_mapping_fails_clearly():
    chunk = _chunk("chunk-a", "text")
    retriever = VectorRetriever(
        index=_index_for([(chunk, (1.0, 0.0))]),
        backend=FakeQueryBackend(),
        chunks={},
    )

    with pytest.raises(VectorIndexError, match="missing from the chunk lookup"):
        retriever.search("query")


def test_stale_chunk_text_fails_source_hash_validation():
    original = _chunk("chunk-a", "original text")
    stale = original.model_copy(update={"text": "changed retrieval text"})
    retriever = VectorRetriever(
        index=_index_for([(original, (1.0, 0.0))]),
        backend=FakeQueryBackend(),
        chunks={stale.chunk_id: stale},
    )

    with pytest.raises(VectorIndexError, match="stale source text"):
        retriever.search("query")


def test_chunk_lookup_key_must_match_chunk_identity():
    chunk = _chunk("chunk-a", "text")

    with pytest.raises(VectorIndexError, match="lookup keys"):
        VectorRetriever(
            index=_index_for([(chunk, (1.0, 0.0))]),
            backend=FakeQueryBackend(),
            chunks={"wrong-key": chunk},
        )


def test_retrieval_is_stable_and_does_not_mutate_chunks():
    first_chunk = _chunk("chunk-a", "first")
    second_chunk = _chunk("chunk-b", "second")
    chunks = {first_chunk.chunk_id: first_chunk, second_chunk.chunk_id: second_chunk}
    original = {key: chunk.model_dump() for key, chunk in chunks.items()}
    retriever = VectorRetriever(
        index=_index_for(
            [(first_chunk, (1.0, 0.0)), (second_chunk, (0.0, 1.0))]
        ),
        backend=FakeQueryBackend((1.0, 0.0)),
        chunks=chunks,
    )

    first = retriever.search("query", k=2)
    second = retriever.search("query", k=2)

    assert first == second
    assert {key: chunk.model_dump() for key, chunk in chunks.items()} == original


def _index_for(
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


def _chunk(chunk_id: str, content: str, *, artifact: bool = False) -> RetrievalChunk:
    repository = GitHubRepository(owner="owner", name="repo")
    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id="document-1",
        event_id="event-1",
        repository=repository,
        section_id=f"section-{chunk_id}",
        section_type=RetrievalSectionType.ISSUE,
        chunk_index=0,
        content=content,
        text=f"Repository: owner/repo\nSection: Issue\n\n{content}",
        artifact=(
            ArtifactReference(
                artifact_type=ArtifactType.ISSUE,
                repository=repository,
                identifier="10",
            )
            if artifact
            else None
        ),
        metadata=EventMetadata(
            event_id="event-1",
            repository=repository,
            issue_numbers=(10,),
        ),
    )
