import hashlib

import pytest

from reporecall.embeddings import (
    EmbeddingError,
    EmbeddingGenerator,
    EmbeddingOutputError,
)
from reporecall.github import GitHubRepository
from reporecall.models import EventMetadata, RetrievalChunk, RetrievalSectionType


class DeterministicBackend:
    def __init__(
        self,
        *,
        model_name: str = "test/model",
        normalized: bool = True,
    ) -> None:
        self._model_name = model_name
        self._normalized = normalized
        self.calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def normalized(self) -> bool:
        return self._normalized

    def embed(self, texts):
        self.calls.append(list(texts))
        return [
            [float(index + 1), float(len(text)), float(sum(text.encode("utf-8")))]
            for index, text in enumerate(texts)
        ]


class StaticBackend(DeterministicBackend):
    def __init__(self, vectors) -> None:
        super().__init__()
        self.vectors = vectors

    def embed(self, texts):
        self.calls.append(list(texts))
        return self.vectors


def test_generator_embeds_exact_chunk_text_and_preserves_all_identities():
    technical_text = (
        "Repository: owner/repo\n"
        "Section: Patch\n\n"
        "ConnectionError src/database/session.py abc123def\n"
        "SELECT * FROM sessions\n"
        "retry_worker()\n"
        "@@ -20,4 +20,8 @@"
    )
    chunk = _chunk("chunk-1", technical_text, section_type=RetrievalSectionType.PATCH)
    backend = DeterministicBackend()

    embedding = EmbeddingGenerator(backend).generate([chunk])[0]

    assert backend.calls == [[technical_text]]
    assert embedding.chunk_id == chunk.chunk_id
    assert embedding.document_id == chunk.document_id
    assert embedding.event_id == chunk.event_id
    assert embedding.repository == chunk.repository
    assert embedding.section_id == chunk.section_id
    assert embedding.section_type is RetrievalSectionType.PATCH
    assert embedding.model_name == "test/model"
    assert embedding.dimension == 3
    assert embedding.normalized is True
    assert embedding.source_text_sha256 == hashlib.sha256(
        technical_text.encode("utf-8")
    ).hexdigest()
    assert embedding.vector == (
        1.0,
        float(len(technical_text)),
        float(sum(technical_text.encode("utf-8"))),
    )


def test_generator_batches_once_and_preserves_input_order():
    chunks = [
        _chunk("chunk-C", "text C"),
        _chunk("chunk-A", "text A"),
        _chunk("chunk-B", "text B"),
    ]
    backend = DeterministicBackend()

    embeddings = EmbeddingGenerator(backend).generate(chunks)

    assert backend.calls == [["text C", "text A", "text B"]]
    assert [embedding.chunk_id for embedding in embeddings] == [
        "chunk-C",
        "chunk-A",
        "chunk-B",
    ]
    assert all(embedding.dimension == 3 for embedding in embeddings)


def test_same_content_with_different_context_embeds_distinct_chunk_text():
    source_content = "This should be released in a finally block."
    issue_text = f"Repository: owner/repo\nSection: Issue\n\n{source_content}"
    review_text = f"Repository: owner/repo\nSection: Review Comment\n\n{source_content}"
    chunks = [
        _chunk("issue-chunk", issue_text, content=source_content),
        _chunk(
            "review-chunk",
            review_text,
            content=source_content,
            section_type=RetrievalSectionType.REVIEW_COMMENT,
        ),
    ]
    backend = DeterministicBackend()

    EmbeddingGenerator(backend).generate(chunks)

    assert backend.calls == [[issue_text, review_text]]
    assert chunks[0].content == chunks[1].content
    assert chunks[0].text != chunks[1].text


def test_source_text_hash_changes_when_one_character_changes():
    backend = DeterministicBackend()
    generator = EmbeddingGenerator(backend)

    first = generator.generate([_chunk("chunk-1", "exact text")])[0]
    second = generator.generate([_chunk("chunk-2", "exact Text")])[0]

    assert first.source_text_sha256 == hashlib.sha256(b"exact text").hexdigest()
    assert second.source_text_sha256 == hashlib.sha256(b"exact Text").hexdigest()
    assert first.source_text_sha256 != second.source_text_sha256


def test_non_normalized_backend_marks_embedding_records_accurately():
    embedding = EmbeddingGenerator(
        DeterministicBackend(normalized=False)
    ).generate([_chunk("chunk-1", "text")])[0]

    assert embedding.normalized is False


def test_identical_duplicate_chunk_ids_are_embedded_once_in_first_order():
    first = _chunk("chunk-1", "first text")
    second = _chunk("chunk-2", "second text")
    backend = DeterministicBackend()

    embeddings = EmbeddingGenerator(backend).generate([first, second, first])

    assert backend.calls == [["first text", "second text"]]
    assert [embedding.chunk_id for embedding in embeddings] == ["chunk-1", "chunk-2"]


def test_conflicting_duplicate_chunk_id_fails_before_backend_call():
    first = _chunk("chunk-1", "first text")
    conflicting = first.model_copy(update={"text": "different text"})
    backend = DeterministicBackend()

    with pytest.raises(EmbeddingOutputError, match="conflicting text"):
        EmbeddingGenerator(backend).generate([first, conflicting])

    assert backend.calls == []


def test_empty_input_returns_without_invoking_backend():
    backend = DeterministicBackend()

    assert EmbeddingGenerator(backend).generate([]) == []
    assert backend.calls == []


@pytest.mark.parametrize(
    "vectors",
    [
        [[1.0, 2.0]],
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
    ],
)
def test_output_count_mismatch_fails_without_partial_results(vectors):
    chunks = [_chunk(f"chunk-{index}", f"text {index}") for index in range(3)]

    with pytest.raises(EmbeddingOutputError, match="different number"):
        EmbeddingGenerator(StaticBackend(vectors)).generate(chunks)


def test_dimension_mismatch_is_rejected_for_whole_batch():
    backend = StaticBackend([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]])

    with pytest.raises(EmbeddingOutputError, match="same dimension"):
        EmbeddingGenerator(backend).generate(
            [_chunk("chunk-1", "first"), _chunk("chunk-2", "second")]
        )


def test_empty_vector_is_rejected():
    with pytest.raises(EmbeddingOutputError, match="must not be empty"):
        EmbeddingGenerator(StaticBackend([[]])).generate([_chunk("chunk-1", "text")])


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_backend_vectors_are_rejected(invalid_value: float):
    with pytest.raises(EmbeddingOutputError, match="finite"):
        EmbeddingGenerator(StaticBackend([[1.0, invalid_value]])).generate(
            [_chunk("chunk-1", "text")]
        )


def test_backend_failure_is_wrapped_and_does_not_return_partial_results():
    class FailingBackend(DeterministicBackend):
        def embed(self, texts):
            self.calls.append(list(texts))
            raise RuntimeError("inference failed")

    with pytest.raises(EmbeddingError, match="backend failed") as exc_info:
        EmbeddingGenerator(FailingBackend()).generate([_chunk("chunk-1", "text")])

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_generation_is_deterministic_and_does_not_mutate_chunks():
    chunks = [_chunk("chunk-1", "first"), _chunk("chunk-2", "second")]
    original = [chunk.model_dump() for chunk in chunks]
    generator = EmbeddingGenerator(DeterministicBackend())

    first = generator.generate(chunks)
    second = generator.generate(chunks)

    assert first == second
    assert [chunk.model_dump() for chunk in chunks] == original


def _chunk(
    chunk_id: str,
    text: str,
    *,
    content: str | None = None,
    section_type: RetrievalSectionType = RetrievalSectionType.ISSUE,
) -> RetrievalChunk:
    repository = GitHubRepository(owner="owner", name="repo")
    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id="document-1",
        event_id="event-1",
        repository=repository,
        section_id=f"section-{chunk_id}",
        section_type=section_type,
        chunk_index=0,
        content=content or text,
        text=text,
        artifact=None,
        metadata=EventMetadata(event_id="event-1", repository=repository),
    )
