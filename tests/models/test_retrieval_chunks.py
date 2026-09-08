import pytest
from pydantic import ValidationError

from reporecall.github import GitHubRepository
from reporecall.models import (
    ArtifactReference,
    ArtifactType,
    EventMetadata,
    RetrievalChunk,
    RetrievalSectionType,
)


def test_retrieval_chunk_is_immutable_and_preserves_structured_provenance():
    chunk = RetrievalChunk(**_chunk_values())

    assert chunk.repository == _repository()
    assert chunk.artifact == _artifact()
    assert chunk.metadata == _metadata()
    with pytest.raises(ValidationError, match="frozen"):
        chunk.content = "changed"


@pytest.mark.parametrize(
    "field_name",
    ["chunk_id", "document_id", "event_id", "section_id", "content", "text"],
)
def test_retrieval_chunk_rejects_empty_required_text(field_name: str):
    values = _chunk_values()
    values[field_name] = ""

    with pytest.raises(ValidationError):
        RetrievalChunk(**values)


def test_retrieval_chunk_rejects_negative_index():
    values = _chunk_values()
    values["chunk_index"] = -1

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        RetrievalChunk(**values)


def test_retrieval_chunk_rejects_mismatched_metadata_identity():
    values = _chunk_values()
    values["metadata"] = EventMetadata(event_id="other", repository=_repository())
    with pytest.raises(ValidationError, match="event ID"):
        RetrievalChunk(**values)

    values = _chunk_values()
    values["metadata"] = EventMetadata(
        event_id="event",
        repository=GitHubRepository(owner="other", name="repo"),
    )
    with pytest.raises(ValidationError, match="repository"):
        RetrievalChunk(**values)


def test_retrieval_chunk_rejects_artifact_from_another_repository():
    values = _chunk_values()
    values["artifact"] = ArtifactReference(
        artifact_type=ArtifactType.ISSUE,
        repository=GitHubRepository(owner="other", name="repo"),
        identifier="10",
    )

    with pytest.raises(ValidationError, match="artifact"):
        RetrievalChunk(**values)


def _chunk_values() -> dict[str, object]:
    return {
        "chunk_id": "document__issue-10__0000__abc123",
        "document_id": "document",
        "event_id": "event",
        "repository": _repository(),
        "section_id": "issue-10",
        "section_type": RetrievalSectionType.ISSUE,
        "chunk_index": 0,
        "content": "Issue content",
        "text": "Repository: owner/repo\n\nIssue content",
        "artifact": _artifact(),
        "metadata": _metadata(),
    }


def _artifact() -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.ISSUE,
        repository=_repository(),
        identifier="10",
    )


def _metadata() -> EventMetadata:
    return EventMetadata(event_id="event", repository=_repository())


def _repository() -> GitHubRepository:
    return GitHubRepository(owner="owner", name="repo")
