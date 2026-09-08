import pytest
from pydantic import ValidationError

from reporecall.github import GitHubRepository
from reporecall.models import (
    ArtifactReference,
    ArtifactType,
    EventMetadata,
    RetrievalDocument,
    RetrievalDocumentSection,
    RetrievalSectionType,
    RetrievalSource,
)


def test_retrieval_document_is_immutable_and_normalizes_collections_to_tuples():
    section = _section()
    source = _source()
    document = RetrievalDocument(
        document_id="event__retrieval",
        event_id="event",
        repository=_repository(),
        title="Issue #10 - Example",
        sections=[section],
        text="Rendered text\n",
        metadata=_metadata(),
        sources=[source],
    )

    assert document.sections == (section,)
    assert document.sources == (source,)
    with pytest.raises(ValidationError, match="frozen"):
        document.title = "Changed"
    with pytest.raises(ValidationError, match="frozen"):
        document.sections[0].heading = "Changed"


def test_retrieval_document_rejects_metadata_for_another_event_or_repository():
    with pytest.raises(ValidationError, match="event ID"):
        RetrievalDocument(
            document_id="event__retrieval",
            event_id="event",
            repository=_repository(),
            title="Example",
            sections=[_section()],
            text="Rendered text\n",
            metadata=EventMetadata(event_id="other", repository=_repository()),
        )

    with pytest.raises(ValidationError, match="repository"):
        RetrievalDocument(
            document_id="event__retrieval",
            event_id="event",
            repository=_repository(),
            title="Example",
            sections=[_section()],
            text="Rendered text\n",
            metadata=EventMetadata(
                event_id="event",
                repository=GitHubRepository(owner="other", name="repo"),
            ),
        )


def test_retrieval_document_requires_unique_section_and_source_identities():
    with pytest.raises(ValidationError, match="section IDs"):
        RetrievalDocument(
            document_id="event__retrieval",
            event_id="event",
            repository=_repository(),
            title="Example",
            sections=[_section(), _section()],
            text="Rendered text\n",
            metadata=_metadata(),
        )

    with pytest.raises(ValidationError, match="sources"):
        RetrievalDocument(
            document_id="event__retrieval",
            event_id="event",
            repository=_repository(),
            title="Example",
            sections=[_section()],
            text="Rendered text\n",
            metadata=_metadata(),
            sources=[_source(), _source()],
        )


def _section() -> RetrievalDocumentSection:
    return RetrievalDocumentSection(
        section_id="overview",
        section_type=RetrievalSectionType.OVERVIEW,
        heading="OVERVIEW",
        content="Repository: owner/repo",
    )


def _source() -> RetrievalSource:
    return RetrievalSource(
        artifact=ArtifactReference(
            artifact_type=ArtifactType.ISSUE,
            repository=_repository(),
            identifier="10",
        ),
        url="https://github.com/owner/repo/issues/10",
        label="Issue #10",
    )


def _metadata() -> EventMetadata:
    return EventMetadata(event_id="event", repository=_repository())


def _repository() -> GitHubRepository:
    return GitHubRepository(owner="owner", name="repo")
