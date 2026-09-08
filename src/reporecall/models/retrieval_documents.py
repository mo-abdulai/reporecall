from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reporecall.github.models import GitHubRepository
from reporecall.models.event_metadata import EventMetadata
from reporecall.models.relationships import ArtifactReference


class RetrievalSectionType(str, Enum):
    """Domain boundaries preserved in a retrieval document before chunking."""

    OVERVIEW = "overview"
    METADATA = "metadata"
    ISSUE = "issue"
    ISSUE_COMMENT = "issue_comment"
    PULL_REQUEST = "pull_request"
    PULL_REQUEST_COMMENT = "pull_request_comment"
    REVIEW = "review"
    REVIEW_COMMENT = "review_comment"
    COMMIT = "commit"
    CHANGED_FILE = "changed_file"
    PATCH = "patch"
    RELATIONSHIP = "relationship"
    CONTEXTUAL_RELATIONSHIP = "contextual_relationship"


class RetrievalDocumentSection(BaseModel):
    """One stable, domain-specific section of a retrieval document."""

    section_id: str = Field(min_length=1)
    section_type: RetrievalSectionType
    heading: str = Field(min_length=1)
    content: str = Field(min_length=1)
    artifact: ArtifactReference | None = None

    model_config = ConfigDict(frozen=True)


class RetrievalSource(BaseModel):
    """Lightweight source identity retained for future citation support."""

    artifact: ArtifactReference
    url: str | None = None
    label: str = Field(min_length=1)

    model_config = ConfigDict(frozen=True)


class RetrievalDocument(BaseModel):
    """Immutable canonical retrieval representation of one engineering event."""

    document_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    repository: GitHubRepository
    title: str = Field(min_length=1)
    sections: tuple[RetrievalDocumentSection, ...]
    text: str = Field(min_length=1)
    metadata: EventMetadata
    sources: tuple[RetrievalSource, ...] = ()

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_document_identity(self) -> "RetrievalDocument":
        if self.metadata.event_id != self.event_id:
            raise ValueError("Retrieval document metadata must match its event ID.")
        if self.metadata.repository != self.repository:
            raise ValueError("Retrieval document metadata must match its repository.")

        section_ids = [section.section_id for section in self.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("Retrieval document section IDs must be unique.")

        source_identities = [(source.artifact, source.url) for source in self.sources]
        if len(source_identities) != len(set(source_identities)):
            raise ValueError("Retrieval document sources must be unique.")
        return self
