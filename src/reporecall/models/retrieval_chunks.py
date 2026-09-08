from pydantic import BaseModel, ConfigDict, Field, model_validator

from reporecall.github.models import GitHubRepository
from reporecall.models.event_metadata import EventMetadata
from reporecall.models.relationships import ArtifactReference
from reporecall.models.retrieval_documents import RetrievalSectionType


class RetrievalChunk(BaseModel):
    """Immutable indexable excerpt from one retrieval-document section."""

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    repository: GitHubRepository
    section_id: str = Field(min_length=1)
    section_type: RetrievalSectionType
    chunk_index: int = Field(ge=0)
    content: str = Field(min_length=1)
    text: str = Field(min_length=1)
    artifact: ArtifactReference | None = None
    metadata: EventMetadata

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_provenance(self) -> "RetrievalChunk":
        if self.metadata.event_id != self.event_id:
            raise ValueError("Retrieval chunk metadata must match its event ID.")
        if self.metadata.repository != self.repository:
            raise ValueError("Retrieval chunk metadata must match its repository.")
        if self.artifact is not None and self.artifact.repository != self.repository:
            raise ValueError("Retrieval chunk artifact must match its repository.")
        return self
