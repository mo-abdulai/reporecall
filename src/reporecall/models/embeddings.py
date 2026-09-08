import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reporecall.github.models import GitHubRepository
from reporecall.models.retrieval_documents import RetrievalSectionType


class ChunkEmbedding(BaseModel):
    """Immutable dense vector tied to one exact retrieval-ready chunk text."""

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    repository: GitHubRepository
    section_id: str = Field(min_length=1)
    section_type: RetrievalSectionType
    model_name: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    normalized: bool
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vector: tuple[float, ...] = Field(min_length=1)

    model_config = ConfigDict(frozen=True)

    @field_validator(
        "chunk_id",
        "document_id",
        "event_id",
        "section_id",
        "model_name",
    )
    @classmethod
    def require_nonblank_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Embedding identity fields must not be blank.")
        return value

    @model_validator(mode="after")
    def validate_vector(self) -> "ChunkEmbedding":
        if len(self.vector) != self.dimension:
            raise ValueError("Embedding vector length must match its dimension.")
        if not all(math.isfinite(value) for value in self.vector):
            raise ValueError("Embedding vector values must be finite.")
        return self
