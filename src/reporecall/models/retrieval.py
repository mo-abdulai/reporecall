import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reporecall.models.retrieval_chunks import RetrievalChunk

SourceTextSha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class VectorIndexManifest(BaseModel):
    """Immutable metadata that maps FAISS rows to source chunk identities."""

    schema_version: Literal[1] = 1
    model_name: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    normalized: bool
    vector_count: int = Field(gt=0)
    chunk_ids: tuple[str, ...] = Field(min_length=1)
    source_text_sha256: tuple[SourceTextSha256, ...] = Field(min_length=1)

    model_config = ConfigDict(frozen=True)

    @field_validator("model_name")
    @classmethod
    def require_nonblank_model_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Vector index model name must not be blank.")
        return value

    @field_validator("chunk_ids")
    @classmethod
    def require_valid_chunk_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not chunk_id.strip() for chunk_id in value):
            raise ValueError("Vector index chunk IDs must not be blank.")
        if len(set(value)) != len(value):
            raise ValueError("Vector index chunk IDs must be unique.")
        return value

    @model_validator(mode="after")
    def validate_row_metadata(self) -> "VectorIndexManifest":
        if len(self.chunk_ids) != self.vector_count:
            raise ValueError("Vector count must match the number of chunk IDs.")
        if len(self.source_text_sha256) != self.vector_count:
            raise ValueError("Vector count must match the number of source hashes.")
        return self


class VectorIndexMatch(BaseModel):
    """One ranked FAISS row match before source chunk reconstruction."""

    chunk_id: str = Field(min_length=1)
    source_text_sha256: SourceTextSha256
    score: float = Field(allow_inf_nan=False)
    rank: int = Field(gt=0)

    model_config = ConfigDict(frozen=True)


class VectorSearchHit(BaseModel):
    """One ranked dense-retrieval result with complete chunk provenance."""

    chunk: RetrievalChunk
    score: float = Field(allow_inf_nan=False)
    rank: int = Field(gt=0)

    model_config = ConfigDict(frozen=True)

    @field_validator("score")
    @classmethod
    def require_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Vector search score must be finite.")
        return value
