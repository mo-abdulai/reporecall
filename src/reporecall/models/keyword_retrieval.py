import math

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reporecall.models.retrieval import SourceTextSha256
from reporecall.models.retrieval_chunks import RetrievalChunk


class KeywordIndexMatch(BaseModel):
    """One ranked BM25 row match before source chunk reconstruction."""

    chunk_id: str = Field(min_length=1)
    source_text_sha256: SourceTextSha256
    score: float = Field(allow_inf_nan=False)
    rank: int = Field(gt=0)

    model_config = ConfigDict(frozen=True)

    @field_validator("score")
    @classmethod
    def require_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("BM25 index score must be finite.")
        return value


class KeywordSearchHit(BaseModel):
    """One ranked lexical-retrieval result with complete chunk provenance."""

    chunk: RetrievalChunk
    score: float = Field(allow_inf_nan=False)
    rank: int = Field(gt=0)

    model_config = ConfigDict(frozen=True)

    @field_validator("score")
    @classmethod
    def require_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Keyword search score must be finite.")
        return value
