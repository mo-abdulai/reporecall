from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class CrossEncoderRerankerConfig(BaseModel):
    """Runtime limits and model settings for cross-encoder reranking."""

    model_name: str = DEFAULT_CROSS_ENCODER_MODEL
    candidate_k: int = Field(default=20, gt=0)
    top_k: int = Field(default=5, gt=0)
    batch_size: int = Field(default=16, gt=0)
    device: str | None = None

    model_config = ConfigDict(frozen=True)

    @field_validator("model_name")
    @classmethod
    def require_nonblank_model_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Cross-encoder model name must not be blank.")
        return value

    @field_validator("device")
    @classmethod
    def require_nonblank_device(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Cross-encoder device must not be blank.")
        return value

    @model_validator(mode="after")
    def validate_depths(self) -> "CrossEncoderRerankerConfig":
        if self.top_k > self.candidate_k:
            raise ValueError("Cross-encoder top-k cannot exceed candidate-k.")
        return self


class RerankerBackend(Protocol):
    """Minimal query-passage scoring boundary used by reranking."""

    @property
    def model_name(self) -> str:
        """Return the stable identity of the scoring model."""

        ...

    def score(
        self,
        query: str,
        passages: Sequence[str],
    ) -> Sequence[float]:
        """Return one relevance score for every passage in input order."""

        ...
