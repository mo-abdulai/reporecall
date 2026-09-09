from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_RAG_MODEL = "gpt-4.1-mini"


class RAGConfig(BaseModel):
    """Small provider-independent configuration for baseline grounded RAG."""

    top_k: int = Field(default=5, gt=0)
    max_context_chars: int = Field(default=12_000, gt=0)
    model_name: str = DEFAULT_RAG_MODEL
    temperature: float = Field(default=0.0, ge=0)

    model_config = ConfigDict(frozen=True)

    @field_validator("model_name")
    @classmethod
    def require_nonblank_model_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("RAG model name must not be blank.")
        return value
