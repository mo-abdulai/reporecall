from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingConfig(BaseModel):
    """Runtime configuration for a SentenceTransformers embedding backend."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = Field(default=32, gt=0)
    normalize_embeddings: bool = True
    device: str | None = None

    model_config = ConfigDict(frozen=True)

    @field_validator("model_name")
    @classmethod
    def require_nonblank_model_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Embedding model name must not be blank.")
        return value
