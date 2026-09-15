from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_QUERY_UNDERSTANDING_MODEL = "gpt-4.1-mini"


class QueryUnderstandingConfig(BaseModel):
    """Small provider-independent configuration for query interpretation."""

    model_name: str = DEFAULT_QUERY_UNDERSTANDING_MODEL
    temperature: float = Field(default=0.0, ge=0)

    model_config = ConfigDict(frozen=True)

    @field_validator("model_name")
    @classmethod
    def require_nonblank_model_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query-understanding model name must not be blank.")
        return value
