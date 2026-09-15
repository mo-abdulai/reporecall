from pydantic import BaseModel, ConfigDict, Field, field_validator

from reporecall.models.retrieval_filters import MetadataFilter


class UnderstoodQuery(BaseModel):
    """Validated query interpretation before retrieval is executed."""

    original_query: str = Field(min_length=1)
    retrieval_query: str = Field(min_length=1)
    metadata_filter: MetadataFilter | None = None
    extracted_constraints: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    model_name: str | None = None

    model_config = ConfigDict(frozen=True)

    @field_validator("original_query", "retrieval_query")
    @classmethod
    def require_nonblank_query_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query text must not be blank.")
        return value

    @field_validator("extracted_constraints", "warnings")
    @classmethod
    def reject_blank_diagnostic_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            if not value.strip():
                raise ValueError("Query-understanding diagnostics must not be blank.")
        return values

    @field_validator("model_name")
    @classmethod
    def reject_blank_model_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Query-understanding model name must not be blank.")
        return value
