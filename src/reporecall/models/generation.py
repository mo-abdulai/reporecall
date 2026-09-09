import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reporecall.github.models import GitHubRepository
from reporecall.models.relationships import ArtifactReference
from reporecall.models.retrieval_documents import RetrievalSectionType


class RAGEvidence(BaseModel):
    """One ranked source record supplied to the generation model."""

    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    rank: int = Field(gt=0)
    score: float = Field(allow_inf_nan=False)
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    repository: GitHubRepository
    section_id: str = Field(min_length=1)
    section_type: RetrievalSectionType
    artifact: ArtifactReference | None
    content: str
    content_truncated: bool = False

    model_config = ConfigDict(frozen=True)

    @field_validator("score")
    @classmethod
    def require_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("RAG evidence score must be finite.")
        return value


class RAGContext(BaseModel):
    """Deterministic bounded repository evidence prepared for a prompt."""

    query: str = Field(min_length=1)
    evidence: tuple[RAGEvidence, ...]
    text: str
    retrieved_count: int = Field(ge=0)
    included_evidence_count: int = Field(ge=0)
    truncated: bool

    model_config = ConfigDict(frozen=True)

    @field_validator("query")
    @classmethod
    def require_nonblank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("RAG context query must not be blank.")
        return value

    @model_validator(mode="after")
    def validate_evidence_counts(self) -> "RAGContext":
        if self.included_evidence_count != len(self.evidence):
            raise ValueError("Included evidence count must match the evidence records.")
        if self.retrieved_count < self.included_evidence_count:
            raise ValueError("Retrieved count cannot be smaller than included evidence.")
        expected_ids = tuple(f"E{index}" for index in range(1, len(self.evidence) + 1))
        if tuple(item.evidence_id for item in self.evidence) != expected_ids:
            raise ValueError("RAG evidence IDs must be sequential and one-based.")
        ranks = tuple(item.rank for item in self.evidence)
        if ranks != tuple(sorted(ranks)) or len(set(ranks)) != len(ranks):
            raise ValueError("RAG evidence must preserve unique retrieval-rank order.")
        if bool(self.evidence) is not bool(self.text):
            raise ValueError("RAG context text must match whether evidence is included.")
        return self


class RAGPrompt(BaseModel):
    """Separated system instructions and user evidence prompt."""

    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)

    model_config = ConfigDict(frozen=True)


class RAGAnswer(BaseModel):
    """Grounded generated answer with the exact supplied evidence."""

    query: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    model_name: str | None
    evidence: tuple[RAGEvidence, ...]
    retrieved_count: int = Field(ge=0)
    included_evidence_count: int = Field(ge=0)
    context_truncated: bool
    insufficient_evidence: bool

    model_config = ConfigDict(frozen=True)

    @field_validator("query", "answer")
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("RAG answer text fields must not be blank.")
        return value

    @field_validator("model_name")
    @classmethod
    def require_valid_model_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("RAG answer model name must not be blank.")
        return value

    @model_validator(mode="after")
    def validate_answer_evidence(self) -> "RAGAnswer":
        if self.included_evidence_count != len(self.evidence):
            raise ValueError("Included evidence count must match answer evidence.")
        if self.retrieved_count < self.included_evidence_count:
            raise ValueError("Retrieved count cannot be smaller than answer evidence.")
        if not self.insufficient_evidence and self.model_name is None:
            raise ValueError("Generated answers must preserve their model name.")
        return self
