"""Explicit HTTP contracts composed from stable domain evidence models."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reporecall.github import GitHubRepository
from reporecall.models import (
    CitationBundle,
    ExpandedContextChunk,
    ExpandedEvidenceCitation,
    MetadataFilter,
    RerankedSearchHit,
    RetrievedEvidenceCitation,
)


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ErrorDetail(APIModel):
    code: str
    message: str


class ErrorResponse(APIModel):
    error: ErrorDetail


class StatusResponse(APIModel):
    status: str


class QueryRequest(APIModel):
    query: str = Field(min_length=1, max_length=10000)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query must not be blank")
        return value


class APIMetadataFilter(MetadataFilter):
    """Reject unknown HTTP filter fields while retaining canonical filter semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SearchRequest(QueryRequest):
    metadata_filter: APIMetadataFilter | None = None
    top_k: int = Field(default=5, ge=1, le=50, strict=True)
    expand_relationships: bool = True
    include_citations: bool = True


class RepositoryResponse(APIModel):
    repositories: tuple[GitHubRepository, ...]


class SeedReferenceResponse(APIModel):
    seed_chunk_id: str
    label: str


class RetrievedCitationResponse(APIModel):
    label: str
    evidence: RetrievedEvidenceCitation
    urls: tuple[str, ...]


class ExpandedCitationResponse(APIModel):
    label: str
    evidence: ExpandedEvidenceCitation
    urls: tuple[str, ...]
    seed_references: tuple[SeedReferenceResponse, ...]


class CitationBundleResponse(APIModel):
    query: str
    retrieved: tuple[RetrievedCitationResponse, ...]
    expanded: tuple[ExpandedCitationResponse, ...]
    context_truncated: bool

    @classmethod
    def from_domain(cls, bundle: CitationBundle) -> "CitationBundleResponse":
        """Expose canonical computed labels and URLs without resolving provenance again."""
        return cls(
            query=bundle.query,
            retrieved=tuple(
                RetrievedCitationResponse(
                    label=item.citation.label, evidence=item, urls=item.urls
                )
                for item in bundle.retrieved
            ),
            expanded=tuple(
                ExpandedCitationResponse(
                    label=item.citation.label,
                    evidence=item,
                    urls=item.urls,
                    seed_references=tuple(
                        SeedReferenceResponse(
                            seed_chunk_id=ref.seed_chunk_id, label=ref.citation.label
                        )
                        for ref in item.seed_references
                    ),
                )
                for item in bundle.expanded
            ),
            context_truncated=bundle.context_truncated,
        )


class SearchResponse(APIModel):
    query: str
    hits: tuple[RerankedSearchHit, ...]
    expanded_context: tuple[ExpandedContextChunk, ...]
    context_truncated: bool
    citations: CitationBundleResponse | None
