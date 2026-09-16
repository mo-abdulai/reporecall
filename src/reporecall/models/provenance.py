"""Immutable citations that retain ranked and structural evidence separately."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reporecall.github.models import GitHubRepository
from reporecall.models.context_expansion import (
    ExpandedContextChunk,
    ExpandedContextResult,
)
from reporecall.models.reranking import RerankedSearchHit
from reporecall.models.retrieval_chunks import RetrievalChunk
from reporecall.models.retrieval_documents import RetrievalSectionType, RetrievalSource


class CitationKind(str, Enum):
    """Distinguish ranked evidence from structurally expanded context."""

    RETRIEVED = "retrieved"
    EXPANDED = "expanded"


class CitationIdentifier(BaseModel):
    """Bundle-local identifier with a derived canonical label."""

    kind: CitationKind
    index: int = Field(gt=0, strict=True)
    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def label(self) -> str:
        prefix = "R" if self.kind == CitationKind.RETRIEVED else "X"
        return f"{prefix}{self.index}"


class CitationSource(BaseModel):
    """An original source record joined to explicit chunk provenance."""

    source: RetrievalSource
    repository: GitHubRepository
    document_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    section_id: str = Field(min_length=1)
    section_type: RetrievalSectionType
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def validate_repository(self) -> "CitationSource":
        if self.source.artifact.repository != self.repository:
            raise ValueError("Citation source repository must match its artifact.")
        return self


def _validate_sources(
    sources: tuple[CitationSource, ...], chunk: RetrievalChunk
) -> None:
    for source in sources:
        if (
            source.repository != chunk.repository
            or source.document_id != chunk.document_id
            or source.event_id != chunk.event_id
            or source.chunk_id != chunk.chunk_id
            or source.section_id != chunk.section_id
            or source.section_type != chunk.section_type
            or source.source.artifact != chunk.artifact
        ):
            raise ValueError(
                "Citation source must match the complete chunk provenance."
            )
    if len(set(sources)) != len(sources):
        raise ValueError("Citation sources must not contain identical records.")


def _urls(sources: tuple[CitationSource, ...]) -> tuple[str, ...]:
    return tuple(
        sorted({item.source.url for item in sources if item.source.url is not None})
    )


class RetrievedEvidenceCitation(BaseModel):
    """A ranked hit preserved without duplicating or rounding diagnostics."""

    citation: CitationIdentifier
    hit: RerankedSearchHit
    sources: tuple[CitationSource, ...] = ()
    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def urls(self) -> tuple[str, ...]:
        return _urls(self.sources)

    @model_validator(mode="after")
    def validate_citation(self) -> "RetrievedEvidenceCitation":
        if self.citation.kind != CitationKind.RETRIEVED:
            raise ValueError("Retrieved evidence requires an R citation identifier.")
        if self.citation.index != self.hit.rank:
            raise ValueError("Retrieved citation index must match reranker rank.")
        _validate_sources(self.sources, self.hit.chunk)
        return self


class SeedCitationReference(BaseModel):
    """Link an original expansion seed identity to its retrieved citation."""

    seed_chunk_id: str = Field(min_length=1)
    citation: CitationIdentifier
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def validate_reference(self) -> "SeedCitationReference":
        if not self.seed_chunk_id.strip():
            raise ValueError("Seed chunk ID must not be blank.")
        if self.citation.kind != CitationKind.RETRIEVED:
            raise ValueError("Expansion seeds must reference retrieved citations.")
        return self


class ExpandedEvidenceCitation(BaseModel):
    """Structural context and seed links, with no retrieval score or rank."""

    citation: CitationIdentifier
    expanded_chunk: ExpandedContextChunk
    sources: tuple[CitationSource, ...] = ()
    seed_references: tuple[SeedCitationReference, ...] = Field(min_length=1)
    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def urls(self) -> tuple[str, ...]:
        return _urls(self.sources)

    @model_validator(mode="after")
    def validate_citation(self) -> "ExpandedEvidenceCitation":
        if self.citation.kind != CitationKind.EXPANDED:
            raise ValueError("Expanded context requires an X citation identifier.")
        _validate_sources(self.sources, self.expanded_chunk.chunk)
        expected = {reason.seed_chunk_id for reason in self.expanded_chunk.reasons}
        actual = [ref.seed_chunk_id for ref in self.seed_references]
        if set(actual) != expected or len(set(actual)) != len(actual):
            raise ValueError(
                "Seed references must exactly cover expansion reason seeds."
            )
        indices = [ref.citation.index for ref in self.seed_references]
        if indices != sorted(set(indices)):
            raise ValueError(
                "Seed references must use unique retrieved citation order."
            )
        return self


class CitationBundle(BaseModel):
    """A deterministic bundle of independently traceable evidence citations."""

    query: str
    retrieved: tuple[RetrievedEvidenceCitation, ...]
    expanded: tuple[ExpandedEvidenceCitation, ...]
    context_truncated: bool = False
    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def retrieved_count(self) -> int:
        return len(self.retrieved)

    @property
    def expanded_count(self) -> int:
        return len(self.expanded)

    @model_validator(mode="after")
    def validate_bundle(self) -> "CitationBundle":
        for citations in (self.retrieved, self.expanded):
            if tuple(item.citation.index for item in citations) != tuple(
                range(1, len(citations) + 1)
            ):
                raise ValueError(
                    "Citation labels must be unique and contiguous in bundle order."
                )
        # Reuse the authoritative structural and membership validation.
        ExpandedContextResult(
            query=self.query,
            seed_hits=tuple(item.hit for item in self.retrieved),
            expanded_chunks=tuple(item.expanded_chunk for item in self.expanded),
            truncated=self.context_truncated,
        )
        seeds = {item.hit.chunk.chunk_id: item.citation for item in self.retrieved}
        for item in self.expanded:
            for ref in item.seed_references:
                if seeds.get(ref.seed_chunk_id) != ref.citation:
                    raise ValueError(
                        "Seed citation reference must resolve to its original chunk."
                    )
        return self
