"""Separate ranked seeds from structurally discovered context."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reporecall.models.relationships import (
    ArtifactReference,
    EngineeringRelationship,
    RelationshipEvidenceType,
    RelationshipType,
)
from reporecall.models.reranking import RerankedSearchHit
from reporecall.models.retrieval_chunks import RetrievalChunk


class RelationshipTraversalDirection(str, Enum):
    """Direction of discovery without reversing the recorded relationship."""

    OUTGOING = "outgoing"
    INCOMING = "incoming"


class ContextExpansionReason(BaseModel):
    """Keep the complete direct relationship, including its original evidence."""

    seed_chunk_id: str
    relationship: EngineeringRelationship
    traversal_direction: RelationshipTraversalDirection
    contextual: bool = False

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("seed_chunk_id")
    @classmethod
    def require_seed_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Seed chunk ID must not be blank.")
        return value

    @property
    def relationship_type(self) -> RelationshipType:
        return self.relationship.relationship_type

    @property
    def evidence_types(self) -> tuple[RelationshipEvidenceType, ...]:
        return (self.relationship.evidence_type,)

    @property
    def source_artifact(self) -> ArtifactReference:
        return self.relationship.source

    @property
    def target_artifact(self) -> ArtifactReference:
        return self.relationship.target

    @property
    def related_artifact(self) -> ArtifactReference:
        if self.traversal_direction == RelationshipTraversalDirection.OUTGOING:
            return self.target_artifact
        return self.source_artifact


class ExpandedContextChunk(BaseModel):
    """Additional context with explicit reasons and no retrieval score or rank."""

    chunk: RetrievalChunk
    reasons: tuple[ContextExpansionReason, ...] = Field(min_length=1)

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def validate_reasons(self) -> "ExpandedContextChunk":
        if any(
            reason.related_artifact != self.chunk.artifact for reason in self.reasons
        ):
            raise ValueError("Expansion reasons must point to the expanded artifact.")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("Expansion reasons must be unique.")
        return self


class ExpandedContextResult(BaseModel):
    """Bounded structural context, preserving the authoritative ranked seeds."""

    query: str
    seed_hits: tuple[RerankedSearchHit, ...]
    expanded_chunks: tuple[ExpandedContextChunk, ...]
    truncated: bool = False

    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def seed_count(self) -> int:
        return len(self.seed_hits)

    @property
    def expanded_count(self) -> int:
        return len(self.expanded_chunks)

    @model_validator(mode="after")
    def validate_membership(self) -> "ExpandedContextResult":
        seeds = {hit.chunk.chunk_id: hit for hit in self.seed_hits}
        expanded_ids = [item.chunk.chunk_id for item in self.expanded_chunks]
        if len(seeds) != len(self.seed_hits) or len(set(expanded_ids)) != len(
            expanded_ids
        ):
            raise ValueError("Context chunk identities must be unique.")
        if seeds.keys() & set(expanded_ids):
            raise ValueError("Seeds must not be duplicated as expanded context.")
        if tuple(hit.rank for hit in self.seed_hits) != tuple(range(1, len(seeds) + 1)):
            raise ValueError("Seeds must retain contiguous reranker rank order.")
        for item in self.expanded_chunks:
            for reason in item.reasons:
                seed = seeds.get(reason.seed_chunk_id)
                if seed is None:
                    raise ValueError("Expansion reason must identify a supplied seed.")
                origin = (
                    reason.source_artifact
                    if reason.traversal_direction
                    == RelationshipTraversalDirection.OUTGOING
                    else reason.target_artifact
                )
                if origin != seed.chunk.artifact:
                    raise ValueError(
                        "Expansion reason must originate at its seed artifact."
                    )
                if (
                    seed.chunk.event_id != item.chunk.event_id
                    or seed.chunk.repository != item.chunk.repository
                ):
                    raise ValueError(
                        "Expanded context must remain event and repository local."
                    )
        return self
