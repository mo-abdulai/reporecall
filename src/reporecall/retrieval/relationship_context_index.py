"""Offline, event-scoped structural lookups without expansion policy."""

from collections.abc import Sequence
from types import MappingProxyType

from reporecall.github.models import GitHubRepository
from reporecall.models.events import EngineeringEvent
from reporecall.models.relationships import ArtifactReference, EngineeringRelationship
from reporecall.models.retrieval_chunks import RetrievalChunk


class RelationshipContextError(ValueError):
    """Inconsistent event, relationship, or chunk state in context expansion."""


def chunk_order(chunk: RetrievalChunk) -> tuple[str, str, int, str]:
    """Use stable document/section IDs, then section-local index and chunk ID."""
    return (chunk.document_id, chunk.section_id, chunk.chunk_index, chunk.chunk_id)


def relationship_order(relationship: EngineeringRelationship) -> tuple[str, ...]:
    """Order full relationship provenance, including distinct evidence payloads."""
    return (
        relationship.relationship_type.value,
        relationship.source.artifact_type.value,
        relationship.source.identifier,
        relationship.target.artifact_type.value,
        relationship.target.identifier,
        relationship.evidence_type.value,
        str(relationship.evidence is not None),
        relationship.evidence or "",
        str(relationship.source_field is not None),
        relationship.source_field or "",
    )


class RelationshipContextIndex:
    """Snapshot immutable records into repository- and event-scoped lookups."""

    def __init__(
        self, *, events: Sequence[EngineeringEvent], chunks: Sequence[RetrievalChunk]
    ) -> None:
        event_records: dict[str, EngineeringEvent] = {}
        repositories: dict[str, GitHubRepository] = {}
        edges: dict[
            tuple[str, ArtifactReference], set[tuple[EngineeringRelationship, bool]]
        ] = {}
        for event in events:
            if event.event_id in event_records:
                if event_records[event.event_id] != event:
                    raise RelationshipContextError("Conflicting duplicate event ID.")
                continue
            event_records[event.event_id] = event
            repositories[event.event_id] = event.repository
            for contextual, relationships in (
                (False, event.relationships),
                (True, event.contextual_relationships),
            ):
                for relationship in relationships:
                    # Contextual records may legitimately reference another repository.
                    # Keep them in the lookup; expansion enforces repository locality.
                    for artifact in {relationship.source, relationship.target}:
                        edges.setdefault((event.event_id, artifact), set()).add(
                            (relationship, contextual)
                        )
        by_id: dict[str, RetrievalChunk] = {}
        by_artifact: dict[tuple[str, ArtifactReference], list[RetrievalChunk]] = {}
        for chunk in chunks:
            if chunk.chunk_id in by_id:
                raise RelationshipContextError("Duplicate chunk ID in context index.")
            repository = repositories.get(chunk.event_id)
            if repository is None:
                raise RelationshipContextError(f"Missing event: {chunk.event_id}")
            if chunk.repository != repository:
                raise RelationshipContextError(
                    "Chunk repository does not match its event."
                )
            by_id[chunk.chunk_id] = chunk
            if chunk.artifact is not None:
                by_artifact.setdefault((chunk.event_id, chunk.artifact), []).append(
                    chunk
                )
        self._repositories = MappingProxyType(repositories)
        self._chunks = MappingProxyType(by_id)
        self._artifacts = MappingProxyType(
            {
                key: tuple(sorted(values, key=chunk_order))
                for key, values in by_artifact.items()
            }
        )
        self._edges = MappingProxyType(
            {
                key: tuple(
                    sorted(
                        values, key=lambda item: (item[1], relationship_order(item[0]))
                    )
                )
                for key, values in edges.items()
            }
        )

    def validate_seed(self, chunk: RetrievalChunk) -> None:
        """Require event provenance and agreement with any indexed seed copy."""
        repository = self._repositories.get(chunk.event_id)
        if repository is None:
            raise RelationshipContextError(f"Missing seed event: {chunk.event_id}")
        if repository != chunk.repository:
            raise RelationshipContextError("Seed repository does not match its event.")
        indexed = self._chunks.get(chunk.chunk_id)
        if indexed is not None and indexed != chunk:
            raise RelationshipContextError(
                "Seed conflicts with indexed chunk identity."
            )

    def related_edges(
        self, event_id: str, artifact: ArtifactReference
    ) -> tuple[tuple[EngineeringRelationship, bool], ...]:
        """Return incoming and outgoing edges with their event-contextual flag."""
        return self._edges.get((event_id, artifact), ())

    def artifact_chunks(
        self, event_id: str, artifact: ArtifactReference
    ) -> tuple[RetrievalChunk, ...]:
        """Return canonical chunks only from the requested event and artifact."""
        return self._artifacts.get((event_id, artifact), ())
