"""Offline source resolution by exact document and artifact identity."""

from collections.abc import Sequence
from types import MappingProxyType

from reporecall.models.provenance import CitationSource
from reporecall.models.relationships import ArtifactReference
from reporecall.models.retrieval_chunks import RetrievalChunk
from reporecall.models.retrieval_documents import RetrievalDocument, RetrievalSource


class CitationProvenanceError(ValueError):
    """Missing or inconsistent canonical citation provenance."""


def _source_order(source: RetrievalSource) -> tuple[str, str, str, str, bool, str, str]:
    artifact = source.artifact
    return (
        artifact.repository.owner,
        artifact.repository.name,
        artifact.artifact_type.value,
        artifact.identifier,
        source.url is not None,
        source.url or "",
        source.label,
    )


class CitationProvenanceIndex:
    """Immutable lookups; sources never cross document or artifact boundaries.

    Artifact-less chunks remain citable through their own structured identity,
    with no document-wide source fallback. Identical records are deduplicated;
    distinct records sharing a URL remain distinct.
    """

    def __init__(
        self,
        *,
        documents: Sequence[RetrievalDocument],
        chunks: Sequence[RetrievalChunk],
    ) -> None:
        by_document: dict[str, RetrievalDocument] = {}
        sources: dict[tuple[str, ArtifactReference], tuple[RetrievalSource, ...]] = {}
        for document in documents:
            ordered = tuple(sorted(set(document.sources), key=_source_order))
            canonical = document.model_copy(update={"sources": ordered})
            previous = by_document.get(document.document_id)
            if previous is not None and previous != canonical:
                raise CitationProvenanceError("Conflicting duplicate document ID.")
            by_document[document.document_id] = canonical
            grouped: dict[ArtifactReference, list[RetrievalSource]] = {}
            for source in ordered:
                if source.artifact.repository != document.repository:
                    raise CitationProvenanceError(
                        "Source repository does not match its document."
                    )
                grouped.setdefault(source.artifact, []).append(source)
            for artifact, records in grouped.items():
                sources[(document.document_id, artifact)] = tuple(records)
        by_chunk: dict[str, RetrievalChunk] = {}
        for chunk in chunks:
            previous_chunk = by_chunk.get(chunk.chunk_id)
            if previous_chunk is not None and previous_chunk != chunk:
                raise CitationProvenanceError("Conflicting duplicate chunk ID.")
            parent_document = by_document.get(chunk.document_id)
            if parent_document is None:
                raise CitationProvenanceError(f"Missing document: {chunk.document_id}")
            if (
                chunk.event_id != parent_document.event_id
                or chunk.repository != parent_document.repository
            ):
                raise CitationProvenanceError(
                    "Chunk event/repository must match its document."
                )
            section = next(
                (
                    s
                    for s in parent_document.sections
                    if s.section_id == chunk.section_id
                ),
                None,
            )
            if section is None:
                raise CitationProvenanceError(
                    "Chunk section is missing from its document."
                )
            if (
                section.section_type != chunk.section_type
                or section.artifact != chunk.artifact
            ):
                raise CitationProvenanceError(
                    "Chunk section type/artifact must match its document section."
                )
            by_chunk[chunk.chunk_id] = chunk
        self._documents = MappingProxyType(by_document)
        self._chunks = MappingProxyType(by_chunk)
        self._sources = MappingProxyType(sources)

    def artifact_sources(
        self, document_id: str, artifact: ArtifactReference
    ) -> tuple[RetrievalSource, ...]:
        """Resolve exact artifact sources inside one known document."""
        if document_id not in self._documents:
            raise CitationProvenanceError(f"Missing document: {document_id}")
        return self._sources.get((document_id, artifact), ())

    def sources_for_chunk(self, chunk: RetrievalChunk) -> tuple[CitationSource, ...]:
        """Resolve an indexed chunk, rejecting stale or conflicting copies."""
        indexed = self._chunks.get(chunk.chunk_id)
        if indexed is None:
            raise CitationProvenanceError(f"Missing indexed chunk: {chunk.chunk_id}")
        if indexed != chunk:
            raise CitationProvenanceError(
                "Citation chunk conflicts with indexed provenance."
            )
        records = (
            ()
            if chunk.artifact is None
            else self.artifact_sources(chunk.document_id, chunk.artifact)
        )
        return tuple(
            CitationSource(
                source=source,
                repository=chunk.repository,
                document_id=chunk.document_id,
                event_id=chunk.event_id,
                chunk_id=chunk.chunk_id,
                section_id=chunk.section_id,
                section_type=chunk.section_type,
            )
            for source in records
        )
