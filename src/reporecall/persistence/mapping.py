"""Validated domain/row conversions; JSON payloads are authoritative."""

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ValidationError
from sqlalchemy import inspect

from reporecall.models import (
    ChunkEmbedding,
    EngineeringEvent,
    RetrievalChunk,
    RetrievalDocument,
)
from reporecall.persistence.exceptions import PersistenceIntegrityError
from reporecall.persistence.orm import (
    EMBEDDING_DIMENSION,
    Base,
    ChunkRecord,
    DocumentRecord,
    EmbeddingRecord,
    EventRecord,
)


def source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_vector(vector: Sequence[float]) -> list[float]:
    """Check float32 finiteness/dimension/unit norm without changing the vector."""
    if len(vector) != EMBEDDING_DIMENSION:
        raise PersistenceIntegrityError(
            "Database vector dimension must be 384; migrate and reindex to change it."
        )
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(vector, dtype=np.float32)
    if values.shape != (EMBEDDING_DIMENSION,) or not np.isfinite(values).all():
        raise PersistenceIntegrityError("Vectors must contain finite float32 values.")
    if not np.isclose(np.linalg.norm(values), 1.0, rtol=1e-4, atol=1e-5):
        raise PersistenceIntegrityError("Vectors must be L2 normalized.")
    return values.tolist()


def _validated[T: BaseModel](model: type[T], payload: object) -> T:
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise PersistenceIntegrityError(
            "Invalid canonical persistence payload."
        ) from exc


def filter_projection(chunk: RetrievalChunk) -> dict[str, Any]:
    """Precompute exact metadata semantics, including Python Unicode casefold.

    Component prefixes avoid SQL LIKE wildcards and retain case-sensitive paths.
    No values are inferred from source text.
    """
    metadata = chunk.metadata
    projected = metadata.model_dump(mode="json")
    for name in ("labels", "languages", "file_extensions"):
        projected[name] = sorted({v.casefold() for v in getattr(metadata, name)})
    projected["actors"] = [
        a.model_dump(mode="json") for a in (*metadata.authors, *metadata.participants)
    ]
    projected["path_prefixes"] = sorted(
        {
            prefix
            for path in metadata.changed_paths
            for prefix in [
                path,
                *(path[:i] for i, char in enumerate(path) if char == "/"),
            ]
        }
    )
    return projected


def event_to_record(event: EngineeringEvent) -> EventRecord:
    event = _validated(EngineeringEvent, event.model_dump(mode="json"))
    return EventRecord(
        event_id=event.event_id,
        repository_owner=event.repository.owner,
        repository_name=event.repository.name,
        payload=event.model_dump(mode="json"),
    )


def document_to_record(document: RetrievalDocument) -> DocumentRecord:
    document = _validated(RetrievalDocument, document.model_dump(mode="json"))
    if any(
        s.artifact.repository != document.repository for s in document.sources
    ) or any(
        s.artifact is not None and s.artifact.repository != document.repository
        for s in document.sections
    ):
        raise PersistenceIntegrityError(
            "Document sources and sections must belong to its repository."
        )
    return DocumentRecord(
        document_id=document.document_id,
        event_id=document.event_id,
        repository_owner=document.repository.owner,
        repository_name=document.repository.name,
        title=document.title,
        payload=document.model_dump(mode="json"),
    )


def chunk_to_record(chunk: RetrievalChunk) -> ChunkRecord:
    chunk = _validated(RetrievalChunk, chunk.model_dump(mode="json"))
    return ChunkRecord(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        event_id=chunk.event_id,
        repository_owner=chunk.repository.owner,
        repository_name=chunk.repository.name,
        section_id=chunk.section_id,
        section_type=chunk.section_type.value,
        chunk_index=chunk.chunk_index,
        artifact_type=chunk.artifact.artifact_type.value if chunk.artifact else None,
        artifact_id=chunk.artifact.identifier if chunk.artifact else None,
        content=chunk.content,
        text=chunk.text,
        source_text_sha256=source_hash(chunk.text),
        event_metadata=chunk.metadata.model_dump(mode="json"),
        filter_metadata=filter_projection(chunk),
        payload=chunk.model_dump(mode="json"),
    )


def embedding_to_record(embedding: ChunkEmbedding) -> EmbeddingRecord:
    embedding = _validated(ChunkEmbedding, embedding.model_dump(mode="json"))
    if not embedding.normalized or embedding.dimension != EMBEDDING_DIMENSION:
        raise PersistenceIntegrityError(
            "Persisted embeddings must be normalized and 384 dimensional."
        )
    vector = validate_vector(embedding.vector)
    return EmbeddingRecord(
        chunk_id=embedding.chunk_id,
        model_name=embedding.model_name,
        dimension=embedding.dimension,
        normalized=embedding.normalized,
        source_text_sha256=embedding.source_text_sha256,
        embedding=vector,
        payload=embedding.model_dump(mode="json"),
    )


def record_values(record: Base) -> dict[str, Any]:
    """Column-name values for bound Core inserts (metadata is a reserved ORM name)."""
    return {
        prop.columns[0].name: getattr(record, prop.key)
        for prop in inspect(type(record)).column_attrs
    }


def _check_projection(actual: Base, expected: Base) -> None:
    for key, value in record_values(expected).items():
        stored = record_values(actual)[key]
        if key == "embedding":
            equal = np.array_equal(
                np.asarray(stored, dtype=np.float32),
                np.asarray(value, dtype=np.float32),
            )
        else:
            equal = stored == value
        if not equal:
            raise PersistenceIntegrityError(
                f"Stored {type(actual).__name__}.{key} disagrees with its canonical payload."
            )


def event_from_record(record: EventRecord) -> EngineeringEvent:
    domain = _validated(EngineeringEvent, record.payload)
    _check_projection(record, event_to_record(domain))
    return domain


def document_from_record(record: DocumentRecord) -> RetrievalDocument:
    domain = _validated(RetrievalDocument, record.payload)
    _check_projection(record, document_to_record(domain))
    return domain


def chunk_from_record(record: ChunkRecord) -> RetrievalChunk:
    domain = _validated(RetrievalChunk, record.payload)
    _check_projection(record, chunk_to_record(domain))
    return domain


def embedding_from_record(record: EmbeddingRecord) -> ChunkEmbedding:
    domain = _validated(ChunkEmbedding, record.payload)
    _check_projection(record, embedding_to_record(domain))
    return domain


def validate_chunk_parent(chunk: RetrievalChunk, document: RetrievalDocument) -> None:
    sections = {section.section_id: section for section in document.sections}
    section = sections.get(chunk.section_id)
    if (
        chunk.document_id != document.document_id
        or chunk.event_id != document.event_id
        or chunk.repository != document.repository
        or chunk.metadata != document.metadata
        or section is None
        or section.section_type != chunk.section_type
        or section.artifact != chunk.artifact
    ):
        raise PersistenceIntegrityError(
            "Chunk provenance/metadata must match its document and section."
        )


def validate_embedding_parent(embedding: ChunkEmbedding, chunk: RetrievalChunk) -> None:
    for field in (
        "chunk_id",
        "document_id",
        "event_id",
        "repository",
        "section_id",
        "section_type",
    ):
        if getattr(embedding, field) != getattr(chunk, field):
            raise PersistenceIntegrityError(
                "Embedding provenance must match its chunk."
            )
    if embedding.source_text_sha256 != source_hash(chunk.text):
        raise PersistenceIntegrityError("Embedding source hash is stale.")
