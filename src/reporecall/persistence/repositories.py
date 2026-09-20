"""Atomic immutable corpus writes and deterministic restart reconstruction."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from sqlalchemy import Table, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from reporecall.models import (
    ChunkEmbedding,
    EngineeringEvent,
    RetrievalChunk,
    RetrievalDocument,
)
from reporecall.persistence.exceptions import (
    PersistenceIntegrityError,
    PersistenceQueryError,
)
from reporecall.persistence.mapping import (
    _check_projection,
    chunk_from_record,
    chunk_to_record,
    document_from_record,
    document_to_record,
    embedding_from_record,
    embedding_to_record,
    event_from_record,
    event_to_record,
    record_values,
    validate_chunk_parent,
    validate_embedding_parent,
)
from reporecall.persistence.orm import (
    Base,
    ChunkRecord,
    DocumentRecord,
    EmbeddingRecord,
    EventRecord,
)


@dataclass(frozen=True)
class PersistentCorpus:
    """One consistent snapshot suitable for rebuilding existing in-memory indexes."""

    events: tuple[EngineeringEvent, ...]
    documents: tuple[RetrievalDocument, ...]
    chunks: tuple[RetrievalChunk, ...]
    embeddings: tuple[ChunkEmbedding, ...]


class PersistentCorpusRepository:
    """Store complete logical units atomically; existing identities are immutable.

    Identical writes are idempotent. Conflicting payloads require an explicit
    future replacement/indexing lifecycle, never an implicit last-writer win.
    """

    def __init__(self, *, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def store(
        self,
        *,
        events: Sequence[EngineeringEvent] = (),
        documents: Sequence[RetrievalDocument] = (),
        chunks: Sequence[RetrievalChunk] = (),
        embeddings: Sequence[ChunkEmbedding] = (),
    ) -> None:
        """Commit all supplied records together, rolling back on any failure."""
        # Mapping also revalidates copied/mutated domain inputs before opening a transaction.
        event_rows = [event_to_record(item) for item in events]
        document_rows = [document_to_record(item) for item in documents]
        chunk_rows = [chunk_to_record(item) for item in chunks]
        embedding_rows = [embedding_to_record(item) for item in embeddings]
        try:
            with self.session_factory.begin() as session:
                for row in event_rows:
                    _insert_immutable(session, row)
                for document_row in document_rows:
                    document = document_from_record(document_row)
                    parent = session.get(EventRecord, document.event_id)
                    if (
                        parent is None
                        or event_from_record(parent).repository != document.repository
                    ):
                        raise PersistenceIntegrityError(
                            "Document must resolve to its own repository event."
                        )
                    _insert_immutable(session, document_row)
                for chunk_row in chunk_rows:
                    chunk = chunk_from_record(chunk_row)
                    document_parent = session.get(DocumentRecord, chunk.document_id)
                    if document_parent is None:
                        raise PersistenceIntegrityError("Chunk document is missing.")
                    validate_chunk_parent(chunk, document_from_record(document_parent))
                    _insert_immutable(session, chunk_row)
                for embedding_row in embedding_rows:
                    embedding = embedding_from_record(embedding_row)
                    chunk_parent = session.get(ChunkRecord, embedding.chunk_id)
                    if chunk_parent is None:
                        raise PersistenceIntegrityError("Embedding chunk is missing.")
                    validate_embedding_parent(
                        embedding, chunk_from_record(chunk_parent)
                    )
                    _insert_immutable(session, embedding_row)
        except IntegrityError as exc:
            raise PersistenceIntegrityError(
                "Corpus write violates database integrity; transaction rolled back."
            ) from exc
        except SQLAlchemyError as exc:
            raise PersistenceQueryError(
                "Corpus write failed; transaction rolled back."
            ) from exc

    def load_events(self) -> tuple[EngineeringEvent, ...]:
        return self._load(EventRecord, event_from_record)

    def load_documents(self) -> tuple[RetrievalDocument, ...]:
        return self._load(DocumentRecord, document_from_record)

    def load_chunks(self) -> tuple[RetrievalChunk, ...]:
        return self._load(ChunkRecord, chunk_from_record)

    def load_embeddings(self) -> tuple[ChunkEmbedding, ...]:
        return self._load(EmbeddingRecord, embedding_from_record)

    def _load[RowT: Base, DomainT](
        self, row_type: type[RowT], mapper: Callable[[RowT], DomainT]
    ) -> tuple[DomainT, ...]:
        try:
            with self.session_factory() as session:
                return _load_rows(session, row_type, mapper)
        except SQLAlchemyError as exc:
            raise PersistenceQueryError(
                "Unable to load persisted corpus records."
            ) from exc

    def load_corpus(self) -> PersistentCorpus:
        """Load and cross-validate all records in one repeatable-read snapshot."""
        try:
            with self.session_factory() as session:
                corpus = PersistentCorpus(
                    events=_load_rows(session, EventRecord, event_from_record),
                    documents=_load_rows(session, DocumentRecord, document_from_record),
                    chunks=_load_rows(session, ChunkRecord, chunk_from_record),
                    embeddings=_load_rows(
                        session, EmbeddingRecord, embedding_from_record
                    ),
                )
                events = {e.event_id: e for e in corpus.events}
                documents = {d.document_id: d for d in corpus.documents}
                chunks = {c.chunk_id: c for c in corpus.chunks}
                for document in corpus.documents:
                    parent = events.get(document.event_id)
                    if parent is None or parent.repository != document.repository:
                        raise PersistenceIntegrityError(
                            "Persisted document event is inconsistent."
                        )
                for chunk in corpus.chunks:
                    if chunk.document_id not in documents:
                        raise PersistenceIntegrityError(
                            "Persisted chunk document is missing."
                        )
                    validate_chunk_parent(chunk, documents[chunk.document_id])
                for embedding in corpus.embeddings:
                    if embedding.chunk_id not in chunks:
                        raise PersistenceIntegrityError(
                            "Persisted embedding chunk is missing."
                        )
                    validate_embedding_parent(embedding, chunks[embedding.chunk_id])
                return corpus
        except SQLAlchemyError as exc:
            raise PersistenceQueryError(
                "Unable to reconstruct persistent corpus."
            ) from exc


def _load_rows[RowT: Base, DomainT](
    session: Session, row_type: type[RowT], mapper: Callable[[RowT], DomainT]
) -> tuple[DomainT, ...]:
    return tuple(
        mapper(row)
        for row in session.scalars(
            select(row_type).order_by(
                *(
                    column.collate("C")
                    for column in cast(Table, row_type.__table__).primary_key.columns
                )
            )
        )
    )


def _insert_immutable(session: Session, record: Base) -> None:
    table = cast(Table, record.__table__)
    values = record_values(record)
    # PostgreSQL serializes competing inserts on the same primary key. A conflicting
    # transaction fails/retries; it cannot silently replace an existing identity.
    session.execute(insert(table).values(values).on_conflict_do_nothing())
    identity = tuple(values[column.name] for column in table.primary_key.columns)
    existing = session.get(type(record), identity)
    if existing is None:
        raise PersistenceIntegrityError(
            "Upsert did not resolve its canonical identity."
        )
    _check_projection(existing, record)
