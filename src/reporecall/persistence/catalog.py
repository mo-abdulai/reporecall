"""Read-only corpus queries used by application services, independent of HTTP."""

from dataclasses import dataclass

from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from reporecall.github import GitHubRepository
from reporecall.models import EngineeringEvent, RetrievalChunk, RetrievalDocument
from reporecall.persistence.exceptions import PersistenceQueryError
from reporecall.persistence.mapping import (
    chunk_from_record,
    document_from_record,
    event_from_record,
)
from reporecall.persistence.orm import (
    ChunkRecord,
    DocumentRecord,
    EmbeddingRecord,
    EventRecord,
)


@dataclass(frozen=True)
class EmbeddingSummary:
    model_name: str
    dimension: int
    normalized: bool
    count: int


@dataclass(frozen=True)
class CorpusStatistics:
    repositories: int
    events: int
    documents: int
    chunks: int
    embeddings: int
    embedding_models: tuple[EmbeddingSummary, ...]


class CorpusCatalog:
    """Small read-side repository with short-lived snapshot transactions."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.sessions = session_factory

    def check_ready(self) -> None:
        """Check connectivity, required schema columns, and pgvector without inference."""
        try:
            with self.sessions() as session:
                for record in (
                    EventRecord,
                    DocumentRecord,
                    ChunkRecord,
                    EmbeddingRecord,
                ):
                    session.execute(select(record).limit(0))
                if not session.scalar(
                    text(
                        "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                    )
                ):
                    raise PersistenceQueryError(
                        "Required vector extension is unavailable."
                    )
        except SQLAlchemyError as exc:
            raise PersistenceQueryError(
                "Required database schema is unavailable."
            ) from exc

    def repositories(self) -> tuple[GitHubRepository, ...]:
        try:
            with self.sessions() as session:
                return tuple(
                    GitHubRepository(owner=row[0], name=row[1])
                    for row in session.execute(
                        select(
                            EventRecord.repository_owner, EventRecord.repository_name
                        )
                        .distinct()
                        .order_by(
                            EventRecord.repository_owner, EventRecord.repository_name
                        )
                    )
                )
        except SQLAlchemyError as exc:
            raise PersistenceQueryError("Unable to list repositories.") from exc

    def statistics(self) -> CorpusStatistics:
        try:
            with self.sessions() as session:
                counts = [
                    session.scalar(select(func.count()).select_from(table)) or 0
                    for table in (
                        EventRecord,
                        DocumentRecord,
                        ChunkRecord,
                        EmbeddingRecord,
                    )
                ]
                repositories = (
                    session.scalar(
                        select(func.count()).select_from(
                            select(
                                EventRecord.repository_owner,
                                EventRecord.repository_name,
                            )
                            .distinct()
                            .subquery()
                        )
                    )
                    or 0
                )
                models = tuple(
                    EmbeddingSummary(*row)
                    for row in session.execute(
                        select(
                            EmbeddingRecord.model_name,
                            EmbeddingRecord.dimension,
                            EmbeddingRecord.normalized,
                            func.count(),
                        )
                        .group_by(
                            EmbeddingRecord.model_name,
                            EmbeddingRecord.dimension,
                            EmbeddingRecord.normalized,
                        )
                        .order_by(
                            EmbeddingRecord.model_name,
                            EmbeddingRecord.dimension,
                            EmbeddingRecord.normalized,
                        )
                    )
                )
                return CorpusStatistics(
                    repositories, counts[0], counts[1], counts[2], counts[3], models
                )
        except SQLAlchemyError as exc:
            raise PersistenceQueryError("Unable to read corpus statistics.") from exc

    def event(self, identifier: str) -> EngineeringEvent | None:
        try:
            with self.sessions() as session:
                row = session.get(EventRecord, identifier)
                return event_from_record(row) if row is not None else None
        except SQLAlchemyError as exc:
            raise PersistenceQueryError("Unable to load event.") from exc

    def document(self, identifier: str) -> RetrievalDocument | None:
        try:
            with self.sessions() as session:
                row = session.get(DocumentRecord, identifier)
                return document_from_record(row) if row is not None else None
        except SQLAlchemyError as exc:
            raise PersistenceQueryError("Unable to load document.") from exc

    def chunk(self, identifier: str) -> RetrievalChunk | None:
        try:
            with self.sessions() as session:
                row = session.get(ChunkRecord, identifier)
                return chunk_from_record(row) if row is not None else None
        except SQLAlchemyError as exc:
            raise PersistenceQueryError("Unable to load chunk.") from exc
