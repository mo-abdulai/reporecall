"""Relational projections alongside complete canonical JSONB domain payloads."""

from typing import Any

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    CheckConstraint,
    ForeignKeyConstraint,
    Index,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Schema contract, not a runtime tuning knob. Changing it requires migration/reindex.
EMBEDDING_DIMENSION = 384


class Base(DeclarativeBase):
    pass


class EventRecord(Base):
    __tablename__ = "engineering_events"
    event_id: Mapped[str] = mapped_column(Text, primary_key=True)
    repository_owner: Mapped[str] = mapped_column(Text)
    repository_name: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    __table_args__ = (
        UniqueConstraint(
            "event_id",
            "repository_owner",
            "repository_name",
            name="uq_event_repository",
        ),
        Index("ix_events_repository", "repository_owner", "repository_name"),
    )


class DocumentRecord(Base):
    __tablename__ = "retrieval_documents"
    document_id: Mapped[str] = mapped_column(Text, primary_key=True)
    event_id: Mapped[str] = mapped_column(Text, index=True)
    repository_owner: Mapped[str] = mapped_column(Text)
    repository_name: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    __table_args__ = (
        ForeignKeyConstraint(
            ["event_id", "repository_owner", "repository_name"],
            [
                "engineering_events.event_id",
                "engineering_events.repository_owner",
                "engineering_events.repository_name",
            ],
            name="fk_document_event",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "document_id",
            "event_id",
            "repository_owner",
            "repository_name",
            name="uq_document_provenance",
        ),
        Index("ix_documents_repository", "repository_owner", "repository_name"),
    )


class ChunkRecord(Base):
    __tablename__ = "retrieval_chunks"
    chunk_id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_id: Mapped[str] = mapped_column(Text, index=True)
    event_id: Mapped[str] = mapped_column(Text, index=True)
    repository_owner: Mapped[str] = mapped_column(Text)
    repository_name: Mapped[str] = mapped_column(Text)
    section_id: Mapped[str] = mapped_column(Text)
    section_type: Mapped[str] = mapped_column(Text, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    artifact_type: Mapped[str | None] = mapped_column(Text)
    artifact_id: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    source_text_sha256: Mapped[str] = mapped_column(Text)
    event_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB)
    filter_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "event_id", "repository_owner", "repository_name"],
            [
                "retrieval_documents.document_id",
                "retrieval_documents.event_id",
                "retrieval_documents.repository_owner",
                "retrieval_documents.repository_name",
            ],
            name="fk_chunk_document",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("chunk_id", "source_text_sha256", name="uq_chunk_hash"),
        CheckConstraint("chunk_index >= 0", name="ck_chunk_index"),
        Index("ix_chunks_repository", "repository_owner", "repository_name"),
    )


class EmbeddingRecord(Base):
    __tablename__ = "chunk_embeddings"
    chunk_id: Mapped[str] = mapped_column(Text, primary_key=True)
    model_name: Mapped[str] = mapped_column(Text, primary_key=True)
    dimension: Mapped[int] = mapped_column(Integer)
    normalized: Mapped[bool]
    source_text_sha256: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(VECTOR(EMBEDDING_DIMENSION))
    # Preserve original float values for exact domain round trips; vector is float32.
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    __table_args__ = (
        ForeignKeyConstraint(
            ["chunk_id", "source_text_sha256"],
            ["retrieval_chunks.chunk_id", "retrieval_chunks.source_text_sha256"],
            name="fk_embedding_chunk_hash",
            ondelete="RESTRICT",
        ),
        CheckConstraint("dimension = 384", name="ck_embedding_dimension"),
        CheckConstraint("normalized", name="ck_embedding_normalized"),
        CheckConstraint(
            "abs((embedding <#> embedding) + 1) <= 0.0003",
            name="ck_embedding_unit_norm",
        ),
        Index("ix_embeddings_model", "model_name"),
    )
