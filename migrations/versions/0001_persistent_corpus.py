"""Canonical corpus and exact 384-dimensional pgvector storage.

Dimension changes require an explicit migration and reindex. No approximate
vector indexes are created. Downgrade retains the potentially shared extension.
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import VECTOR
from sqlalchemy.dialects import postgresql

revision = "0001_persistent_corpus"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    try:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
    except sa.exc.DBAPIError as exc:
        raise RuntimeError(
            "Unable to enable pgvector. Install the extension on the PostgreSQL server "
            "and run migrations with a role permitted to CREATE EXTENSION vector."
        ) from exc
    op.create_table(
        "engineering_events",
        sa.Column("event_id", sa.Text(), primary_key=True),
        sa.Column("repository_owner", sa.Text(), nullable=False),
        sa.Column("repository_name", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.UniqueConstraint(
            "event_id",
            "repository_owner",
            "repository_name",
            name="uq_event_repository",
        ),
    )
    op.create_table(
        "retrieval_documents",
        sa.Column("document_id", sa.Text(), primary_key=True),
        sa.Column("event_id", sa.Text(), nullable=False),
        sa.Column("repository_owner", sa.Text(), nullable=False),
        sa.Column("repository_name", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ["event_id", "repository_owner", "repository_name"],
            [
                "engineering_events.event_id",
                "engineering_events.repository_owner",
                "engineering_events.repository_name",
            ],
            name="fk_document_event",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "document_id",
            "event_id",
            "repository_owner",
            "repository_name",
            name="uq_document_provenance",
        ),
    )
    op.create_table(
        "retrieval_chunks",
        sa.Column("chunk_id", sa.Text(), primary_key=True),
        sa.Column("document_id", sa.Text(), nullable=False),
        sa.Column("event_id", sa.Text(), nullable=False),
        sa.Column("repository_owner", sa.Text(), nullable=False),
        sa.Column("repository_name", sa.Text(), nullable=False),
        sa.Column("section_id", sa.Text(), nullable=False),
        sa.Column("section_type", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("artifact_type", sa.Text(), nullable=True),
        sa.Column("artifact_id", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("source_text_sha256", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.Column("filter_metadata", postgresql.JSONB(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
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
        sa.UniqueConstraint("chunk_id", "source_text_sha256", name="uq_chunk_hash"),
        sa.CheckConstraint("chunk_index >= 0", name="ck_chunk_index"),
    )
    op.create_table(
        "chunk_embeddings",
        sa.Column("chunk_id", sa.Text(), primary_key=True),
        sa.Column("model_name", sa.Text(), primary_key=True),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("normalized", sa.Boolean(), nullable=False),
        sa.Column("source_text_sha256", sa.Text(), nullable=False),
        sa.Column("embedding", VECTOR(384), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ["chunk_id", "source_text_sha256"],
            ["retrieval_chunks.chunk_id", "retrieval_chunks.source_text_sha256"],
            name="fk_embedding_chunk_hash",
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint("dimension = 384", name="ck_embedding_dimension"),
        sa.CheckConstraint("normalized", name="ck_embedding_normalized"),
        sa.CheckConstraint(
            "abs((embedding <#> embedding) + 1) <= 0.0003",
            name="ck_embedding_unit_norm",
        ),
    )
    for table, name in (
        ("engineering_events", "events"),
        ("retrieval_documents", "documents"),
        ("retrieval_chunks", "chunks"),
    ):
        op.create_index(
            f"ix_{name}_repository", table, ["repository_owner", "repository_name"]
        )
    for table, column in (
        ("retrieval_documents", "event_id"),
        ("retrieval_chunks", "document_id"),
        ("retrieval_chunks", "event_id"),
        ("retrieval_chunks", "section_type"),
    ):
        op.create_index(f"ix_{table}_{column}", table, [column])
    op.create_index("ix_embeddings_model", "chunk_embeddings", ["model_name"])


def downgrade() -> None:
    for table in (
        "chunk_embeddings",
        "retrieval_chunks",
        "retrieval_documents",
        "engineering_events",
    ):
        op.drop_table(table)
