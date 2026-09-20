"""Exact persistent dense retrieval using pgvector negative inner product."""

import math

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from reporecall.embeddings import EmbeddingBackend
from reporecall.models import MetadataFilter, VectorSearchHit
from reporecall.persistence.exceptions import (
    PersistenceIntegrityError,
    PersistenceQueryError,
)
from reporecall.persistence.filters import metadata_predicates
from reporecall.persistence.mapping import (
    chunk_from_record,
    embedding_from_record,
    validate_embedding_parent,
    validate_vector,
)
from reporecall.persistence.orm import EMBEDDING_DIMENSION, ChunkRecord, EmbeddingRecord


class PostgresVectorRetriever:
    """Search one explicit embedding model with filtering before exact top-k.

    Model selection is explicit when multiple model corpora coexist. A nonempty
    store lacking the requested model is an error, not an empty result.
    """

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        embedding_backend: EmbeddingBackend,
        model_name: str,
    ) -> None:
        if not model_name.strip():
            raise PersistenceIntegrityError("Embedding model name must not be blank.")
        self.session_factory = session_factory
        self.backend = embedding_backend
        self.model_name = model_name

    def search(
        self, query: str, *, k: int = 5, metadata_filter: MetadataFilter | None = None
    ) -> list[VectorSearchHit]:
        """Embed the unchanged raw query; return positive similarity and real ranks."""
        if not query.strip() or isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise PersistenceQueryError(
                "Search needs a nonblank query and a positive integer k."
            )
        if self.backend.model_name != self.model_name or not self.backend.normalized:
            raise PersistenceIntegrityError(
                "Query backend must match the configured model and be normalized."
            )
        predicates = metadata_predicates(metadata_filter)
        try:
            with self.session_factory() as session:
                models = session.execute(
                    select(
                        EmbeddingRecord.model_name,
                        EmbeddingRecord.dimension,
                        EmbeddingRecord.normalized,
                    ).distinct()
                ).all()
                if not models:
                    return []
                selected = [row for row in models if row.model_name == self.model_name]
                if not selected or any(
                    row.dimension != EMBEDDING_DIMENSION or not row.normalized
                    for row in selected
                ):
                    raise PersistenceIntegrityError(
                        "Stored model/dimension/normalization is incompatible with the query backend."
                    )
                eligible = (
                    select(EmbeddingRecord, ChunkRecord)
                    .join(ChunkRecord, EmbeddingRecord.chunk_id == ChunkRecord.chunk_id)
                    .where(EmbeddingRecord.model_name == self.model_name, *predicates)
                )
                if not session.scalar(
                    select(func.count()).select_from(eligible.subquery())
                ):
                    return []
                vectors = self.backend.embed([query])
                if len(vectors) != 1:
                    raise PersistenceIntegrityError(
                        "Query backend must return exactly one vector."
                    )
                vector = validate_vector(vectors[0])
                distance = EmbeddingRecord.embedding.max_inner_product(vector)
                rows = session.execute(
                    eligible.add_columns((-distance).label("score"))
                    .order_by(distance, ChunkRecord.chunk_id.collate("C"))
                    .limit(k)
                ).all()
                hits = []
                for rank, (embedding_row, chunk_row, score) in enumerate(rows, 1):
                    chunk = chunk_from_record(chunk_row)
                    embedding = embedding_from_record(embedding_row)
                    validate_embedding_parent(embedding, chunk)
                    if not math.isfinite(score):
                        raise PersistenceIntegrityError(
                            "Database returned a nonfinite similarity score."
                        )
                    hits.append(
                        VectorSearchHit(chunk=chunk, score=float(score), rank=rank)
                    )
                return hits
        except SQLAlchemyError as exc:
            raise PersistenceQueryError(
                "PostgreSQL exact vector search failed."
            ) from exc
