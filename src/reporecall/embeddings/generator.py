import hashlib
import math
from collections.abc import Sequence

from reporecall.embeddings.backend import EmbeddingBackend
from reporecall.embeddings.exceptions import EmbeddingError, EmbeddingOutputError
from reporecall.models import ChunkEmbedding, RetrievalChunk


class EmbeddingGenerator:
    """Generate validated embedding records from retrieval-ready chunk text."""

    def __init__(self, backend: EmbeddingBackend) -> None:
        self.backend = backend

    def generate(self, chunks: Sequence[RetrievalChunk]) -> list[ChunkEmbedding]:
        """Embed unique chunks once while preserving first-occurrence ordering."""

        unique_chunks = _unique_chunks(chunks)
        if not unique_chunks:
            return []

        texts = [chunk.text for chunk in unique_chunks]
        try:
            raw_vectors = self.backend.embed(texts)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Embedding backend failed.") from exc

        if len(raw_vectors) != len(unique_chunks):
            raise EmbeddingOutputError(
                "Embedding backend returned a different number of vectors than chunks."
            )

        vectors = _validated_vectors(raw_vectors)
        return [
            _embedding_record(
                chunk,
                vector,
                model_name=self.backend.model_name,
                normalized=self.backend.normalized,
            )
            for chunk, vector in zip(unique_chunks, vectors, strict=True)
        ]


def _unique_chunks(chunks: Sequence[RetrievalChunk]) -> list[RetrievalChunk]:
    by_id: dict[str, RetrievalChunk] = {}
    for chunk in chunks:
        existing = by_id.get(chunk.chunk_id)
        if existing is None:
            by_id[chunk.chunk_id] = chunk
            continue
        if existing.text != chunk.text:
            raise EmbeddingOutputError(
                f"Duplicate chunk ID {chunk.chunk_id!r} has conflicting text."
            )
    return list(by_id.values())


def _validated_vectors(
    raw_vectors: Sequence[Sequence[float]],
) -> list[tuple[float, ...]]:
    vectors: list[tuple[float, ...]] = []
    dimension: int | None = None
    for raw_vector in raw_vectors:
        try:
            vector = tuple(float(value) for value in raw_vector)
        except (TypeError, ValueError) as exc:
            raise EmbeddingOutputError(
                "Embedding vectors must contain numeric values."
            ) from exc
        if not vector:
            raise EmbeddingOutputError("Embedding vectors must not be empty.")
        if not all(math.isfinite(value) for value in vector):
            raise EmbeddingOutputError("Embedding vectors must contain finite values.")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise EmbeddingOutputError(
                "All embedding vectors in a batch must have the same dimension."
            )
        vectors.append(vector)
    return vectors


def _embedding_record(
    chunk: RetrievalChunk,
    vector: tuple[float, ...],
    *,
    model_name: str,
    normalized: bool,
) -> ChunkEmbedding:
    return ChunkEmbedding(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        event_id=chunk.event_id,
        repository=chunk.repository,
        section_id=chunk.section_id,
        section_type=chunk.section_type,
        model_name=model_name,
        dimension=len(vector),
        normalized=normalized,
        source_text_sha256=hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
        vector=vector,
    )
