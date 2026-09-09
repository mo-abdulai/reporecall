import hashlib
from collections.abc import Mapping

from reporecall.embeddings import EmbeddingBackend
from reporecall.models import MetadataFilter, RetrievalChunk, VectorSearchHit
from reporecall.retrieval.exceptions import (
    RetrievalError,
    VectorIndexCompatibilityError,
    VectorIndexError,
)
from reporecall.retrieval.faiss_index import FaissVectorIndex
from reporecall.retrieval.metadata_filter import filter_chunk_ids


class VectorRetriever:
    """Embed raw queries and reconstruct ranked chunks from a FAISS index."""

    def __init__(
        self,
        *,
        index: FaissVectorIndex,
        backend: EmbeddingBackend,
        chunks: Mapping[str, RetrievalChunk],
    ) -> None:
        self.index = index
        self.backend = backend
        self._chunks = dict(chunks)
        for chunk_id, chunk in self._chunks.items():
            if chunk_id != chunk.chunk_id:
                raise VectorIndexError(
                    "Chunk lookup keys must match their RetrievalChunk chunk IDs."
                )

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[VectorSearchHit]:
        """Return top-k chunks after optional pre-ranking metadata filtering."""

        if not query.strip():
            raise RetrievalError("Search query must not be blank.")
        if k <= 0:
            raise RetrievalError("Search result count k must be greater than zero.")
        if self.index.is_empty:
            return []

        manifest = self.index.manifest
        if manifest is None:
            raise VectorIndexError("Non-empty vector index has no manifest.")
        if self.backend.model_name != manifest.model_name:
            raise VectorIndexCompatibilityError(
                "Query backend model does not match the indexed embedding model."
            )
        if not self.backend.normalized or not manifest.normalized:
            raise VectorIndexCompatibilityError(
                "Query and indexed embeddings must both be normalized."
            )

        missing_chunk_ids = [
            chunk_id
            for chunk_id in manifest.chunk_ids
            if chunk_id not in self._chunks
        ]
        if missing_chunk_ids:
            raise VectorIndexError(
                f"Indexed chunk {missing_chunk_ids[0]!r} is missing from the chunk lookup."
            )

        candidate_chunk_ids: tuple[str, ...] | None = None
        if metadata_filter is not None and not metadata_filter.is_empty:
            indexed_ids = set(manifest.chunk_ids)
            ordered_chunks = [self._chunks[chunk_id] for chunk_id in manifest.chunk_ids]
            ordered_chunks.extend(
                self._chunks[chunk_id]
                for chunk_id in sorted(set(self._chunks) - indexed_ids)
            )
            candidate_chunk_ids = filter_chunk_ids(ordered_chunks, metadata_filter)
            if not candidate_chunk_ids:
                return []
            unindexed_candidate_ids = sorted(
                set(candidate_chunk_ids) - indexed_ids
            )
            if unindexed_candidate_ids:
                raise VectorIndexError(
                    f"Candidate chunk ID {unindexed_candidate_ids[0]!r} "
                    "is missing from the vector index."
                )

        query_vectors = self.backend.embed([query])
        if len(query_vectors) != 1:
            raise RetrievalError(
                "Embedding backend must return exactly one vector for one query."
            )

        matches = self.index.search(
            query_vectors[0],
            k=k,
            candidate_chunk_ids=candidate_chunk_ids,
        )
        hits: list[VectorSearchHit] = []
        for match in matches:
            chunk = self._chunks.get(match.chunk_id)
            if chunk is None:
                raise VectorIndexError(
                    f"Indexed chunk {match.chunk_id!r} is missing from the chunk lookup."
                )
            source_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
            if source_hash != match.source_text_sha256:
                raise VectorIndexError(
                    f"Indexed chunk {match.chunk_id!r} has stale source text."
                )
            hits.append(VectorSearchHit(chunk=chunk, score=match.score, rank=match.rank))
        return hits
