import hashlib
from collections.abc import Mapping

from reporecall.embeddings import EmbeddingBackend
from reporecall.models import RetrievalChunk, VectorSearchHit
from reporecall.retrieval.exceptions import (
    RetrievalError,
    VectorIndexCompatibilityError,
    VectorIndexError,
)
from reporecall.retrieval.faiss_index import FaissVectorIndex


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

    def search(self, query: str, *, k: int = 5) -> list[VectorSearchHit]:
        """Return top-k chunks for one unmodified natural-language query."""

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

        query_vectors = self.backend.embed([query])
        if len(query_vectors) != 1:
            raise RetrievalError(
                "Embedding backend must return exactly one vector for one query."
            )

        matches = self.index.search(query_vectors[0], k=k)
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
