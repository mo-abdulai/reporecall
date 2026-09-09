import hashlib
from collections.abc import Mapping

from reporecall.models import KeywordSearchHit, MetadataFilter, RetrievalChunk
from reporecall.retrieval.bm25_index import BM25Index
from reporecall.retrieval.exceptions import BM25IndexError, KeywordRetrievalError
from reporecall.retrieval.metadata_filter import filter_chunk_ids


class KeywordRetriever:
    """Retrieve original chunks through independent BM25 lexical ranking."""

    def __init__(
        self,
        *,
        index: BM25Index,
        chunks: Mapping[str, RetrievalChunk],
    ) -> None:
        self.index = index
        self._chunks = dict(chunks)
        for chunk_id, chunk in self._chunks.items():
            if chunk_id != chunk.chunk_id:
                raise BM25IndexError(
                    "Chunk lookup keys must match their RetrievalChunk chunk IDs."
                )

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[KeywordSearchHit]:
        """Return positive-score lexical hits after optional metadata filtering."""

        if not query.strip():
            raise KeywordRetrievalError("Keyword search query must not be blank.")
        if k <= 0:
            raise KeywordRetrievalError(
                "Keyword search result count k must be greater than zero."
            )
        if self.index.is_empty:
            return []

        missing_chunk_ids = [
            chunk_id for chunk_id in self.index.chunk_ids if chunk_id not in self._chunks
        ]
        if missing_chunk_ids:
            raise BM25IndexError(
                f"Indexed chunk {missing_chunk_ids[0]!r} is missing from the chunk lookup."
            )

        candidate_chunk_ids: tuple[str, ...] | None = None
        if metadata_filter is not None and not metadata_filter.is_empty:
            indexed_ids = set(self.index.chunk_ids)
            ordered_chunks = [self._chunks[chunk_id] for chunk_id in self.index.chunk_ids]
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
                raise BM25IndexError(
                    f"Candidate chunk ID {unindexed_candidate_ids[0]!r} "
                    "is missing from the BM25 index."
                )

        query_tokens = self.index.tokenizer.tokenize(query)
        if not query_tokens:
            return []
        matches = self.index.search(
            query_tokens,
            k=k,
            candidate_chunk_ids=candidate_chunk_ids,
        )

        hits: list[KeywordSearchHit] = []
        for match in matches:
            chunk = self._chunks.get(match.chunk_id)
            if chunk is None:
                raise BM25IndexError(
                    f"Indexed chunk {match.chunk_id!r} is missing from the chunk lookup."
                )
            source_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
            if source_hash != match.source_text_sha256:
                raise BM25IndexError(
                    f"Indexed chunk {match.chunk_id!r} has stale source text."
                )
            hits.append(
                KeywordSearchHit(chunk=chunk, score=match.score, rank=match.rank)
            )
        return hits
