import math
from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from reporecall.models import ChunkEmbedding, VectorIndexManifest, VectorIndexMatch
from reporecall.retrieval.exceptions import (
    VectorIndexCompatibilityError,
    VectorIndexError,
)

_NORMALIZATION_RTOL = 1e-4
_NORMALIZATION_ATOL = 1e-5


class _FaissIndex(Protocol):
    ntotal: int

    def add(self, vectors: NDArray[np.float32]) -> None: ...

    def search(
        self,
        vectors: NDArray[np.float32],
        k: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]: ...


class _FaissModule(Protocol):
    IndexFlatIP: Callable[[int], _FaissIndex]


class FaissVectorIndex:
    """Exact inner-product index over normalized chunk embeddings."""

    def __init__(self) -> None:
        self._faiss = _load_faiss_module()
        self._index: _FaissIndex | None = None
        self._manifest: VectorIndexManifest | None = None
        self._vectors: NDArray[np.float32] | None = None

    @property
    def manifest(self) -> VectorIndexManifest | None:
        """Return immutable metadata for the currently built index."""

        return self._manifest

    @property
    def vector_count(self) -> int:
        """Return the number of indexed chunk vectors."""

        return 0 if self._manifest is None else self._manifest.vector_count

    @property
    def is_empty(self) -> bool:
        """Return whether the index contains no vectors."""

        return self.vector_count == 0

    def build(self, embeddings: Sequence[ChunkEmbedding]) -> None:
        """Replace index contents using embedding input order as FAISS row order."""

        records = list(embeddings)
        if not records:
            self._index = None
            self._manifest = None
            self._vectors = None
            return

        manifest = _manifest_for(records)
        matrix = _embedding_matrix(records, dimension=manifest.dimension)

        try:
            index = self._faiss.IndexFlatIP(manifest.dimension)
            index.add(matrix)
        except Exception as exc:
            raise VectorIndexError("FAISS failed while building the vector index.") from exc

        if index.ntotal != manifest.vector_count:
            raise VectorIndexError(
                "FAISS vector count does not match the index manifest."
            )

        self._index = index
        self._manifest = manifest
        self._vectors = matrix.copy()

    def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int,
        candidate_chunk_ids: Sequence[str] | None = None,
    ) -> list[VectorIndexMatch]:
        """Return exact inner-product matches over all or selected chunk IDs."""

        if k <= 0:
            raise VectorIndexError("Search result count k must be greater than zero.")
        if self._index is None or self._manifest is None:
            return []

        candidate_rows: tuple[int, ...] | None = None
        if candidate_chunk_ids is not None:
            candidate_rows = _candidate_rows(self._manifest, candidate_chunk_ids)
            if not candidate_rows:
                return []

        query = _query_matrix(query_vector, dimension=self._manifest.dimension)
        if candidate_rows is None or len(candidate_rows) == self._manifest.vector_count:
            global_rows = tuple(range(self._manifest.vector_count))
            result_count = min(k, self._manifest.vector_count)
            return self._search_index(
                self._index,
                query,
                global_rows=global_rows,
                result_count=result_count,
                limit=k,
            )

        if self._vectors is None:
            raise VectorIndexError("FAISS candidate search has no retained vector matrix.")
        candidate_matrix = np.ascontiguousarray(
            self._vectors[np.asarray(candidate_rows, dtype=np.intp)]
        )
        try:
            candidate_index = self._faiss.IndexFlatIP(self._manifest.dimension)
            candidate_index.add(candidate_matrix)
        except Exception as exc:
            raise VectorIndexError("FAISS failed while building a candidate index.") from exc
        if candidate_index.ntotal != len(candidate_rows):
            raise VectorIndexError(
                "FAISS candidate vector count does not match the selected rows."
            )

        return self._search_index(
            candidate_index,
            query,
            global_rows=candidate_rows,
            result_count=len(candidate_rows),
            limit=k,
        )

    def _search_index(
        self,
        index: _FaissIndex,
        query: NDArray[np.float32],
        *,
        global_rows: tuple[int, ...],
        result_count: int,
        limit: int,
    ) -> list[VectorIndexMatch]:
        if self._manifest is None:
            raise VectorIndexError("FAISS search has no index manifest.")
        try:
            scores, row_ids = index.search(query, result_count)
        except Exception as exc:
            raise VectorIndexError("FAISS failed while searching the vector index.") from exc

        candidates: list[tuple[str, str, float]] = []
        for local_row_value, score_value in zip(row_ids[0], scores[0], strict=True):
            local_row = int(local_row_value)
            if local_row < 0:
                continue
            if local_row >= len(global_rows):
                raise VectorIndexError("FAISS returned an invalid vector row ID.")

            global_row = global_rows[local_row]
            score = float(score_value)
            if not math.isfinite(score):
                raise VectorIndexError("FAISS returned a non-finite similarity score.")
            candidates.append(
                (
                    self._manifest.chunk_ids[global_row],
                    self._manifest.source_text_sha256[global_row],
                    score,
                )
            )

        candidates.sort(key=lambda candidate: (-candidate[2], candidate[0]))
        return [
            VectorIndexMatch(
                chunk_id=chunk_id,
                source_text_sha256=source_hash,
                score=score,
                rank=rank,
            )
            for rank, (chunk_id, source_hash, score) in enumerate(
                candidates[:limit],
                start=1,
            )
        ]


def _candidate_rows(
    manifest: VectorIndexManifest,
    candidate_chunk_ids: Sequence[str],
) -> tuple[int, ...]:
    candidate_ids = tuple(candidate_chunk_ids)
    if any(not chunk_id.strip() for chunk_id in candidate_ids):
        raise VectorIndexError("Candidate chunk IDs must not be blank.")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise VectorIndexError("Candidate chunk IDs must be unique.")

    indexed_ids = set(manifest.chunk_ids)
    missing_ids = sorted(set(candidate_ids) - indexed_ids)
    if missing_ids:
        raise VectorIndexError(
            f"Candidate chunk ID {missing_ids[0]!r} is missing from the vector index."
        )

    selected = set(candidate_ids)
    return tuple(
        row_id
        for row_id, chunk_id in enumerate(manifest.chunk_ids)
        if chunk_id in selected
    )


def _manifest_for(embeddings: Sequence[ChunkEmbedding]) -> VectorIndexManifest:
    first = embeddings[0]
    if not first.normalized:
        raise VectorIndexCompatibilityError(
            "FAISS inner-product retrieval requires normalized embeddings."
        )

    seen_chunk_ids: set[str] = set()
    for embedding in embeddings:
        if embedding.chunk_id in seen_chunk_ids:
            raise VectorIndexError(
                f"Duplicate chunk ID {embedding.chunk_id!r} cannot be indexed."
            )
        seen_chunk_ids.add(embedding.chunk_id)

        if embedding.model_name != first.model_name:
            raise VectorIndexCompatibilityError(
                "All indexed embeddings must use the same model."
            )
        if embedding.dimension != first.dimension:
            raise VectorIndexCompatibilityError(
                "All indexed embeddings must have the same dimension."
            )
        if embedding.normalized != first.normalized:
            raise VectorIndexCompatibilityError(
                "All indexed embeddings must use the same normalization setting."
            )

    return VectorIndexManifest(
        model_name=first.model_name,
        dimension=first.dimension,
        normalized=first.normalized,
        vector_count=len(embeddings),
        chunk_ids=tuple(embedding.chunk_id for embedding in embeddings),
        source_text_sha256=tuple(
            embedding.source_text_sha256 for embedding in embeddings
        ),
    )


def _embedding_matrix(
    embeddings: Sequence[ChunkEmbedding],
    *,
    dimension: int,
) -> NDArray[np.float32]:
    with np.errstate(over="ignore", invalid="ignore"):
        matrix = np.asarray(
            [embedding.vector for embedding in embeddings],
            dtype=np.float32,
        )
    matrix = np.ascontiguousarray(matrix)
    if matrix.shape != (len(embeddings), dimension):
        raise VectorIndexCompatibilityError(
            "Embedding vectors do not match the declared index dimension."
        )
    _require_finite(matrix, subject="Indexed embeddings")
    _require_unit_vectors(matrix, subject="Indexed embeddings")
    return matrix


def _query_matrix(
    query_vector: Sequence[float],
    *,
    dimension: int,
) -> NDArray[np.float32]:
    if len(query_vector) != dimension:
        raise VectorIndexCompatibilityError(
            "Query vector dimension does not match the FAISS index dimension."
        )
    with np.errstate(over="ignore", invalid="ignore"):
        vector = np.asarray(tuple(query_vector), dtype=np.float32)
    vector = np.ascontiguousarray(vector)
    if vector.shape != (dimension,):
        raise VectorIndexCompatibilityError("Query embedding must be one vector.")
    query = vector.reshape(1, dimension)
    _require_finite(query, subject="Query embedding")
    _require_unit_vectors(query, subject="Query embedding")
    return query


def _require_finite(matrix: NDArray[np.float32], *, subject: str) -> None:
    if not bool(np.isfinite(matrix).all()):
        raise VectorIndexCompatibilityError(f"{subject} must contain finite values.")


def _require_unit_vectors(matrix: NDArray[np.float32], *, subject: str) -> None:
    norms = np.linalg.norm(matrix, axis=1)
    if not bool(
        np.allclose(
            norms,
            1.0,
            rtol=_NORMALIZATION_RTOL,
            atol=_NORMALIZATION_ATOL,
        )
    ):
        raise VectorIndexCompatibilityError(
            f"{subject} must contain L2-normalized vectors."
        )


def _load_faiss_module() -> _FaissModule:
    try:
        module = import_module("faiss")
    except ModuleNotFoundError as exc:
        raise VectorIndexError(
            "FAISS retrieval requires the 'faiss-cpu' dependency."
        ) from exc
    return cast(_FaissModule, module)
