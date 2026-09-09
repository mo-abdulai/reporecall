import hashlib
import math

import numpy as np
import pytest

import reporecall.retrieval.faiss_index as faiss_index_module
from reporecall.github import GitHubRepository
from reporecall.models import ChunkEmbedding
from reporecall.retrieval import (
    FaissVectorIndex,
    VectorIndexCompatibilityError,
    VectorIndexError,
)

SQRT_HALF = math.sqrt(0.5)


def test_faiss_is_loaded_only_when_index_is_instantiated():
    assert "faiss" not in vars(faiss_index_module)


def test_missing_faiss_dependency_has_actionable_error(monkeypatch):
    def missing_module(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(faiss_index_module, "import_module", missing_module)

    with pytest.raises(VectorIndexError, match="faiss-cpu"):
        FaissVectorIndex()


def test_index_flat_ip_exact_ranking_and_scores():
    index = FaissVectorIndex()
    index.build(
        [
            _embedding("chunk-a", (1.0, 0.0)),
            _embedding("chunk-b", (0.0, 1.0)),
            _embedding("chunk-c", (SQRT_HALF, SQRT_HALF)),
        ]
    )

    matches = index.search((1.0, 0.0), k=3)

    assert [match.chunk_id for match in matches] == [
        "chunk-a",
        "chunk-c",
        "chunk-b",
    ]
    assert [match.rank for match in matches] == [1, 2, 3]
    assert [match.score for match in matches] == pytest.approx(
        [1.0, SQRT_HALF, 0.0]
    )
    assert all(math.isfinite(match.score) for match in matches)


def test_manifest_preserves_input_row_order_and_source_hashes():
    embeddings = [
        _embedding("chunk-c", (1.0, 0.0), text="C"),
        _embedding("chunk-a", (0.0, 1.0), text="A"),
    ]
    index = FaissVectorIndex()

    index.build(embeddings)

    assert index.manifest is not None
    assert index.manifest.model_name == "test/model"
    assert index.manifest.dimension == 2
    assert index.manifest.normalized is True
    assert index.manifest.vector_count == 2
    assert index.manifest.chunk_ids == ("chunk-c", "chunk-a")
    assert index.manifest.source_text_sha256 == tuple(
        embedding.source_text_sha256 for embedding in embeddings
    )


def test_empty_build_resets_index_and_searches_without_faiss_errors():
    index = FaissVectorIndex()
    index.build([_embedding("chunk-a", (1.0, 0.0))])

    index.build([])

    assert index.is_empty is True
    assert index.vector_count == 0
    assert index.manifest is None
    assert index.search((), k=5) == []


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_nonpositive_k(k: int):
    index = FaissVectorIndex()
    index.build([_embedding("chunk-a", (1.0, 0.0))])

    with pytest.raises(VectorIndexError, match="greater than zero"):
        index.search((1.0, 0.0), k=k)


def test_k_larger_than_index_returns_only_available_vectors():
    index = FaissVectorIndex()
    index.build(
        [
            _embedding("chunk-a", (1.0, 0.0)),
            _embedding("chunk-b", (0.0, 1.0)),
        ]
    )

    assert len(index.search((1.0, 0.0), k=10)) == 2


def test_index_rejects_mixed_models():
    with pytest.raises(VectorIndexCompatibilityError, match="same model"):
        FaissVectorIndex().build(
            [
                _embedding("chunk-a", (1.0, 0.0), model_name="model-a"),
                _embedding("chunk-b", (0.0, 1.0), model_name="model-b"),
            ]
        )


def test_index_rejects_mixed_dimensions():
    with pytest.raises(VectorIndexCompatibilityError, match="same dimension"):
        FaissVectorIndex().build(
            [
                _embedding("chunk-a", (1.0, 0.0)),
                _embedding("chunk-b", (0.0, 0.0, 1.0)),
            ]
        )


def test_index_rejects_non_normalized_configuration():
    with pytest.raises(VectorIndexCompatibilityError, match="requires normalized"):
        FaissVectorIndex().build(
            [_embedding("chunk-a", (1.0, 0.0), normalized=False)]
        )


def test_index_rejects_mixed_normalization_settings():
    with pytest.raises(VectorIndexCompatibilityError, match="normalization setting"):
        FaissVectorIndex().build(
            [
                _embedding("chunk-a", (1.0, 0.0)),
                _embedding("chunk-b", (0.0, 1.0), normalized=False),
            ]
        )


def test_index_rejects_duplicate_chunk_ids_even_when_vectors_match():
    with pytest.raises(VectorIndexError, match="Duplicate chunk ID"):
        FaissVectorIndex().build(
            [
                _embedding("chunk-a", (1.0, 0.0)),
                _embedding("chunk-a", (1.0, 0.0)),
            ]
        )


def test_index_rejects_vectors_marked_normalized_that_are_not_unit_length():
    with pytest.raises(VectorIndexCompatibilityError, match="L2-normalized"):
        FaissVectorIndex().build([_embedding("chunk-a", (3.0, 4.0))])


def test_query_dimension_must_match_index():
    index = FaissVectorIndex()
    index.build([_embedding("chunk-a", (1.0, 0.0))])

    with pytest.raises(VectorIndexCompatibilityError, match="dimension"):
        index.search((1.0, 0.0, 0.0), k=1)


@pytest.mark.parametrize(
    "query",
    [(float("nan"), 0.0), (float("inf"), 0.0), (float("-inf"), 0.0)],
)
def test_query_vector_must_be_finite(query: tuple[float, float]):
    index = FaissVectorIndex()
    index.build([_embedding("chunk-a", (1.0, 0.0))])

    with pytest.raises(VectorIndexCompatibilityError, match="finite"):
        index.search(query, k=1)


def test_query_vector_must_be_unit_normalized():
    index = FaissVectorIndex()
    index.build([_embedding("chunk-a", (1.0, 0.0))])

    with pytest.raises(VectorIndexCompatibilityError, match="L2-normalized"):
        index.search((3.0, 4.0), k=1)


def test_float32_overflow_is_rejected_before_faiss_search():
    index = FaissVectorIndex()
    index.build([_embedding("chunk-a", (1.0, 0.0))])

    with pytest.raises(VectorIndexCompatibilityError, match="finite"):
        index.search((1e100, 0.0), k=1)


def test_tied_scores_use_chunk_identity_for_stable_ordering():
    index = FaissVectorIndex()
    index.build(
        [
            _embedding("chunk-b", (1.0, 0.0)),
            _embedding("chunk-a", (1.0, 0.0)),
        ]
    )

    first = index.search((1.0, 0.0), k=2)
    second = index.search((1.0, 0.0), k=2)

    assert first == second
    assert [match.chunk_id for match in first] == ["chunk-a", "chunk-b"]


def test_faiss_sentinel_rows_are_not_returned():
    index = FaissVectorIndex()
    index.build(
        [
            _embedding("chunk-a", (1.0, 0.0)),
            _embedding("chunk-b", (0.0, 1.0)),
        ]
    )
    index._index = _FakeFaissIndex(
        scores=np.asarray([[1.0, -math.inf]], dtype=np.float32),
        row_ids=np.asarray([[0, -1]], dtype=np.int64),
    )

    matches = index.search((1.0, 0.0), k=2)

    assert [match.chunk_id for match in matches] == ["chunk-a"]


def test_nonfinite_faiss_scores_are_rejected():
    index = FaissVectorIndex()
    index.build([_embedding("chunk-a", (1.0, 0.0))])
    index._index = _FakeFaissIndex(
        scores=np.asarray([[math.nan]], dtype=np.float32),
        row_ids=np.asarray([[0]], dtype=np.int64),
    )

    with pytest.raises(VectorIndexError, match="non-finite"):
        index.search((1.0, 0.0), k=1)


def test_invalid_faiss_row_is_rejected():
    index = FaissVectorIndex()
    index.build([_embedding("chunk-a", (1.0, 0.0))])
    index._index = _FakeFaissIndex(
        scores=np.asarray([[1.0]], dtype=np.float32),
        row_ids=np.asarray([[5]], dtype=np.int64),
    )

    with pytest.raises(VectorIndexError, match="invalid vector row"):
        index.search((1.0, 0.0), k=1)


def test_build_and_search_do_not_mutate_embeddings():
    embeddings = [
        _embedding("chunk-a", (1.0, 0.0)),
        _embedding("chunk-b", (0.0, 1.0)),
    ]
    original = [embedding.model_dump() for embedding in embeddings]
    index = FaissVectorIndex()

    index.build(embeddings)
    index.search((1.0, 0.0), k=2)

    assert [embedding.model_dump() for embedding in embeddings] == original


class _FakeFaissIndex:
    def __init__(self, *, scores: np.ndarray, row_ids: np.ndarray) -> None:
        self.ntotal = row_ids.shape[1]
        self._scores = scores
        self._row_ids = row_ids

    def add(self, vectors: np.ndarray) -> None:
        raise AssertionError("not used")

    def search(self, vectors: np.ndarray, k: int):
        return self._scores, self._row_ids


def _embedding(
    chunk_id: str,
    vector: tuple[float, ...],
    *,
    model_name: str = "test/model",
    normalized: bool = True,
    text: str | None = None,
) -> ChunkEmbedding:
    source_text = text or f"text for {chunk_id}"
    return ChunkEmbedding(
        chunk_id=chunk_id,
        document_id="document-1",
        event_id="event-1",
        repository=GitHubRepository(owner="owner", name="repo"),
        section_id=f"section-{chunk_id}",
        section_type="issue",
        model_name=model_name,
        dimension=len(vector),
        normalized=normalized,
        source_text_sha256=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        vector=vector,
    )
