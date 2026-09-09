import pytest
from pydantic import ValidationError

from reporecall.github import GitHubRepository
from reporecall.models import (
    EventMetadata,
    RetrievalChunk,
    RetrievalSectionType,
    VectorIndexManifest,
    VectorIndexMatch,
    VectorSearchHit,
)


def test_vector_index_manifest_is_immutable_and_serializable():
    manifest = _manifest()

    assert manifest.model_dump(mode="json") == {
        "schema_version": 1,
        "model_name": "test/model",
        "dimension": 2,
        "normalized": True,
        "vector_count": 2,
        "chunk_ids": ["chunk-a", "chunk-b"],
        "source_text_sha256": ["a" * 64, "b" * 64],
    }
    with pytest.raises(ValidationError, match="frozen"):
        manifest.dimension = 3


@pytest.mark.parametrize(
    "update,error_message",
    [
        ({"vector_count": 3}, "number of chunk IDs"),
        ({"source_text_sha256": ("a" * 64,)}, "number of source hashes"),
        ({"chunk_ids": ("chunk-a", "chunk-a")}, "must be unique"),
        ({"chunk_ids": ("chunk-a", "   ")}, "must not be blank"),
        ({"model_name": "   "}, "must not be blank"),
    ],
)
def test_vector_index_manifest_rejects_invalid_metadata(
    update: dict[str, object],
    error_message: str,
):
    values = _manifest().model_dump()
    values.update(update)

    with pytest.raises(ValidationError, match=error_message):
        VectorIndexManifest(**values)


def test_vector_index_manifest_rejects_invalid_source_hash():
    values = _manifest().model_dump()
    values["source_text_sha256"] = ("invalid", "b" * 64)

    with pytest.raises(ValidationError):
        VectorIndexManifest(**values)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_vector_index_match_rejects_nonfinite_score(score: float):
    with pytest.raises(ValidationError):
        VectorIndexMatch(
            chunk_id="chunk-a",
            source_text_sha256="a" * 64,
            score=score,
            rank=1,
        )


@pytest.mark.parametrize("rank", [0, -1])
def test_vector_index_match_rejects_nonpositive_rank(rank: int):
    with pytest.raises(ValidationError):
        VectorIndexMatch(
            chunk_id="chunk-a",
            source_text_sha256="a" * 64,
            score=1.0,
            rank=rank,
        )


def test_vector_search_hit_preserves_chunk_and_is_immutable():
    chunk = _chunk()
    hit = VectorSearchHit(chunk=chunk, score=0.75, rank=1)

    assert hit.chunk == chunk
    assert hit.score == 0.75
    assert hit.rank == 1
    with pytest.raises(ValidationError, match="frozen"):
        hit.rank = 2


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_vector_search_hit_rejects_nonfinite_score(score: float):
    with pytest.raises(ValidationError):
        VectorSearchHit(chunk=_chunk(), score=score, rank=1)


@pytest.mark.parametrize("rank", [0, -1])
def test_vector_search_hit_rejects_nonpositive_rank(rank: int):
    with pytest.raises(ValidationError):
        VectorSearchHit(chunk=_chunk(), score=1.0, rank=rank)


def _manifest() -> VectorIndexManifest:
    return VectorIndexManifest(
        model_name="test/model",
        dimension=2,
        normalized=True,
        vector_count=2,
        chunk_ids=("chunk-a", "chunk-b"),
        source_text_sha256=("a" * 64, "b" * 64),
    )


def _chunk() -> RetrievalChunk:
    repository = GitHubRepository(owner="owner", name="repo")
    return RetrievalChunk(
        chunk_id="chunk-a",
        document_id="document-1",
        event_id="event-1",
        repository=repository,
        section_id="issue-10",
        section_type=RetrievalSectionType.ISSUE,
        chunk_index=0,
        content="Connection leak",
        text="Repository: owner/repo\n\nConnection leak",
        artifact=None,
        metadata=EventMetadata(event_id="event-1", repository=repository),
    )
