import pytest
from pydantic import ValidationError

from reporecall.github import GitHubRepository
from reporecall.models import ChunkEmbedding, RetrievalSectionType


def test_chunk_embedding_is_immutable_and_serializable():
    embedding = ChunkEmbedding(**_embedding_values())

    assert embedding.vector == (0.1, 0.2, 0.3)
    assert embedding.model_dump(mode="json")["vector"] == [0.1, 0.2, 0.3]
    with pytest.raises(ValidationError, match="frozen"):
        embedding.dimension = 4


@pytest.mark.parametrize(
    "field_name",
    ["chunk_id", "document_id", "event_id", "section_id", "model_name"],
)
@pytest.mark.parametrize("value", ["", "   "])
def test_chunk_embedding_rejects_blank_required_identity(
    field_name: str,
    value: str,
):
    values = _embedding_values()
    values[field_name] = value

    with pytest.raises(ValidationError):
        ChunkEmbedding(**values)


@pytest.mark.parametrize("dimension", [0, -1])
def test_chunk_embedding_rejects_nonpositive_dimension(dimension: int):
    values = _embedding_values()
    values["dimension"] = dimension

    with pytest.raises(ValidationError):
        ChunkEmbedding(**values)


def test_chunk_embedding_rejects_vector_dimension_mismatch_and_empty_vector():
    values = _embedding_values()
    values["dimension"] = 2
    with pytest.raises(ValidationError, match="length must match"):
        ChunkEmbedding(**values)

    values = _embedding_values()
    values["vector"] = []
    with pytest.raises(ValidationError):
        ChunkEmbedding(**values)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_chunk_embedding_rejects_nonfinite_values(invalid_value: float):
    values = _embedding_values()
    values["vector"] = [0.1, invalid_value, 0.3]

    with pytest.raises(ValidationError, match="must be finite"):
        ChunkEmbedding(**values)


@pytest.mark.parametrize(
    "source_hash",
    ["", "a" * 63, "A" * 64, "z" * 64, "a" * 65],
)
def test_chunk_embedding_rejects_malformed_source_hash(source_hash: str):
    values = _embedding_values()
    values["source_text_sha256"] = source_hash

    with pytest.raises(ValidationError):
        ChunkEmbedding(**values)


def _embedding_values() -> dict[str, object]:
    return {
        "chunk_id": "chunk-1",
        "document_id": "document-1",
        "event_id": "event-1",
        "repository": GitHubRepository(owner="owner", name="repo"),
        "section_id": "issue-10",
        "section_type": RetrievalSectionType.ISSUE,
        "model_name": "sentence-transformers/test-model",
        "dimension": 3,
        "normalized": True,
        "source_text_sha256": "a" * 64,
        "vector": [0.1, 0.2, 0.3],
    }
