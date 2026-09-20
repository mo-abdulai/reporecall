"""Offline checks of canonical payloads, projections, and embedding integrity."""

import pytest
from pydantic import ValidationError

from reporecall.persistence import (
    DatabaseConfig,
    PersistenceIntegrityError,
    create_database_engine,
)
from reporecall.persistence.mapping import (
    chunk_from_record,
    chunk_to_record,
    document_from_record,
    document_to_record,
    embedding_from_record,
    embedding_to_record,
    event_from_record,
    event_to_record,
    validate_chunk_parent,
    validate_embedding_parent,
    validate_vector,
)
from tests.persistence.fixtures import corpus_data, vector


@pytest.mark.parametrize(
    "attribute,to_record,from_record",
    [
        ("events", event_to_record, event_from_record),
        ("documents", document_to_record, document_from_record),
        ("chunks", chunk_to_record, chunk_from_record),
        ("embeddings", embedding_to_record, embedding_from_record),
    ],
)
def test_roundtrip_and_corrupt_projection(attribute, to_record, from_record):
    item = getattr(corpus_data(), attribute)[0]
    before = item.model_dump(mode="json")
    record = to_record(item)
    assert from_record(record) == item
    assert item.model_dump(mode="json") == before
    identity = {
        "events": "event_id",
        "documents": "document_id",
        "chunks": "chunk_id",
        "embeddings": "model_name",
    }[attribute]
    setattr(record, identity, "corrupt")
    with pytest.raises(PersistenceIntegrityError, match="disagrees"):
        from_record(record)


@pytest.mark.parametrize(
    "changes",
    [
        {"dimension": 768, "vector": (1.0,) + (0.0,) * 767},
        {"normalized": False},
        {"vector": vector(2)},
        {"vector": vector(float("nan"))},
        {"vector": vector(float("inf"))},
        {"vector": vector(1e300)},
    ],
)
def test_reject_bad_embeddings(changes):
    embedding = corpus_data().embeddings[0].model_copy(update=changes)
    with pytest.raises(PersistenceIntegrityError):
        embedding_to_record(embedding)


@pytest.mark.parametrize("values", [[1, 0], [float("inf")] * 384, [0.0] * 384])
def test_query_vector_constraints(values):
    with pytest.raises(PersistenceIntegrityError):
        validate_vector(values)


def test_embedding_identity_and_stale_hash():
    corpus = corpus_data()
    for changes in ({"source_text_sha256": "0" * 64}, {"document_id": "other"}):
        with pytest.raises(PersistenceIntegrityError):
            validate_embedding_parent(
                corpus.embeddings[0].model_copy(update=changes), corpus.chunks[0]
            )
    with pytest.raises(PersistenceIntegrityError):
        validate_chunk_parent(corpus.chunks[0], corpus.documents[1])


def test_exact_float_payload_and_vector_projection():
    original = corpus_data().embeddings[2]
    row = embedding_to_record(original)
    assert embedding_from_record(row).vector == original.vector
    row.embedding = list(vector())
    with pytest.raises(PersistenceIntegrityError, match="embedding"):
        embedding_from_record(row)


def test_metadata_projection_uses_casefold_and_component_paths():
    row = chunk_to_record(corpus_data().chunks[2])
    assert "strasse" in row.filter_metadata["labels"]
    assert row.filter_metadata["path_prefixes"] == [
        "src",
        "src/database",
        "src/database/session.py",
    ]
    assert row.event_metadata["labels"] == ["BUG", "Straße"]
    row.filter_metadata["labels"] = ["wrong"]
    with pytest.raises(PersistenceIntegrityError):
        chunk_from_record(row)


@pytest.mark.parametrize(
    "url",
    ["", "sqlite://", "postgresql://localhost/test", "postgresql+psycopg://localhost"],
)
def test_database_configuration_rejects_wrong_driver(url):
    with pytest.raises(ValidationError):
        DatabaseConfig(url=url)


def test_engine_is_lazy_and_credentials_hidden():
    config = DatabaseConfig(
        url="postgresql+psycopg://synthetic:placeholder@localhost:1/test"
    )
    assert "placeholder" not in repr(config)
    engine = create_database_engine(config)
    assert engine.hide_parameters
    engine.dispose()
