"""Real PostgreSQL atomic writes and deterministic restart reconstruction."""

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from reporecall.persistence import PersistenceIntegrityError, PersistentCorpusRepository
from reporecall.persistence.mapping import chunk_to_record, record_values
from reporecall.persistence.orm import (
    ChunkRecord,
    DocumentRecord,
    EmbeddingRecord,
    EventRecord,
)
from tests.persistence.fixtures import corpus_data, store_data

pytestmark = pytest.mark.postgres


def test_atomic_idempotent_roundtrip(database):
    _, sessions = database
    repository = PersistentCorpusRepository(session_factory=sessions)
    corpus = store_data(repository)
    store_data(repository, corpus)
    restored = repository.load_corpus()
    assert restored.events == tuple(sorted(corpus.events, key=lambda x: x.event_id))
    assert restored.documents == tuple(
        sorted(corpus.documents, key=lambda x: x.document_id)
    )
    assert restored.chunks == tuple(sorted(corpus.chunks, key=lambda x: x.chunk_id))
    assert restored.embeddings == tuple(
        sorted(corpus.embeddings, key=lambda x: (x.chunk_id, x.model_name))
    )
    assert repository.load_events() == restored.events
    assert repository.load_documents() == restored.documents
    assert repository.load_chunks() == restored.chunks
    assert repository.load_embeddings() == restored.embeddings
    with sessions() as session:
        assert [
            session.scalar(select(func.count()).select_from(table))
            for table in (EventRecord, DocumentRecord, ChunkRecord, EmbeddingRecord)
        ] == [4, 4, 8, 8]


@pytest.mark.parametrize(
    "field,value",
    [("text", "changed content"), ("document_id", "other"), ("event_id", "other")],
)
def test_conflicts_do_not_replace_corpus(database, field, value):
    _, sessions = database
    repository = PersistentCorpusRepository(session_factory=sessions)
    corpus = store_data(repository)
    previous = repository.load_corpus()
    with pytest.raises(PersistenceIntegrityError):
        repository.store(chunks=[corpus.chunks[0].model_copy(update={field: value})])
    assert repository.load_corpus() == previous


def test_rollback_after_partial_write(database):
    _, sessions = database
    repository = PersistentCorpusRepository(session_factory=sessions)
    corpus = corpus_data()
    stale = corpus.embeddings[0].model_copy(update={"source_text_sha256": "0" * 64})
    with pytest.raises(PersistenceIntegrityError, match="stale"):
        repository.store(
            events=corpus.events,
            documents=corpus.documents,
            chunks=corpus.chunks,
            embeddings=[stale],
        )
    assert (
        repository.load_events()
        == repository.load_documents()
        == repository.load_chunks()
        == repository.load_embeddings()
        == ()
    )


def test_foreign_key_and_vector_constraints(database):
    _, sessions = database
    corpus = corpus_data()
    with pytest.raises(IntegrityError), sessions.begin() as session:
        row = chunk_to_record(corpus.chunks[0])
        session.execute(ChunkRecord.__table__.insert().values(record_values(row)))
    repository = PersistentCorpusRepository(session_factory=sessions)
    store_data(repository)
    for values in (
        {"normalized": False},
        {"dimension": 768},
        {"source_text_sha256": "0" * 64},
        {"embedding": [0.0] * 384},
    ):
        with pytest.raises(IntegrityError), sessions.begin() as session:
            session.execute(update(EmbeddingRecord).values(**values))


def test_corrupt_projection_is_not_silently_loaded(database):
    _, sessions = database
    repository = PersistentCorpusRepository(session_factory=sessions)
    store_data(repository)
    with sessions.begin() as session:
        session.execute(update(ChunkRecord).values(text="corrupt"))
    with pytest.raises(PersistenceIntegrityError, match="disagrees"):
        repository.load_corpus()


def test_two_model_versions_are_explicit_and_idempotent(database):
    _, sessions = database
    repository = PersistentCorpusRepository(session_factory=sessions)
    corpus = store_data(repository)
    alternate = corpus.embeddings[0].model_copy(update={"model_name": "other-model"})
    repository.store(embeddings=[alternate])
    repository.store(embeddings=[alternate])
    assert len(repository.load_embeddings()) == 9
