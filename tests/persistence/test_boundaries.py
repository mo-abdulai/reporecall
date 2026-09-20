"""Offline architectural and failure-boundary checks."""

import ast
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError

import reporecall.persistence as package
from reporecall.config import Settings
from reporecall.persistence import (
    PersistenceIntegrityError,
    PersistenceQueryError,
    PersistentCorpusRepository,
    PostgresVectorRetriever,
)
from tests.persistence.fixtures import MODEL, FakeEmbeddingBackend, corpus_data


def test_no_generation_web_or_ingestion_dependencies():
    forbidden = (
        "openai",
        "fastapi",
        "reporecall.generation",
        "reporecall.ingestion",
        "reporecall.query",
    )
    for path in Path(package.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(forbidden)
            elif isinstance(node, ast.Import):
                assert not any(alias.name.startswith(forbidden) for alias in node.names)
    # Canonical domain models must not depend on SQLAlchemy/ORM.
    for path in (Path(package.__file__).parent.parent / "models").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    ("sqlalchemy", "pgvector", "psycopg", "reporecall.persistence")
                )


def test_configuration_remains_optional():
    settings = Settings(_env_file=None, database_url=None)
    assert settings.database_url is None
    assert settings.database_echo is False


@pytest.mark.parametrize("operation", ["store", "load_events", "load_corpus", "search"])
def test_provider_errors_keep_cause(operation):
    failure = OperationalError("synthetic", {}, RuntimeError("offline"))
    sessions = Mock(side_effect=failure)
    sessions.begin.side_effect = failure
    repository = PersistentCorpusRepository(session_factory=sessions)
    with pytest.raises(PersistenceQueryError) as caught:
        if operation == "search":
            PostgresVectorRetriever(
                session_factory=sessions,
                embedding_backend=FakeEmbeddingBackend(),
                model_name=MODEL,
            ).search("query")
        else:
            getattr(repository, operation)()
    assert caught.value.__cause__ is failure


def test_integrity_errors_are_distinct():
    failure = IntegrityError("synthetic", {}, RuntimeError("constraint"))
    sessions = Mock()
    sessions.begin.side_effect = failure
    with pytest.raises(PersistenceIntegrityError) as caught:
        PersistentCorpusRepository(session_factory=sessions).store()
    assert caught.value.__cause__ is failure


def test_invalid_write_fails_before_opening_database():
    sessions = Mock()
    embedding = corpus_data().embeddings[0].model_copy(update={"normalized": False})
    with pytest.raises(PersistenceIntegrityError):
        PersistentCorpusRepository(session_factory=sessions).store(
            embeddings=[embedding]
        )
    sessions.begin.assert_not_called()
