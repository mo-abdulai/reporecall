"""Exact SQL ranking/filter semantics compared with the unchanged FAISS path."""

import pytest

from reporecall.github import GitHubRepository
from reporecall.models import EventActor, MetadataFilter
from reporecall.persistence import (
    PersistenceIntegrityError,
    PersistenceQueryError,
    PersistentCorpusRepository,
    PostgresVectorRetriever,
)
from reporecall.retrieval import FaissVectorIndex, VectorRetriever
from reporecall.retrieval.metadata_filter import MetadataFilterMatcher
from tests.persistence.fixtures import MODEL, FakeEmbeddingBackend, store_data, vector

pytestmark = pytest.mark.postgres

FILTERS = [
    None,
    MetadataFilter(),
    MetadataFilter(repositories=(GitHubRepository(owner="synthetic", name="fixture"),)),
    MetadataFilter(
        repositories=(GitHubRepository(owner="synthetic", name="other"),),
        issue_numbers=(10,),
    ),
    MetadataFilter(section_types=("pull_request",)),
    MetadataFilter(languages=("PYTHON", "Java")),
    MetadataFilter(languages=("Python",), labels=("bug",)),
    MetadataFilter(labels=("STRASSE",)),
    MetadataFilter(labels=("bu",)),
    MetadataFilter(path_prefixes=("src/database",)),
    MetadataFilter(path_prefixes=("src/data%",)),
    MetadataFilter(directories=("src/database",)),
    MetadataFilter(directories=("SRC/database",)),
    MetadataFilter(has_test_changes=True),
    MetadataFilter(has_test_changes=False),
    MetadataFilter(has_documentation_changes=True),
    MetadataFilter(has_configuration_changes=False),
    MetadataFilter(has_dependency_changes=True),
    MetadataFilter(
        min_added_lines=2,
        max_added_lines=3,
        min_deleted_lines=1,
        max_deleted_lines=2,
        min_changed_lines=3,
        max_changed_lines=5,
    ),
    MetadataFilter(issue_numbers=(9, 10)),
    MetadataFilter(pull_request_numbers=(10,)),
    MetadataFilter(commit_shas=("sha-b",)),
    MetadataFilter(commit_shas=("SHA-b",)),
    MetadataFilter(milestones=("Release A",)),
    MetadataFilter(milestones=("release a",)),
    MetadataFilter(extensions=("py",)),
    MetadataFilter(
        actors=(EventActor(actor_type="github_user", identifier="alice", name="Alice"),)
    ),
    MetadataFilter(actors=(EventActor(actor_type="github_user", identifier="alice"),)),
    MetadataFilter(
        actors=(
            EventActor(
                actor_type="git_author",
                identifier="local",
                email="synthetic@example.test",
            ),
        )
    ),
    MetadataFilter(languages=("Rust",)),
]


@pytest.mark.parametrize("metadata_filter", FILTERS)
def test_pgvector_faiss_and_filter_equivalence(database, metadata_filter):
    _, sessions = database
    corpus = store_data(PersistentCorpusRepository(session_factory=sessions))
    backend = FakeEmbeddingBackend()
    postgres = PostgresVectorRetriever(
        session_factory=sessions, embedding_backend=backend, model_name=MODEL
    )
    index = FaissVectorIndex()
    index.build(sorted(corpus.embeddings, key=lambda item: item.chunk_id))
    faiss = VectorRetriever(
        index=index,
        backend=FakeEmbeddingBackend(),
        chunks={c.chunk_id: c for c in corpus.chunks},
    )
    query = "  session cleanup\n"
    actual = postgres.search(query, k=20, metadata_filter=metadata_filter)
    expected = faiss.search(query, k=20, metadata_filter=metadata_filter)
    assert [hit.chunk.chunk_id for hit in actual] == [
        hit.chunk.chunk_id for hit in expected
    ]
    assert [hit.score for hit in actual] == pytest.approx(
        [hit.score for hit in expected], abs=1e-6
    )
    assert [hit.rank for hit in actual] == list(range(1, len(actual) + 1))
    matcher = MetadataFilterMatcher()
    eligible = {
        c.chunk_id
        for c in corpus.chunks
        if metadata_filter is None or matcher.matches(c, metadata_filter)
    }
    assert {hit.chunk.chunk_id for hit in actual} == eligible
    assert backend.inputs == ([(query,)] if actual else [])
    if actual:
        first = postgres.search(query, k=1, metadata_filter=metadata_filter)
        assert first[0] == actual[0]


def test_filtered_top_k_excludes_global_winner_and_ties_are_stable(database):
    _, sessions = database
    store_data(PersistentCorpusRepository(session_factory=sessions))
    retriever = PostgresVectorRetriever(
        session_factory=sessions,
        embedding_backend=FakeEmbeddingBackend(),
        model_name=MODEL,
    )
    hits = retriever.search(
        "query", k=1, metadata_filter=MetadataFilter(path_prefixes=("src/database",))
    )
    assert hits[0].chunk.chunk_id == "b-commit"
    assert hits[0].score == pytest.approx(0.8, abs=1e-6)
    assert [h.chunk.chunk_id for h in retriever.search("query", k=2)] == [
        "a-commit",
        "a-pr",
    ]


def test_empty_corpus_does_not_embed(database):
    _, sessions = database
    backend = FakeEmbeddingBackend()
    retriever = PostgresVectorRetriever(
        session_factory=sessions, embedding_backend=backend, model_name=MODEL
    )
    assert retriever.search("query") == []
    assert not backend.inputs


@pytest.mark.parametrize(
    "output", [[], [vector(), vector()], [[1, 0]], [vector(2)], [vector(float("nan"))]]
)
def test_invalid_query_embeddings(database, output):
    _, sessions = database
    store_data(PersistentCorpusRepository(session_factory=sessions))
    retriever = PostgresVectorRetriever(
        session_factory=sessions,
        embedding_backend=FakeEmbeddingBackend(output),
        model_name=MODEL,
    )
    with pytest.raises(PersistenceIntegrityError):
        retriever.search("query")


def test_model_and_normalization_checks(database):
    _, sessions = database
    store_data(PersistentCorpusRepository(session_factory=sessions))
    backend = FakeEmbeddingBackend()
    backend.model_name = "wrong"
    with pytest.raises(PersistenceIntegrityError):
        PostgresVectorRetriever(
            session_factory=sessions, embedding_backend=backend, model_name=MODEL
        ).search("query")
    with pytest.raises(PersistenceIntegrityError):
        PostgresVectorRetriever(
            session_factory=sessions, embedding_backend=backend, model_name="wrong"
        ).search("query")
    backend.model_name = MODEL
    backend.normalized = False
    with pytest.raises(PersistenceIntegrityError):
        PostgresVectorRetriever(
            session_factory=sessions, embedding_backend=backend, model_name=MODEL
        ).search("query")
    assert not backend.inputs


@pytest.mark.parametrize(
    "query,k", [(" ", 1), ("q", 0), ("q", -1), ("q", True), ("q", 1.5)]
)
def test_bad_search_input(database, query, k):
    _, sessions = database
    retriever = PostgresVectorRetriever(
        session_factory=sessions,
        embedding_backend=FakeEmbeddingBackend(),
        model_name=MODEL,
    )
    with pytest.raises(PersistenceQueryError):
        retriever.search(query, k=k)
