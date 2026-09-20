"""Reconstruct existing consumers from a fresh connection without ingestion."""

import os
import subprocess
import sys

import pytest

from reporecall.evaluation import normalize_ranked_hits
from reporecall.persistence import (
    DatabaseConfig,
    PersistentCorpusRepository,
    PostgresVectorRetriever,
    create_database_engine,
    create_session_factory,
)
from reporecall.provenance import CitationBundleBuilder, CitationProvenanceIndex
from reporecall.retrieval import (
    BM25Index,
    RelationshipContextExpander,
    RelationshipContextIndex,
)
from tests.persistence.fixtures import MODEL, FakeEmbeddingBackend, store_data
from tests.retrieval.test_relationship_context_expander import ranked


@pytest.mark.postgres
def test_restart_rebuilds_search_relationships_and_citations(database):
    engine, sessions = database
    original = store_data(PersistentCorpusRepository(session_factory=sessions))
    connection_url = engine.url.render_as_string(hide_password=False)
    engine.dispose()
    # A separate interpreter has no access to the writer's in-memory domain objects.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
from reporecall.persistence import DatabaseConfig, create_database_engine, create_session_factory, PersistentCorpusRepository
from reporecall.retrieval import BM25Index
engine = create_database_engine(DatabaseConfig(url=os.environ['TEST_DATABASE_URL']))
try:
    corpus = PersistentCorpusRepository(session_factory=create_session_factory(engine)).load_corpus()
    index = BM25Index()
    index.build(corpus.chunks)
    assert index.search('session cleanup', k=3)
    print(len(corpus.events), len(corpus.documents), len(corpus.chunks), len(corpus.embeddings))
finally:
    engine.dispose()
""",
        ],
        env={**os.environ, "TEST_DATABASE_URL": connection_url},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "4 4 8 8"
    fresh = create_database_engine(DatabaseConfig(url=connection_url))
    try:
        fresh_sessions = create_session_factory(fresh)
        restored = PersistentCorpusRepository(
            session_factory=fresh_sessions
        ).load_corpus()
        bm25 = BM25Index()
        bm25.build(restored.chunks)
        assert bm25.search("session cleanup", k=3)
        context = RelationshipContextExpander(
            index=RelationshipContextIndex(
                events=restored.events, chunks=restored.chunks
            )
        ).expand(ranked([next(c for c in restored.chunks if c.chunk_id == "b-pr")]))
        assert [c.chunk.chunk_id for c in context.expanded_chunks] == ["b-commit"]
        bundle = CitationBundleBuilder(
            provenance_index=CitationProvenanceIndex(
                documents=restored.documents, chunks=restored.chunks
            )
        ).build(context)
        assert bundle.retrieved[0].urls == ("https://example.test/b/pr/10",)
        assert bundle.expanded[0].urls == ()
        assert (
            bundle.expanded[0].expanded_chunk.reasons[0].relationship
            == original.events[1].relationships[0]
        )
        assert bundle.expanded[0].seed_references[0].citation.label == "R1"
        hits = PostgresVectorRetriever(
            session_factory=fresh_sessions,
            embedding_backend=FakeEmbeddingBackend(),
            model_name=MODEL,
        ).search("query", k=3)
        assert len(normalize_ranked_hits(hits)) == 3
    finally:
        fresh.dispose()
