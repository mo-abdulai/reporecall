"""Opt-in PostgreSQL HTTP tests with synthetic vectors and fake neural scoring."""

from fastapi.testclient import TestClient

from reporecall.api.app import create_app
from reporecall.persistence import PersistentCorpusRepository
from reporecall.services.runtime import ServiceConfig
from tests.persistence.fixtures import MODEL, FakeEmbeddingBackend, store_data


def api(database, monkeypatch):
    engine, sessions = database
    from reporecall.services import runtime

    monkeypatch.setattr(runtime, "create_database_engine", lambda config: engine)
    monkeypatch.setattr(runtime, "create_session_factory", lambda engine: sessions)
    embedding = FakeEmbeddingBackend()
    monkeypatch.setattr(
        runtime, "SentenceTransformerEmbeddingBackend", lambda config: embedding
    )

    class Reranker:
        def __init__(self, config):
            self.model_name = config.model_name

        def score(self, query, passages):
            return [float(len(passages) - i) for i, _ in enumerate(passages)]

    monkeypatch.setattr(runtime, "SentenceTransformerCrossEncoderBackend", Reranker)
    config = ServiceConfig(
        _env_file=None,
        database_url="postgresql+psycopg://placeholder/test",
        api_embedding_model=MODEL,
    )
    return create_app(config=config), embedding


def test_persisted_full_http_pipeline(database, monkeypatch):
    corpus = store_data(PersistentCorpusRepository(session_factory=database[1]))
    app, embedding = api(database, monkeypatch)
    with TestClient(app) as client:
        assert client.get("/api/v1/ready").status_code == 200
        stats = client.get("/api/v1/corpus").json()
        assert stats["events"] == 4
        assert stats["chunks"] == 8
        assert stats["embeddings"] == 8
        assert stats["repositories"] == 2
        assert stats["embedding_models"][0]["model_name"] == MODEL
        repos = client.get("/api/v1/repositories").json()["repositories"]
        assert [r["name"] for r in repos] == ["fixture", "other"]
        assert (
            client.get("/api/v1/chunks/" + corpus.chunks[0].chunk_id).json()["text"]
            == corpus.chunks[0].text
        )
        assert (
            client.get("/api/v1/documents/" + corpus.documents[0].document_id).json()[
                "document_id"
            ]
            == corpus.documents[0].document_id
        )
        assert (
            client.get("/api/v1/events/" + corpus.events[0].event_id).json()["event_id"]
            == corpus.events[0].event_id
        )
        assert client.get("/api/v1/chunks/missing").status_code == 404
        query = "  session cleanup\n stack.trace()  "
        response = client.post("/api/v1/search", json={"query": query, "top_k": 1})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["query"] == query
        assert embedding.inputs == [(query,)]
        assert len(result["hits"]) == 1
        assert result["hits"][0]["rank"] == 1
        assert result["citations"]["retrieved"][0]["label"] == "R1"
        assert result["expanded_context"]
        assert result["citations"]["expanded"][0]["label"] == "X1"
        assert result["citations"]["expanded"][0]["seed_references"][0]["label"] == "R1"
        assert "rank" not in result["expanded_context"][0]
        assert "reranker_score" not in result["expanded_context"][0]
        all_citations = (
            result["citations"]["retrieved"] + result["citations"]["expanded"]
        )
        urls = {url for item in all_citations for url in item["urls"]}
        assert urls
        assert urls <= {
            source.url
            for doc in corpus.documents
            for source in doc.sources
            if source.url
        }
        again = client.post("/api/v1/search", json={"query": query, "top_k": 1}).json()
        assert result == again
        disabled = client.post(
            "/api/v1/search",
            json={
                "query": query,
                "expand_relationships": False,
                "include_citations": False,
            },
        ).json()
        assert disabled["expanded_context"] == []
        assert disabled["citations"] is None
        assert disabled["context_truncated"] is False


def test_empty_database_ready_and_no_inference(database, monkeypatch):
    app, embedding = api(database, monkeypatch)
    with TestClient(app) as client:
        assert client.get("/api/v1/ready").status_code == 200
        assert client.get("/api/v1/corpus").json()["chunks"] == 0
        result = client.post("/api/v1/search", json={"query": "anything"}).json()
        assert result["hits"] == []
        assert result["expanded_context"] == []
        assert result["citations"]["retrieved"] == []
        assert embedding.inputs == []


def test_http_filter_and_seed_only_citations(database, monkeypatch):
    store_data(PersistentCorpusRepository(session_factory=database[1]))
    app, _ = api(database, monkeypatch)
    with TestClient(app) as client:
        result = client.post(
            "/api/v1/search",
            json={
                "query": "session",
                "metadata_filter": {"languages": ["Python"]},
                "expand_relationships": False,
            },
        ).json()
        assert result["hits"]
        assert all(
            "Python" in hit["chunk"]["metadata"]["languages"] for hit in result["hits"]
        )
        assert result["expanded_context"] == []
        assert result["citations"]["expanded"] == []
        assert [item["label"] for item in result["citations"]["retrieved"]] == [
            "R" + str(i + 1) for i in range(len(result["hits"]))
        ]


def test_incomplete_model_corpus_unready(database, monkeypatch):
    corpus = store_data(PersistentCorpusRepository(session_factory=database[1]))
    from sqlalchemy import delete

    from reporecall.persistence.orm import EmbeddingRecord

    with database[1].begin() as session:
        session.execute(
            delete(EmbeddingRecord).where(
                EmbeddingRecord.chunk_id == corpus.chunks[0].chunk_id
            )
        )
    app, embedding = api(database, monkeypatch)
    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/api/v1/ready").status_code == 503
        assert embedding.inputs == []


def test_schema_failure_and_recovery(database, monkeypatch):
    from sqlalchemy import text

    app, _ = api(database, monkeypatch)
    with TestClient(app) as client:
        with database[0].begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE retrieval_chunks RENAME TO temporarily_unavailable_chunks"
                )
            )
        try:
            assert client.get("/api/v1/health").status_code == 200
            assert client.get("/api/v1/ready").status_code == 503
        finally:
            with database[0].begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE temporarily_unavailable_chunks RENAME TO retrieval_chunks"
                    )
                )
        assert client.get("/api/v1/ready").status_code == 200
