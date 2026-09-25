"""Offline HTTP contract tests with explicitly synthetic evidence."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from reporecall.api.app import create_app
from reporecall.models import ExpandedContextResult
from reporecall.persistence import PersistenceQueryError
from reporecall.provenance import CitationBundleBuilder, CitationProvenanceIndex
from reporecall.query import QueryUnderstandingBackendError
from reporecall.services.errors import ResourceNotFound
from reporecall.services.runtime import ServiceConfig
from reporecall.services.search_service import SearchServiceResult


def configuration(**kwargs):
    return ServiceConfig(_env_file=None, database_url=None, **kwargs)


@pytest.fixture
def services():
    context = ExpandedContextResult(
        query="query", seed_hits=(), expanded_chunks=(), truncated=False
    )
    bundle = CitationBundleBuilder(
        provenance_index=CitationProvenanceIndex(documents=(), chunks=())
    ).build(context)
    return SimpleNamespace(
        corpus=Mock(),
        search=Mock(
            search=Mock(
                return_value=SearchServiceResult(context=context, citations=bundle)
            )
        ),
        query=Mock(),
    )


@contextmanager
def client_for(services, **config):
    @contextmanager
    def resources():
        yield services

    with TestClient(
        create_app(config=configuration(**config), resources_factory=resources),
        raise_server_exceptions=False,
    ) as client:
        yield client


def test_missing_database_does_not_break_liveness():
    with TestClient(create_app(config=configuration())) as client:
        assert client.get("/api/v1/health").json() == {"status": "ok"}
        assert client.get("/api/v1/ready").status_code == 503
        assert client.post("/api/v1/search", json={"query": "test"}).status_code == 503


def test_shared_resources_lifecycle(services):
    events = []

    @contextmanager
    def resources():
        events.append("open")
        try:
            yield services
        finally:
            events.append("close")

    app = create_app(config=configuration(), resources_factory=resources)
    assert events == []
    with TestClient(app) as client:
        assert client.get("/api/v1/ready").status_code == 200
        assert client.get("/api/v1/ready").status_code == 200
        assert events == ["open"]
    assert events == ["open", "close"]
    assert app.state.services is None


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": " \n "},
        {"query": "x" * 10001},
        {"query": "ok", "top_k": 0},
        {"query": "ok", "top_k": 51},
        {"query": "ok", "top_k": True},
        {"query": "ok", "metadata_filter": {"invented": True}},
        {"query": "ok", "extra": 1},
    ],
)
def test_bad_search_requests(services, payload):
    with client_for(services) as client:
        response = client.post("/api/v1/search", json=payload)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"
        services.search.search.assert_not_called()


def test_search_preserves_input_and_does_not_interpret(services):
    query = "  traceback\n  session.close()  "
    with client_for(services) as client:
        response = client.post(
            "/api/v1/search",
            json={
                "query": query,
                "top_k": 3,
                "expand_relationships": False,
                "include_citations": False,
            },
        )
        assert response.status_code == 200
        assert response.json()["hits"] == []
        services.search.search.assert_called_once_with(
            query=query,
            metadata_filter=None,
            top_k=3,
            expand_relationships=False,
            include_citations=False,
        )
        services.query.understand.assert_not_called()


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (PersistenceQueryError("postgresql://secret:password"), 503),
        (RuntimeError("secret stack trace"), 500),
        (ResourceNotFound("chunk"), 404),
    ],
)
def test_safe_errors(services, error, status):
    services.corpus.chunk.side_effect = error
    with client_for(services) as client:
        response = client.get("/api/v1/chunks/missing")
        assert response.status_code == status
        assert "secret" not in response.text
        assert set(response.json()) == {"error"}


def test_readiness_database_outage(services):
    services.corpus.check_ready.side_effect = PersistenceQueryError("private")
    with client_for(services) as client:
        assert client.get("/api/v1/ready").status_code == 503
        assert client.get("/api/v1/health").status_code == 200


def test_query_provider_failure(services):
    services.query.understand.side_effect = QueryUnderstandingBackendError(
        "token secret"
    )
    with client_for(services) as client:
        response = client.post("/api/v1/query/understand", json={"query": "why?"})
        assert response.status_code == 502
        assert "secret" not in response.text
        services.search.search.assert_not_called()


def test_docs_and_routes(services):
    with client_for(services) as client:
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200
        schema = client.get("/openapi.json").json()
        assert set(schema["paths"]) == {
            "/api/v1/" + path
            for path in (
                "health",
                "ready",
                "corpus",
                "repositories",
                "events/{event_id}",
                "documents/{document_id}",
                "chunks/{chunk_id}",
                "query/understand",
                "search",
            )
        }
        assert schema["paths"]["/api/v1/search"]["post"]["responses"]["422"]["content"][
            "application/json"
        ]["schema"]["$ref"].endswith("ErrorResponse")
    with client_for(services, api_docs_enabled=False) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


def test_cors_explicit_origin_only(services):
    with client_for(services, api_cors_origins=("http://localhost:3000",)) as client:
        headers = {
            "origin": "http://localhost:3000",
            "access-control-request-method": "POST",
        }
        response = client.options("/api/v1/search", headers=headers)
        assert response.headers["access-control-allow-origin"] == headers["origin"]
        assert "access-control-allow-credentials" not in response.headers
        headers["origin"] = "https://untrusted.example"
        assert client.options("/api/v1/search", headers=headers).status_code == 400


def test_query_success_is_independent(services):
    from reporecall.models import UnderstoodQuery

    query = " Find Python bugs\n "
    services.query.understand.return_value = UnderstoodQuery(
        original_query=query, retrieval_query="bugs", model_name="synthetic"
    )
    with client_for(services) as client:
        response = client.post("/api/v1/query/understand", json={"query": query})
        assert response.status_code == 200
        assert response.json()["original_query"] == query
        assert response.json()["retrieval_query"] == "bugs"
        services.query.understand.assert_called_once_with(query)
        services.search.search.assert_not_called()
        services.corpus.assert_not_called()


def test_startup_database_failure_is_safe():
    @contextmanager
    def unavailable():
        raise PersistenceQueryError("secret")
        yield

    with TestClient(
        create_app(config=configuration(), resources_factory=unavailable)
    ) as client:
        assert client.get("/api/v1/health").status_code == 200
        response = client.get("/api/v1/ready")
        assert response.status_code == 503
        assert "secret" not in response.text
