from types import SimpleNamespace

import pytest

import reporecall.query.openai_backend as openai_backend_module
from reporecall.query import (
    OpenAIQueryUnderstandingBackend,
    QueryUnderstandingBackendError,
    QueryUnderstandingBackendResult,
    QueryUnderstandingConfig,
    QueryUnderstandingValidationError,
)

_DEFAULT_OUTPUT = object()


class FakeResponses:
    def __init__(self, output_parsed: object = _DEFAULT_OUTPUT) -> None:
        if output_parsed is _DEFAULT_OUTPUT:
            output_parsed = QueryUnderstandingBackendResult(
                retrieval_query="authentication fixes",
                languages=("Python",),
            )
        self.output_parsed = output_parsed
        self.calls: list[dict[str, object]] = []
        self.error: Exception | None = None

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output_parsed=self.output_parsed)


class FakeClient:
    def __init__(self, responses: FakeResponses | None = None) -> None:
        self.responses = responses or FakeResponses()


def test_openai_backend_calls_responses_parse_with_structured_schema():
    responses = FakeResponses()
    backend = OpenAIQueryUnderstandingBackend(
        QueryUnderstandingConfig(model_name="test-model", temperature=0.0),
        client=FakeClient(responses),
    )

    result = backend.understand("Find Python authentication fixes")

    assert result.retrieval_query == "authentication fixes"
    assert result.languages == ("Python",)
    assert backend.model_name == "test-model"
    assert len(responses.calls) == 1
    call = responses.calls[0]
    assert call["model"] == "test-model"
    assert call["input"] == "Find Python authentication fixes"
    assert call["text_format"] is QueryUnderstandingBackendResult
    assert call["temperature"] == 0.0
    assert "conservative" in str(call["instructions"]).lower()
    assert "chain" not in str(call["instructions"]).lower()


def test_openai_backend_loads_api_key_from_settings_lazily(monkeypatch):
    client = FakeClient()
    loader_calls: list[str] = []
    monkeypatch.setattr(
        openai_backend_module,
        "get_settings",
        lambda: SimpleNamespace(openai_api_key="settings-key"),
    )

    def fake_loader(api_key: str):
        loader_calls.append(api_key)
        return client

    monkeypatch.setattr(openai_backend_module, "_load_openai_client", fake_loader)
    backend = OpenAIQueryUnderstandingBackend()

    assert loader_calls == []
    backend.understand("query")
    backend.understand("query")

    assert loader_calls == ["settings-key"]


def test_explicit_api_key_overrides_settings(monkeypatch):
    loader_calls: list[str] = []
    monkeypatch.setattr(
        openai_backend_module,
        "get_settings",
        lambda: SimpleNamespace(openai_api_key="settings-key"),
    )
    monkeypatch.setattr(
        openai_backend_module,
        "_load_openai_client",
        lambda api_key: loader_calls.append(api_key) or FakeClient(),
    )

    OpenAIQueryUnderstandingBackend(api_key="explicit-key").understand("query")

    assert loader_calls == ["explicit-key"]


def test_missing_api_key_fails_before_provider_call(monkeypatch):
    monkeypatch.setattr(
        openai_backend_module,
        "get_settings",
        lambda: SimpleNamespace(openai_api_key=None),
    )

    with pytest.raises(QueryUnderstandingBackendError, match="OPENAI_API_KEY"):
        OpenAIQueryUnderstandingBackend().understand("query")


def test_provider_failure_is_wrapped():
    responses = FakeResponses()
    responses.error = RuntimeError("network failed")
    backend = OpenAIQueryUnderstandingBackend(client=FakeClient(responses))

    with pytest.raises(QueryUnderstandingBackendError, match="OpenAI") as exc_info:
        backend.understand("query")

    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    "output",
    [
        {"retrieval_query": "query", "unsupported_filter": "FastAPI"},
        {"languages": ("Python",)},
        None,
    ],
)
def test_invalid_structured_output_is_rejected(output: object):
    backend = OpenAIQueryUnderstandingBackend(
        client=FakeClient(FakeResponses(output))
    )

    with pytest.raises(QueryUnderstandingValidationError, match="invalid"):
        backend.understand("query")


def test_blank_query_is_rejected_without_provider_call():
    responses = FakeResponses()
    backend = OpenAIQueryUnderstandingBackend(client=FakeClient(responses))

    with pytest.raises(QueryUnderstandingValidationError, match="must not be blank"):
        backend.understand("   ")

    assert responses.calls == []


def test_openai_sdk_is_not_imported_at_module_import_time():
    assert "openai" not in vars(openai_backend_module)
