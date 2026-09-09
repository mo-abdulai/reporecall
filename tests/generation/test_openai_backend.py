from types import SimpleNamespace

import pytest

import reporecall.generation.openai_backend as openai_backend_module
from reporecall.generation import (
    LLMBackendError,
    OpenAILLMBackend,
    RAGConfig,
    RAGResponseError,
)


class FakeResponses:
    def __init__(self, output_text: object = "Grounded answer [E1].") -> None:
        self.output_text = output_text
        self.calls: list[dict[str, object]] = []
        self.error: Exception | None = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, responses: FakeResponses | None = None) -> None:
        self.responses = responses or FakeResponses()


def test_openai_backend_calls_responses_api_with_configured_values():
    responses = FakeResponses()
    backend = OpenAILLMBackend(
        RAGConfig(model_name="test-model", temperature=0.25),
        client=FakeClient(responses),
    )

    output = backend.generate(system_prompt="system", user_prompt="user")

    assert output == "Grounded answer [E1]."
    assert backend.model_name == "test-model"
    assert responses.calls == [
        {
            "model": "test-model",
            "instructions": "system",
            "input": "user",
            "temperature": 0.25,
        }
    ]


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
    backend = OpenAILLMBackend()

    assert loader_calls == []
    backend.generate(system_prompt="system", user_prompt="user")
    backend.generate(system_prompt="system", user_prompt="user")

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

    OpenAILLMBackend(api_key="explicit-key").generate(
        system_prompt="system",
        user_prompt="user",
    )

    assert loader_calls == ["explicit-key"]


def test_missing_api_key_fails_before_provider_call(monkeypatch):
    monkeypatch.setattr(
        openai_backend_module,
        "get_settings",
        lambda: SimpleNamespace(openai_api_key=None),
    )

    with pytest.raises(LLMBackendError, match="OPENAI_API_KEY"):
        OpenAILLMBackend().generate(system_prompt="system", user_prompt="user")


def test_provider_failure_is_wrapped():
    responses = FakeResponses()
    responses.error = RuntimeError("network failed")
    backend = OpenAILLMBackend(client=FakeClient(responses))

    with pytest.raises(LLMBackendError, match="generation failed") as exc_info:
        backend.generate(system_prompt="system", user_prompt="user")

    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.parametrize("output", ["", "   ", None, 42])
def test_empty_or_invalid_provider_response_is_rejected(output: object):
    backend = OpenAILLMBackend(client=FakeClient(FakeResponses(output)))

    with pytest.raises(RAGResponseError, match="empty text response"):
        backend.generate(system_prompt="system", user_prompt="user")


@pytest.mark.parametrize(
    "system_prompt,user_prompt",
    [("", "user"), ("   ", "user"), ("system", ""), ("system", "   ")],
)
def test_blank_prompts_are_rejected_without_provider_call(
    system_prompt: str,
    user_prompt: str,
):
    responses = FakeResponses()
    backend = OpenAILLMBackend(client=FakeClient(responses))

    with pytest.raises(RAGResponseError, match="must not be blank"):
        backend.generate(system_prompt=system_prompt, user_prompt=user_prompt)

    assert responses.calls == []


def test_openai_sdk_is_not_imported_at_module_import_time():
    assert "openai" not in vars(openai_backend_module)
