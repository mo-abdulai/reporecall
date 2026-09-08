import sys
from types import ModuleType

import pytest
from pydantic import ValidationError

import reporecall.embeddings.sentence_transformer_backend as backend_module
from reporecall.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingConfig,
    EmbeddingModelError,
    EmbeddingOutputError,
    SentenceTransformerEmbeddingBackend,
)


class FakeArray:
    def __init__(self, values) -> None:
        self.values = values

    def tolist(self):
        return self.values


class FakeModel:
    def __init__(self, output=None, error: Exception | None = None) -> None:
        self.output = output or [[1.0, 2.0], [3.0, 4.0]]
        self.error = error
        self.calls: list[dict[str, object]] = []

    def encode(
        self,
        sentences,
        *,
        batch_size,
        show_progress_bar,
        convert_to_numpy,
        normalize_embeddings,
    ):
        self.calls.append(
            {
                "sentences": sentences,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        if self.error is not None:
            raise self.error
        return FakeArray(self.output)


def test_embedding_config_defaults_are_valid_and_immutable():
    config = EmbeddingConfig()

    assert config.model_name == DEFAULT_EMBEDDING_MODEL
    assert config.batch_size == 32
    assert config.normalize_embeddings is True
    assert config.device is None
    with pytest.raises(ValidationError, match="frozen"):
        config.batch_size = 8


@pytest.mark.parametrize("model_name", ["", "   "])
def test_embedding_config_rejects_blank_model_name(model_name: str):
    with pytest.raises(ValidationError, match="must not be blank"):
        EmbeddingConfig(model_name=model_name)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_embedding_config_rejects_nonpositive_batch_size(batch_size: int):
    with pytest.raises(ValidationError):
        EmbeddingConfig(batch_size=batch_size)


def test_backend_loads_lazily_once_and_propagates_encode_configuration(monkeypatch):
    model = FakeModel()
    loader_calls: list[tuple[str, str | None]] = []

    def fake_loader(model_name: str, device: str | None):
        loader_calls.append((model_name, device))
        return model

    monkeypatch.setattr(
        backend_module,
        "_load_sentence_transformer_model",
        fake_loader,
    )
    config = EmbeddingConfig(
        model_name="test/model",
        batch_size=2,
        normalize_embeddings=True,
        device="cpu",
    )
    backend = SentenceTransformerEmbeddingBackend(config)

    assert loader_calls == []
    assert backend.model_name == "test/model"
    assert backend.normalized is True
    assert backend.embed(["text A", "text B"]) == [[1.0, 2.0], [3.0, 4.0]]
    assert loader_calls == [("test/model", "cpu")]
    assert model.calls == [
        {
            "sentences": ["text A", "text B"],
            "batch_size": 2,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
    ]

    backend.embed(["text C", "text D"])
    assert loader_calls == [("test/model", "cpu")]


def test_backend_preserves_non_normalized_configuration(monkeypatch):
    model = FakeModel(output=[[3.0, 4.0]])
    monkeypatch.setattr(
        backend_module,
        "_load_sentence_transformer_model",
        lambda model_name, device: model,
    )
    backend = SentenceTransformerEmbeddingBackend(
        EmbeddingConfig(batch_size=1, normalize_embeddings=False)
    )

    assert backend.embed(["text"]) == [[3.0, 4.0]]
    assert backend.normalized is False
    assert model.calls[0]["normalize_embeddings"] is False


def test_backend_delegates_batch_splitting_to_sentence_transformer(monkeypatch):
    output = [[float(index)] for index in range(5)]
    model = FakeModel(output=output)
    monkeypatch.setattr(
        backend_module,
        "_load_sentence_transformer_model",
        lambda model_name, device: model,
    )
    backend = SentenceTransformerEmbeddingBackend(EmbeddingConfig(batch_size=2))
    texts = [f"text {index}" for index in range(5)]

    assert backend.embed(texts) == output
    assert len(model.calls) == 1
    assert model.calls[0]["sentences"] == texts
    assert model.calls[0]["batch_size"] == 2


def test_empty_backend_input_does_not_load_model(monkeypatch):
    def unexpected_loader(model_name, device):
        raise AssertionError("model should not load")

    monkeypatch.setattr(
        backend_module,
        "_load_sentence_transformer_model",
        unexpected_loader,
    )

    assert SentenceTransformerEmbeddingBackend().embed([]) == []


@pytest.mark.parametrize(
    "output",
    [
        [[1.0, 2.0]],
        [[1.0], [2.0], [3.0]],
    ],
)
def test_backend_rejects_output_count_mismatch(monkeypatch, output):
    monkeypatch.setattr(
        backend_module,
        "_load_sentence_transformer_model",
        lambda model_name, device: FakeModel(output=output),
    )

    with pytest.raises(EmbeddingOutputError, match="different number"):
        SentenceTransformerEmbeddingBackend().embed(["first", "second"])


@pytest.mark.parametrize(
    "output,error_message",
    [
        (42, "vector batch"),
        ([1.0, 2.0], "numeric vector"),
        ([["invalid"]], "numeric values"),
    ],
)
def test_backend_rejects_malformed_model_output(monkeypatch, output, error_message):
    model = FakeModel()
    model.output = output
    monkeypatch.setattr(
        backend_module,
        "_load_sentence_transformer_model",
        lambda model_name, device: model,
    )

    with pytest.raises(EmbeddingOutputError, match=error_message):
        SentenceTransformerEmbeddingBackend().embed(["text"])


def test_backend_wraps_model_load_failure(monkeypatch):
    def failing_loader(model_name, device):
        raise RuntimeError("download failed")

    monkeypatch.setattr(
        backend_module,
        "_load_sentence_transformer_model",
        failing_loader,
    )

    with pytest.raises(EmbeddingModelError, match="Unable to load") as exc_info:
        SentenceTransformerEmbeddingBackend().embed(["text"])

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_backend_wraps_inference_failure(monkeypatch):
    model = FakeModel(error=RuntimeError("inference failed"))
    monkeypatch.setattr(
        backend_module,
        "_load_sentence_transformer_model",
        lambda model_name, device: model,
    )

    with pytest.raises(EmbeddingModelError, match="inference failed") as exc_info:
        SentenceTransformerEmbeddingBackend().embed(["text"])

    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    "device,expected_kwargs",
    [(None, {}), ("cpu", {"device": "cpu"})],
)
def test_real_loader_passes_model_name_and_optional_device(
    monkeypatch,
    device,
    expected_kwargs,
):
    constructor_calls: list[tuple[str, dict[str, str]]] = []
    model = FakeModel(output=[[1.0]])

    def fake_sentence_transformer(model_name: str, **kwargs):
        constructor_calls.append((model_name, kwargs))
        return model

    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = fake_sentence_transformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    loaded = backend_module._load_sentence_transformer_model("test/model", device)

    assert loaded is model
    assert constructor_calls == [("test/model", expected_kwargs)]


def test_backend_module_does_not_import_sentence_transformers_eagerly():
    assert "sentence_transformers" not in vars(backend_module)
