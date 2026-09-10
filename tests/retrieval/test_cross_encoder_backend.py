import sys
from types import ModuleType

import pytest

import reporecall.retrieval.cross_encoder_backend as backend_module
from reporecall.retrieval import (
    CrossEncoderRerankerConfig,
    RerankerModelError,
    RerankerOutputError,
    SentenceTransformerCrossEncoderBackend,
)


class FakeArray:
    def __init__(self, values) -> None:
        self.values = values

    def tolist(self):
        return self.values


class FakeModel:
    def __init__(self, output=None, error: Exception | None = None) -> None:
        self.output = [1.0, -2.0] if output is None else output
        self.error = error
        self.calls: list[dict[str, object]] = []

    def predict(
        self,
        inputs,
        *,
        batch_size,
        show_progress_bar,
        convert_to_numpy,
    ):
        self.calls.append(
            {
                "inputs": inputs,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
            }
        )
        if self.error is not None:
            raise self.error
        return FakeArray(self.output)


def test_backend_loads_lazily_once_and_scores_exact_pairs_in_one_batch(monkeypatch):
    model = FakeModel()
    loader_calls: list[tuple[str, str | None]] = []

    def fake_loader(model_name: str, device: str | None):
        loader_calls.append((model_name, device))
        return model

    monkeypatch.setattr(backend_module, "_load_cross_encoder_model", fake_loader)
    backend = SentenceTransformerCrossEncoderBackend(
        CrossEncoderRerankerConfig(
            model_name="test/model",
            candidate_k=2,
            top_k=2,
            batch_size=2,
            device="cpu",
        )
    )
    query = "ConnectionResetError after retry_worker() crashes"
    passages = [
        "src/database/session.py\n@@ -10,4 +10,8 @@",
        "unrelated text",
    ]

    assert loader_calls == []
    assert backend.model_name == "test/model"
    assert backend.score(query, passages) == [1.0, -2.0]
    assert loader_calls == [("test/model", "cpu")]
    assert model.calls == [
        {
            "inputs": [(query, passages[0]), (query, passages[1])],
            "batch_size": 2,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
    ]

    backend.score("second query", passages)
    assert loader_calls == [("test/model", "cpu")]
    assert len(model.calls) == 2


def test_backend_delegates_batch_splitting_to_one_predict_call(monkeypatch):
    model = FakeModel(output=[float(index) for index in range(5)])
    monkeypatch.setattr(
        backend_module,
        "_load_cross_encoder_model",
        lambda model_name, device: model,
    )
    backend = SentenceTransformerCrossEncoderBackend(
        CrossEncoderRerankerConfig(candidate_k=5, top_k=5, batch_size=2)
    )
    passages = [f"passage {index}" for index in range(5)]

    assert backend.score("query", passages) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert len(model.calls) == 1
    assert model.calls[0]["batch_size"] == 2


def test_empty_input_does_not_load_or_invoke_model(monkeypatch):
    def unexpected_loader(model_name, device):
        raise AssertionError("model should not load")

    monkeypatch.setattr(
        backend_module,
        "_load_cross_encoder_model",
        unexpected_loader,
    )

    assert SentenceTransformerCrossEncoderBackend().score("query", []) == []


@pytest.mark.parametrize("output", [[1.0], [1.0, 2.0, 3.0]])
def test_backend_rejects_output_count_mismatch(monkeypatch, output):
    monkeypatch.setattr(
        backend_module,
        "_load_cross_encoder_model",
        lambda model_name, device: FakeModel(output=output),
    )

    with pytest.raises(RerankerOutputError, match="different number"):
        SentenceTransformerCrossEncoderBackend().score("query", ["a", "b"])


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (42, "score sequence"),
        ([[1.0]], "one score"),
        (["invalid"], "numeric"),
        ([float("nan")], "finite"),
        ([float("inf")], "finite"),
        ([float("-inf")], "finite"),
    ],
)
def test_backend_rejects_malformed_or_nonfinite_output(
    monkeypatch,
    output,
    message,
):
    monkeypatch.setattr(
        backend_module,
        "_load_cross_encoder_model",
        lambda model_name, device: FakeModel(output=output),
    )

    with pytest.raises(RerankerOutputError, match=message):
        SentenceTransformerCrossEncoderBackend().score("query", ["passage"])


def test_backend_accepts_negative_finite_logits(monkeypatch):
    monkeypatch.setattr(
        backend_module,
        "_load_cross_encoder_model",
        lambda model_name, device: FakeModel(output=[-7.25]),
    )

    assert SentenceTransformerCrossEncoderBackend().score("query", ["text"]) == [
        -7.25
    ]


def test_backend_wraps_model_load_failure(monkeypatch):
    def failing_loader(model_name, device):
        raise RuntimeError("download failed")

    monkeypatch.setattr(
        backend_module,
        "_load_cross_encoder_model",
        failing_loader,
    )

    with pytest.raises(RerankerModelError, match="Unable to load") as exc_info:
        SentenceTransformerCrossEncoderBackend().score("query", ["text"])

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_backend_wraps_prediction_failure(monkeypatch):
    model = FakeModel(error=RuntimeError("inference failed"))
    monkeypatch.setattr(
        backend_module,
        "_load_cross_encoder_model",
        lambda model_name, device: model,
    )

    with pytest.raises(RerankerModelError, match="inference failed") as exc_info:
        SentenceTransformerCrossEncoderBackend().score("query", ["text"])

    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    ("device", "expected_kwargs"),
    [(None, {}), ("cpu", {"device": "cpu"})],
)
def test_real_loader_passes_model_name_and_optional_device(
    monkeypatch,
    device,
    expected_kwargs,
):
    constructor_calls: list[tuple[str, dict[str, str]]] = []
    model = FakeModel(output=[1.0])

    def fake_cross_encoder(model_name: str, **kwargs):
        constructor_calls.append((model_name, kwargs))
        return model

    fake_module = ModuleType("sentence_transformers")
    fake_module.CrossEncoder = fake_cross_encoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    loaded = backend_module._load_cross_encoder_model("test/model", device)

    assert loaded is model
    assert constructor_calls == [("test/model", expected_kwargs)]


def test_backend_module_does_not_import_sentence_transformers_eagerly():
    assert "sentence_transformers" not in vars(backend_module)
