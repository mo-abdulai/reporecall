from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from reporecall.retrieval import (
    DEFAULT_CROSS_ENCODER_MODEL,
    CrossEncoderRerankerConfig,
    RerankerBackend,
)


class FakeRerankerBackend:
    @property
    def model_name(self) -> str:
        return "fake/reranker"

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        return [float(len(query) + len(passage)) for passage in passages]


def test_backend_protocol_accepts_a_minimal_scoring_implementation():
    backend: RerankerBackend = FakeRerankerBackend()

    assert backend.model_name == "fake/reranker"
    assert backend.score("q", ["a", "bb"]) == [2.0, 3.0]


def test_config_has_immutable_baseline_defaults_and_custom_values():
    defaults = CrossEncoderRerankerConfig()
    custom = CrossEncoderRerankerConfig(
        model_name="test/model",
        candidate_k=10,
        top_k=3,
        batch_size=4,
        device="cpu",
    )

    assert defaults.model_name == DEFAULT_CROSS_ENCODER_MODEL
    assert defaults.candidate_k == 20
    assert defaults.top_k == 5
    assert defaults.batch_size == 16
    assert defaults.device is None
    assert custom.model_name == "test/model"
    assert custom.device == "cpu"
    with pytest.raises(ValidationError, match="frozen"):
        custom.top_k = 2


@pytest.mark.parametrize(
    "values",
    [
        {"candidate_k": 0},
        {"candidate_k": -1},
        {"top_k": 0},
        {"top_k": -1},
        {"batch_size": 0},
        {"batch_size": -1},
    ],
)
def test_config_rejects_nonpositive_values(values: dict[str, int]):
    with pytest.raises(ValidationError):
        CrossEncoderRerankerConfig(**values)


@pytest.mark.parametrize("model_name", ["", "   "])
def test_config_rejects_blank_model_name(model_name: str):
    with pytest.raises(ValidationError, match="must not be blank"):
        CrossEncoderRerankerConfig(model_name=model_name)


@pytest.mark.parametrize("device", ["", "   "])
def test_config_rejects_blank_device(device: str):
    with pytest.raises(ValidationError, match="device must not be blank"):
        CrossEncoderRerankerConfig(device=device)


def test_config_rejects_top_k_above_candidate_k():
    with pytest.raises(ValidationError, match="cannot exceed candidate-k"):
        CrossEncoderRerankerConfig(candidate_k=4, top_k=5)
