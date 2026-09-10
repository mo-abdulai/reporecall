import math
from collections.abc import Sequence
from typing import Protocol, TypeGuard, cast

from reporecall.retrieval.exceptions import (
    RerankerModelError,
    RerankerOutputError,
)
from reporecall.retrieval.reranker_backend import CrossEncoderRerankerConfig


class _CrossEncoderModel(Protocol):
    def predict(
        self,
        inputs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> object: ...


class SentenceTransformerCrossEncoderBackend:
    """Lazy local reranker backend powered by SentenceTransformers."""

    def __init__(self, config: CrossEncoderRerankerConfig | None = None) -> None:
        self.config = config or CrossEncoderRerankerConfig()
        self._model: _CrossEncoderModel | None = None

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Score exact query-passage pairs in one configured model call."""

        if not passages:
            return []

        pairs = [(query, passage) for passage in passages]
        model = self._get_model()
        try:
            output = model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise RerankerModelError(
                f"Cross-encoder inference failed for model {self.model_name!r}."
            ) from exc

        scores = _plain_scores(output)
        if len(scores) != len(passages):
            raise RerankerOutputError(
                "CrossEncoder returned a different number of scores than passages."
            )
        return scores

    def _get_model(self) -> _CrossEncoderModel:
        if self._model is None:
            try:
                self._model = _load_cross_encoder_model(
                    self.config.model_name,
                    self.config.device,
                )
            except RerankerModelError:
                raise
            except Exception as exc:
                raise RerankerModelError(
                    f"Unable to load CrossEncoder model {self.model_name!r}."
                ) from exc
        return self._model


def _load_cross_encoder_model(
    model_name: str,
    device: str | None,
) -> _CrossEncoderModel:
    try:
        from sentence_transformers import CrossEncoder

        model = (
            CrossEncoder(model_name)
            if device is None
            else CrossEncoder(model_name, device=device)
        )
    except Exception as exc:
        raise RerankerModelError(
            f"Unable to load CrossEncoder model {model_name!r}."
        ) from exc
    return cast(_CrossEncoderModel, model)


def _plain_scores(output: object) -> list[float]:
    converted = _tolist(output)
    if not _is_sequence(converted):
        raise RerankerOutputError("CrossEncoder output must be a score sequence.")

    scores: list[float] = []
    for value in converted:
        scalar = _tolist(value)
        if _is_sequence(scalar):
            raise RerankerOutputError("Each CrossEncoder output must be one score.")
        try:
            score = float(cast(float, scalar))
        except (TypeError, ValueError) as exc:
            raise RerankerOutputError(
                "CrossEncoder output scores must be numeric."
            ) from exc
        if not math.isfinite(score):
            raise RerankerOutputError("CrossEncoder output scores must be finite.")
        scores.append(score)
    return scores


def _tolist(value: object) -> object:
    method = getattr(value, "tolist", None)
    return method() if callable(method) else value


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
