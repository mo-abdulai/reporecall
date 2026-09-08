from collections.abc import Sequence
from typing import Protocol, TypeGuard, cast

from reporecall.embeddings.config import EmbeddingConfig
from reporecall.embeddings.exceptions import EmbeddingModelError, EmbeddingOutputError


class _SentenceTransformerModel(Protocol):
    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object: ...


class SentenceTransformerEmbeddingBackend:
    """Lazy local embedding backend powered by SentenceTransformers."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self._model: _SentenceTransformerModel | None = None

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def normalized(self) -> bool:
        return self.config.normalize_embeddings

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a text batch and return serialization-safe vectors."""

        if not texts:
            return []

        model = self._get_model()
        try:
            output = model.encode(
                list(texts),
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings,
            )
        except Exception as exc:
            raise EmbeddingModelError(
                f"Embedding inference failed for model {self.model_name!r}."
            ) from exc

        vectors = _plain_vectors(output)
        if len(vectors) != len(texts):
            raise EmbeddingOutputError(
                "SentenceTransformer returned a different number of vectors than texts."
            )
        return vectors

    def _get_model(self) -> _SentenceTransformerModel:
        if self._model is None:
            try:
                self._model = _load_sentence_transformer_model(
                    self.config.model_name,
                    self.config.device,
                )
            except EmbeddingModelError:
                raise
            except Exception as exc:
                raise EmbeddingModelError(
                    f"Unable to load SentenceTransformer model {self.model_name!r}."
                ) from exc
        return self._model


def _load_sentence_transformer_model(
    model_name: str,
    device: str | None,
) -> _SentenceTransformerModel:
    try:
        from sentence_transformers import SentenceTransformer

        model = (
            SentenceTransformer(model_name)
            if device is None
            else SentenceTransformer(model_name, device=device)
        )
    except Exception as exc:
        raise EmbeddingModelError(
            f"Unable to load SentenceTransformer model {model_name!r}."
        ) from exc
    return cast(_SentenceTransformerModel, model)


def _plain_vectors(output: object) -> list[list[float]]:
    converted = _tolist(output)
    if not _is_sequence(converted):
        raise EmbeddingOutputError("SentenceTransformer output must be a vector batch.")

    vectors: list[list[float]] = []
    for row in converted:
        row_values = _tolist(row)
        if not _is_sequence(row_values):
            raise EmbeddingOutputError(
                "Each SentenceTransformer output row must be a numeric vector."
            )
        try:
            vectors.append([float(cast(float, value)) for value in row_values])
        except (TypeError, ValueError) as exc:
            raise EmbeddingOutputError(
                "SentenceTransformer output vectors must contain numeric values."
            ) from exc
    return vectors


def _tolist(value: object) -> object:
    method = getattr(value, "tolist", None)
    return method() if callable(method) else value


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
