from collections.abc import Sequence
from typing import Protocol


class EmbeddingBackend(Protocol):
    """Minimal text-to-vector boundary used by embedding generation."""

    @property
    def model_name(self) -> str:
        """Return the stable model identity used for generated vectors."""

        ...

    @property
    def normalized(self) -> bool:
        """Return whether generated vectors are L2 normalized."""

        ...

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Generate one vector for every input text in order."""

        ...
