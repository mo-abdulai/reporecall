class EmbeddingError(Exception):
    """Base error for embedding model and generation failures."""


class EmbeddingModelError(EmbeddingError):
    """Raised when an embedding model cannot load or run inference."""


class EmbeddingOutputError(EmbeddingError):
    """Raised when backend vectors do not satisfy the embedding contract."""
