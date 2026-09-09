class RetrievalError(Exception):
    """Base error for dense retrieval failures."""


class VectorIndexError(RetrievalError):
    """Raised when a vector index cannot be built or searched safely."""


class VectorIndexCompatibilityError(VectorIndexError):
    """Raised when vectors, models, or normalization settings are incompatible."""
