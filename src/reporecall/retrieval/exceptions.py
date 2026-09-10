class RetrievalError(Exception):
    """Base error for retrieval failures."""


class HybridRetrievalError(RetrievalError):
    """Raised when hybrid candidate collection receives inconsistent results."""


class VectorIndexError(RetrievalError):
    """Raised when a vector index cannot be built or searched safely."""


class VectorIndexCompatibilityError(VectorIndexError):
    """Raised when vectors, models, or normalization settings are incompatible."""


class KeywordRetrievalError(RetrievalError):
    """Raised when lexical retrieval cannot produce a valid result."""


class BM25IndexError(KeywordRetrievalError):
    """Raised when a BM25 index is invalid or fails during scoring."""


class RerankingError(RetrievalError):
    """Base error for cross-encoder reranking failures."""


class RerankerModelError(RerankingError):
    """Raised when a reranker model cannot load or run inference."""


class RerankerOutputError(RerankingError):
    """Raised when backend scores do not satisfy the reranker contract."""
