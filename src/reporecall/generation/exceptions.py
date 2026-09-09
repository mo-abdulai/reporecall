class GenerationError(Exception):
    """Base error for grounded answer-generation failures."""


class LLMBackendError(GenerationError):
    """Raised when an LLM provider cannot generate a response."""


class RAGContextError(GenerationError):
    """Raised when retrieved evidence cannot form a valid RAG context."""


class RAGResponseError(GenerationError):
    """Raised when generated output violates the baseline response contract."""
