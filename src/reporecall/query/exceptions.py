class QueryUnderstandingError(Exception):
    """Base error for query-understanding failures."""


class QueryUnderstandingBackendError(QueryUnderstandingError):
    """Raised when a provider backend cannot interpret a query."""


class QueryUnderstandingValidationError(QueryUnderstandingError):
    """Raised when backend output cannot be converted into domain models."""
