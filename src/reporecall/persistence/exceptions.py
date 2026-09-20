"""Focused persistence failures; causes remain available for diagnosis."""


class PersistenceError(ValueError):
    """Persistence operation could not complete."""


class PersistenceConfigurationError(PersistenceError):
    """Invalid or missing database configuration."""


class PersistenceIntegrityError(PersistenceError):
    """Canonical identity, payload, or embedding provenance is inconsistent."""


class PersistenceQueryError(PersistenceError):
    """Database connection or query execution failed."""
