"""Optional PostgreSQL corpus persistence and exact dense retrieval."""

from reporecall.persistence.database import (
    DatabaseConfig,
    create_database_engine,
    create_session_factory,
)
from reporecall.persistence.exceptions import (
    PersistenceConfigurationError,
    PersistenceError,
    PersistenceIntegrityError,
    PersistenceQueryError,
)
from reporecall.persistence.repositories import (
    PersistentCorpus,
    PersistentCorpusRepository,
)
from reporecall.persistence.vector_store import PostgresVectorRetriever

__all__ = [
    "DatabaseConfig",
    "PersistenceConfigurationError",
    "PersistenceError",
    "PersistenceIntegrityError",
    "PersistenceQueryError",
    "PersistentCorpus",
    "PersistentCorpusRepository",
    "PostgresVectorRetriever",
    "create_database_engine",
    "create_session_factory",
]
