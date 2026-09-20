"""Explicit synchronous database lifecycle; importing never opens a connection."""

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator
from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.orm import Session, sessionmaker

from reporecall.persistence.exceptions import PersistenceConfigurationError


class DatabaseConfig(BaseModel):
    """PostgreSQL/Psycopg connection settings with credentials hidden in repr."""

    url: SecretStr
    echo: bool = False
    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: SecretStr) -> SecretStr:
        try:
            parsed = make_url(value.get_secret_value())
        except ArgumentError as exc:
            raise ValueError(
                "A valid PostgreSQL + Psycopg database URL is required."
            ) from exc
        if parsed.drivername != "postgresql+psycopg" or not parsed.database:
            raise ValueError("Use postgresql+psycopg:// with an explicit database.")
        return value


def create_database_engine(config: DatabaseConfig) -> Engine:
    """Create a lazy engine with SQLAlchemy pooling and consistent read snapshots."""
    if not isinstance(config, DatabaseConfig):
        raise PersistenceConfigurationError("An explicit DatabaseConfig is required.")
    return create_engine(
        config.url.get_secret_value(),
        echo=config.echo,
        hide_parameters=True,
        pool_pre_ping=True,
        isolation_level="REPEATABLE READ",
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return injectable, short-lived sessions; callers own transaction scopes."""
    return sessionmaker(engine, expire_on_commit=False)
