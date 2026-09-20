"""Explicit migration entry point; never imported as application startup logic."""

from alembic import context
from pydantic import SecretStr
from sqlalchemy import text

from reporecall.config import get_settings
from reporecall.persistence.database import DatabaseConfig, create_database_engine
from reporecall.persistence.exceptions import PersistenceConfigurationError
from reporecall.persistence.orm import Base

config = context.config


def configured_database() -> DatabaseConfig:
    url = config.get_main_option("sqlalchemy.url") or get_settings().database_url
    if not url:
        raise PersistenceConfigurationError("Set DATABASE_URL before running Alembic.")
    return DatabaseConfig(url=SecretStr(url))


def run_online() -> None:
    supplied = config.attributes.get("connection")
    if supplied is not None:
        context.configure(
            connection=supplied,
            target_metadata=Base.metadata,
            compare_type=True,
            version_table_schema=supplied.scalar(text("SELECT current_schema()")),
        )
        with context.begin_transaction():
            context.run_migrations()
        return
    engine = create_database_engine(configured_database())
    try:
        with engine.begin() as connection:
            context.configure(
                connection=connection,
                target_metadata=Base.metadata,
                compare_type=True,
                version_table_schema=connection.scalar(text("SELECT current_schema()")),
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    context.configure(
        url=configured_database().url.get_secret_value(),
        target_metadata=Base.metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    run_online()
