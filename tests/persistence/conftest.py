"""Opt-in real PostgreSQL tests, isolated to a newly created temporary schema."""

import os
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from pydantic import SecretStr
from sqlalchemy.engine import make_url
from sqlalchemy.schema import CreateSchema, DropSchema

from reporecall.persistence import (
    DatabaseConfig,
    create_database_engine,
    create_session_factory,
)

ROOT = Path(__file__).resolve().parents[2]


def migrate(engine, direction="upgrade"):
    config = Config(str(ROOT / "alembic.ini"))
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        if direction == "upgrade":
            command.upgrade(config, "head")
        else:
            command.downgrade(config, "base")


@pytest.fixture
def database():
    raw_url = os.environ.get("TEST_DATABASE_URL")
    if not raw_url:
        pytest.skip("PostgreSQL integration requires explicit TEST_DATABASE_URL")
    # Never fall back to DATABASE_URL or touch preexisting corpus tables.
    admin = create_database_engine(DatabaseConfig(url=SecretStr(raw_url)))
    schema = "reporecall_test_" + uuid4().hex
    with admin.begin() as connection:
        connection.execute(CreateSchema(schema))
    url = make_url(raw_url).update_query_dict(
        {"options": f"-csearch_path={schema},public"}
    )
    engine = create_database_engine(
        DatabaseConfig(url=SecretStr(url.render_as_string(hide_password=False)))
    )
    try:
        migrate(engine)
        yield engine, create_session_factory(engine)
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True))
        admin.dispose()
