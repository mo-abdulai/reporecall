"""Migration-created schema, extension, and reversible ownership boundaries."""

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text

from reporecall.persistence.orm import Base
from tests.persistence.conftest import migrate


@pytest.mark.postgres
def test_migration_upgrade_downgrade_and_schema_matches(database):
    engine, _ = database
    expected = {
        "alembic_version",
        "engineering_events",
        "retrieval_documents",
        "retrieval_chunks",
        "chunk_embeddings",
    }
    assert set(inspect(engine).get_table_names(schema=engine.dialect.default_schema_name)) == expected
    with engine.connect() as connection:
        assert connection.scalar(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
        assert (
            compare_metadata(MigrationContext.configure(connection), Base.metadata)
            == []
        )
        for name in expected - {"alembic_version"}:
            assert all(
                index.get("dialect_options", {}).get("postgresql_using")
                not in {"hnsw", "ivfflat"}
                for index in inspect(connection).get_indexes(name)
            )
    migrate(engine, "downgrade")
    assert set(inspect(engine).get_table_names(schema=engine.dialect.default_schema_name)) == {"alembic_version"}
    migrate(engine)
    assert set(inspect(engine).get_table_names(schema=engine.dialect.default_schema_name)) == expected
