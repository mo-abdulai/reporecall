"""Keep transport and algorithms separate as the public API grows."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "reporecall"


def imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (name.name for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""


def test_core_does_not_depend_on_http():
    for directory in (
        "models",
        "persistence",
        "retrieval",
        "provenance",
        "query",
        "services",
    ):
        for path in (ROOT / directory).rglob("*.py"):
            assert not any(
                name.startswith(("fastapi", "starlette", "uvicorn"))
                for name in imports(path)
            ), path


def test_routes_do_not_execute_algorithms_or_sql():
    for path in (ROOT / "api" / "routes").rglob("*.py"):
        assert not any(
            name.startswith(
                (
                    "sqlalchemy",
                    "psycopg",
                    "reporecall.retrieval",
                    "reporecall.provenance",
                    "reporecall.generation",
                )
            )
            for name in imports(path)
        )


def test_services_do_not_generate_answers():
    for path in (ROOT / "services").rglob("*.py"):
        assert not any(
            name.startswith(("reporecall.generation", "reporecall.evaluation"))
            for name in imports(path)
        )
