import ast
from pathlib import Path

import reporecall.models.provenance as models
from reporecall import provenance


def test_no_retrieval_network_generation_or_text_inference_dependencies():
    paths = [Path(models.__file__), *Path(provenance.__file__).parent.glob("*.py")]
    allowed = {
        "__future__",
        "collections",
        "types",
        "enum",
        "re",
        "pydantic",
        "reporecall.models",
        "reporecall.github.models",
        "reporecall.provenance",
    }
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert any(
                    module == name or module.startswith(name + ".") for name in allowed
                ), module
            elif isinstance(node, ast.Import):
                assert all(alias.name in allowed for alias in node.names)
