"""Evaluation consumes final evidence/answers; provider execution is isolated."""

import ast
from pathlib import Path

import reporecall.evaluation as package


def test_core_rag_evaluation_boundaries():
    root = Path(package.__file__).parent
    allowed = (
        "reporecall.models",
        "reporecall.evaluation",
        "reporecall.provenance",
        "collections",
        "typing",
        "pydantic",
        "pathlib",
    )
    for name in (
        "rag_evaluator.py",
        "rag_metrics.py",
        "rag_benchmark.py",
        "rag_judge_backend.py",
    ):
        tree = ast.parse((root / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").startswith(allowed)
            elif isinstance(node, ast.Import):
                assert all(alias.name.startswith(allowed) for alias in node.names)
    model = root.parent / "models" / "rag_evaluation.py"
    for node in ast.walk(ast.parse(model.read_text())):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").startswith(
                ("reporecall.models", "pydantic", "enum", "math", "typing")
            )


def test_provider_does_not_generate_answers_or_retrieve():
    source = (Path(package.__file__).parent / "openai_rag_judge.py").read_text()
    for forbidden in (
        "RAGPipeline",
        "OpenAILLMBackend",
        "QueryUnderstanding",
        "VectorRetriever",
        "EmbeddingBackend",
        "HybridRetriever",
        "CrossEncoder",
        "Faiss",
        "BM25",
        "web_search",
        "file_search",
    ):
        assert forbidden not in source
