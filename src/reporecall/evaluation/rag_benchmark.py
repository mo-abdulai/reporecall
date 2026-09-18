"""JSON persistence for supplied RAG reference facts, never generated labels."""

from pathlib import Path

from reporecall.evaluation.benchmark import _load_benchmark, _save_benchmark
from reporecall.models.rag_evaluation import RAGEvaluationBenchmark


def load_rag_benchmark(path: str | Path) -> RAGEvaluationBenchmark:
    """Read and validate a human-authored RAG benchmark."""
    return _load_benchmark(path, RAGEvaluationBenchmark)


def save_rag_benchmark(benchmark: RAGEvaluationBenchmark, path: str | Path) -> None:
    """Write stable UTF-8 JSON with no secrets, timestamps, or guessed facts."""
    _save_benchmark(benchmark, path)
