"""UTF-8 JSON storage for explicitly supplied benchmarks."""

from pathlib import Path

from reporecall.models.evaluation import RetrievalBenchmark


def load_retrieval_benchmark(path: str | Path) -> RetrievalBenchmark:
    """Load and validate human- or benchmark-authored relevance judgments."""
    return RetrievalBenchmark.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def save_retrieval_benchmark(benchmark: RetrievalBenchmark, path: str | Path) -> None:
    """Write stable JSON without timestamps or generated relevance judgments."""
    Path(path).write_text(benchmark.model_dump_json(indent=2) + "\n", encoding="utf-8")
