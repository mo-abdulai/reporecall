"""UTF-8 JSON storage for explicitly supplied benchmarks."""

from pathlib import Path

from pydantic import BaseModel

from reporecall.models.evaluation import RetrievalBenchmark


def load_retrieval_benchmark(path: str | Path) -> RetrievalBenchmark:
    """Load and validate human- or benchmark-authored relevance judgments."""
    return _load_benchmark(path, RetrievalBenchmark)


def save_retrieval_benchmark(benchmark: RetrievalBenchmark, path: str | Path) -> None:
    """Write stable JSON without timestamps or generated relevance judgments."""
    _save_benchmark(benchmark, path)


def _load_benchmark[BenchmarkT: BaseModel](
    path: str | Path, model: type[BenchmarkT]
) -> BenchmarkT:
    return model.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _save_benchmark(benchmark: BaseModel, path: str | Path) -> None:
    Path(path).write_text(benchmark.model_dump_json(indent=2) + "\n", encoding="utf-8")
