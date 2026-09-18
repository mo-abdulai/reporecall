"""Safe UTF-8 persistence for supplied synthetic benchmark cases."""

import pytest
from pydantic import ValidationError

from reporecall.evaluation import load_rag_benchmark, save_rag_benchmark
from reporecall.models import RAGEvaluationBenchmark
from tests.evaluation.rag_fixtures import make_case


def test_benchmark_json_roundtrip(tmp_path):
    benchmark = RAGEvaluationBenchmark(
        benchmark_id="synthetic-only",
        cases=(make_case(facts=("Synthetic café fact.",)),),
    )
    path = tmp_path / "synthetic.json"
    save_rag_benchmark(benchmark, path)
    assert load_rag_benchmark(path) == benchmark
    first = path.read_bytes()
    save_rag_benchmark(benchmark, str(path))
    assert first == path.read_bytes()
    assert "café" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "payload", ["not JSON", "{}", '{"benchmark_id":"synthetic","cases":[]}']
)
def test_invalid_json(tmp_path, payload):
    path = tmp_path / "invalid.json"
    path.write_text(payload)
    with pytest.raises(ValidationError):
        load_rag_benchmark(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rag_benchmark(tmp_path / "missing.json")
