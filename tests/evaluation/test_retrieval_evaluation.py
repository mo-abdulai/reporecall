"""Synthetic evaluation fixtures only; these labels assert no real-world relevance."""

import ast
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from reporecall.evaluation import (
    RetrievalEvaluationConfig,
    RetrievalEvaluator,
    dcg_at_k,
    load_retrieval_benchmark,
    normalize_ranked_hits,
    save_retrieval_benchmark,
)
from reporecall.github import GitHubRepository
from reporecall.models import (
    EventMetadata,
    HybridSearchHit,
    KeywordSearchHit,
    RankedRetrievedItem,
    RelevanceJudgment,
    RerankedSearchHit,
    RetrievalBenchmark,
    RetrievalChunk,
    RetrievalEvaluationCase,
    RetrievalEvaluationReport,
    RetrievalMetricsAtK,
    RetrievalSystemComparison,
    VectorSearchHit,
)


def case(judgments=None, case_id="synthetic-1", query="  synthetic query\n"):
    return RetrievalEvaluationCase(
        case_id=case_id,
        query=query,
        description="Synthetic labels for arithmetic tests only",
        judgments=tuple(
            RelevanceJudgment(chunk_id=c, relevance=r)
            for c, r in (judgments or {"A": 1, "C": 1}).items()
        ),
    )


def ranked(*ids):
    return tuple(RankedRetrievedItem(chunk_id=c, rank=i) for i, c in enumerate(ids, 1))


def benchmark():
    return RetrievalBenchmark(
        benchmark_id="synthetic", cases=(case(), case({"D": 2}, "synthetic-2"))
    )


def report(name="synthetic-system", config=None):
    return RetrievalEvaluator(config).evaluate_benchmark(
        benchmark(),
        system_name=name,
        ranked_results={"synthetic-1": ranked("A", "B", "C"), "synthetic-2": ()},
    )


def test_exact_precision_recall_hit_rate_binary_ndcg():
    result = RetrievalEvaluator(
        RetrievalEvaluationConfig(k_values=(1, 2, 3, 4))
    ).evaluate_case(case(), ranked("A", "B", "C", "D"))
    assert [m.precision for m in result.metrics] == pytest.approx([1, 0.5, 2 / 3, 0.5])
    assert [m.recall for m in result.metrics] == [0.5, 0.5, 1, 1]
    assert [m.hit_rate for m in result.metrics] == [1] * 4
    assert [m.reciprocal_rank for m in result.metrics] == [1] * 4
    assert result.metrics[2].ndcg == pytest.approx(1.5 / (1 + 1 / math.log2(3)))
    assert result.query == "  synthetic query\n"
    assert result.ranked_chunk_ids == ("A", "B", "C", "D")
    assert result.retrieved_count == 4
    assert result.relevant_count == 2


def test_reciprocal_rank_cutoff():
    result = RetrievalEvaluator().evaluate_case(case(), ranked("B", "C", "A"))
    assert result.metrics[0].reciprocal_rank == 0
    assert result.metrics[1].reciprocal_rank == 0.5
    late = RetrievalEvaluator().evaluate_case(
        case({"A": 1}), ranked(*[f"X{i}" for i in range(7)], "A")
    )
    assert late.metrics[2].reciprocal_rank == 0
    assert late.metrics[3].reciprocal_rank == 1 / 8


def test_graded_dcg_idcg_ndcg():
    result = RetrievalEvaluator(RetrievalEvaluationConfig(k_values=(2,))).evaluate_case(
        case({"A": 2, "B": 1}), ranked("B", "A")
    )
    dcg = 1 + 3 / math.log2(3)
    ideal = 3 + 1 / math.log2(3)
    assert dcg_at_k([1, 2], 2) == pytest.approx(dcg)
    assert dcg_at_k([2, 1], 2) == pytest.approx(ideal)
    assert result.metrics[0].ndcg == pytest.approx(dcg / ideal)
    assert result.metrics[0].precision == result.metrics[0].recall == 1
    assert result.metrics[0].reciprocal_rank == result.metrics[0].hit_rate == 1
    assert (
        RetrievalEvaluator()
        .evaluate_case(case({"A": 2, "B": 1}), ranked("A", "B"))
        .metrics[1]
        .ndcg
        == 1
    )


@pytest.mark.parametrize("ids", [(), ("X",), ("X", "Y")])
def test_empty_and_unjudged_results(ids):
    result = RetrievalEvaluator().evaluate_case(case(), ranked(*ids))
    for metric in result.metrics:
        assert all(
            getattr(metric, name) == 0
            for name in ("precision", "recall", "hit_rate", "reciprocal_rank", "ndcg")
        )


def test_fewer_than_k_and_explicit_zero():
    result = RetrievalEvaluator(RetrievalEvaluationConfig(k_values=(5,))).evaluate_case(
        case({"A": 1, "X": 0}), ranked("A")
    )
    m = result.metrics[0]
    assert m.precision == 0.2
    assert m.recall == m.hit_rate == m.reciprocal_rank == m.ndcg == 1
    assert result.relevant_count == 1
    assert (
        RetrievalEvaluator()
        .evaluate_case(case({"A": 1, "X": 0}), ranked("X"))
        .metrics[0]
        .ndcg
        == 0
    )


def test_config_order_judgment_order_determinism_and_no_mutation():
    config = RetrievalEvaluationConfig(k_values=(10, 1, 5, 5))
    assert config.k_values == (1, 5, 10)
    a, b = case({"A": 1, "C": 2}), case({"C": 2, "A": 1})
    items = ranked("C", "X", "A")
    before = (a.model_dump(), [i.model_dump() for i in items])
    evaluator = RetrievalEvaluator(config)
    assert evaluator.evaluate_case(a, items) == evaluator.evaluate_case(b, items)
    assert before == (a.model_dump(), [i.model_dump() for i in items])
    assert a == b


@pytest.mark.parametrize("values", [(), (0,), (-1,), (True,), (1.5,), ("1",)])
def test_invalid_k(values):
    with pytest.raises(ValidationError):
        RetrievalEvaluationConfig(k_values=values)


@pytest.mark.parametrize("value", [-1, 3, 1.5, True, "1"])
def test_invalid_relevance(value):
    with pytest.raises(ValidationError):
        RelevanceJudgment(chunk_id="A", relevance=value)


@pytest.mark.parametrize(
    "judgments",
    [
        (),
        (RelevanceJudgment(chunk_id="A", relevance=0),),
        (
            RelevanceJudgment(chunk_id="A", relevance=1),
            RelevanceJudgment(chunk_id="A", relevance=2),
        ),
    ],
)
def test_invalid_judgments(judgments):
    with pytest.raises(ValidationError):
        RetrievalEvaluationCase(case_id="synthetic", query="query", judgments=judgments)


@pytest.mark.parametrize("field", ["case_id", "query"])
def test_nonblank_case_fields(field):
    with pytest.raises(ValidationError):
        RetrievalEvaluationCase.model_validate(case().model_dump() | {field: " \n"})


def test_invalid_benchmark():
    for cases in [(), (case(), case())]:
        with pytest.raises(ValidationError):
            RetrievalBenchmark(benchmark_id="synthetic", cases=cases)
    with pytest.raises(ValidationError):
        RetrievalBenchmark(benchmark_id=" ", cases=(case(),))


@pytest.mark.parametrize("rank", [0, -1, True, 1.5, "1"])
def test_invalid_rank(rank):
    with pytest.raises(ValidationError):
        RankedRetrievedItem(chunk_id="A", rank=rank)


@pytest.mark.parametrize(
    "items",
    [
        ranked("A", "A"),
        (
            RankedRetrievedItem(chunk_id="A", rank=1),
            RankedRetrievedItem(chunk_id="B", rank=1),
        ),
        (
            RankedRetrievedItem(chunk_id="A", rank=1),
            RankedRetrievedItem(chunk_id="B", rank=3),
        ),
        tuple(reversed(ranked("A", "B"))),
    ],
)
def test_malformed_rankings(items):
    with pytest.raises(ValueError):
        RetrievalEvaluator().evaluate_case(case(), items)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), -0.1, 1.1]
)
@pytest.mark.parametrize(
    "field", ["precision", "recall", "hit_rate", "reciprocal_rank", "ndcg"]
)
def test_metric_finiteness_and_bounds(field, value):
    data = {
        "k": 1,
        "precision": 0,
        "recall": 0,
        "hit_rate": 0,
        "reciprocal_rank": 0,
        "ndcg": 0,
    }
    with pytest.raises(ValidationError):
        RetrievalMetricsAtK.model_validate(data | {field: value})


def test_macro_averages_case_counts_and_complete_empty_case():
    result = report()
    assert result.case_count == 2
    assert result.benchmark_id == "synthetic"
    for i, metric in enumerate(result.aggregate_metrics):
        for name in ("precision", "recall", "hit_rate", "reciprocal_rank", "ndcg"):
            assert getattr(metric, name) == pytest.approx(
                sum(getattr(c.metrics[i], name) for c in result.per_case) / 2
            )
    assert result.aggregate_metrics[1].recall == 0.5  # macro, not 2/3 micro recall
    assert result.aggregate_metrics[1].precision == pytest.approx(1 / 3)
    assert result.per_case[1].retrieved_count == 0
    assert (
        RetrievalEvaluationReport.model_validate_json(
            result.model_dump_json(round_trip=True)
        )
        == result
    )
    assert report() == result
    assert report().model_dump_json() == result.model_dump_json()


def test_missing_extra_duplicate_cases_and_wrong_cutoffs():
    evaluator = RetrievalEvaluator()
    for mapping in [
        {},
        {"synthetic-1": ()},
        {"synthetic-1": (), "synthetic-2": (), "extra": ()},
    ]:
        with pytest.raises(ValueError, match="exactly"):
            evaluator.evaluate_benchmark(
                benchmark(), system_name="system", ranked_results=mapping
            )
    result = report()
    for evaluations in [(), (result.per_case[0],) * 2]:
        with pytest.raises(ValueError):
            evaluator.aggregate(
                benchmark=benchmark(), system_name="system", evaluations=evaluations
            )
    with pytest.raises(ValueError, match="K values"):
        RetrievalEvaluator(RetrievalEvaluationConfig(k_values=(2,))).aggregate(
            benchmark=benchmark(), system_name="system", evaluations=result.per_case
        )
    assert (
        evaluator.aggregate(
            benchmark=benchmark(),
            system_name="synthetic-system",
            evaluations=tuple(reversed(result.per_case)),
        )
        == result
    )


def test_comparison_stable_order_and_same_benchmark():
    a, b = report("dense"), report("bm25")
    comparison = RetrievalSystemComparison(reports=(a, b))
    assert [r.system_name for r in comparison.reports] == ["bm25", "dense"]
    assert comparison == RetrievalSystemComparison(reports=(b, a))
    assert "winner" not in RetrievalSystemComparison.model_fields
    assert comparison.benchmark_id == "synthetic"
    with pytest.raises(ValidationError):
        RetrievalSystemComparison(reports=(a, a))
    with pytest.raises(ValidationError):
        RetrievalSystemComparison(
            reports=(a, report("different", RetrievalEvaluationConfig(k_values=(2,))))
        )
    changed = benchmark().model_copy(update={"version": "different-label-set"})
    other = RetrievalEvaluator().evaluate_benchmark(
        changed,
        system_name="other",
        ranked_results={"synthetic-1": (), "synthetic-2": ()},
    )
    with pytest.raises(ValidationError, match="same benchmark"):
        RetrievalSystemComparison(reports=(a, other))


def test_benchmark_utf8_json_roundtrip(tmp_path):
    data = benchmark().model_copy(update={"description": "Synthetic café labels"})
    path = tmp_path / "synthetic.json"
    save_retrieval_benchmark(data, path)
    first = path.read_bytes()
    assert load_retrieval_benchmark(path) == data
    save_retrieval_benchmark(data, path)
    assert first == path.read_bytes()
    assert "café" in path.read_text(encoding="utf-8")
    path.write_text('{"benchmark_id":"invalid","cases":[]}', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_retrieval_benchmark(path)
    path.write_text("not JSON", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_retrieval_benchmark(path)
    with pytest.raises(FileNotFoundError):
        load_retrieval_benchmark(tmp_path / "missing.json")


def test_adapters_all_existing_ranked_types_preserve_order_and_ignore_scores():
    repo = GitHubRepository(owner="synthetic", name="fixture")

    def chunk(identifier):
        return RetrievalChunk(
            chunk_id=identifier,
            document_id="doc",
            event_id="event",
            repository=repo,
            section_id="section",
            section_type="overview",
            chunk_index=0,
            content="Synthetic",
            text="Synthetic",
            metadata=EventMetadata(event_id="event", repository=repo),
        )

    for hit_type in (
        VectorSearchHit,
        KeywordSearchHit,
        HybridSearchHit,
        RerankedSearchHit,
    ):
        hits = []
        for rank, identifier in enumerate(("Z", "A"), 1):
            args = {"chunk": chunk(identifier), "rank": rank}
            if hit_type in (VectorSearchHit, KeywordSearchHit):
                args.update(score=float(rank))  # Deliberately not descending.
            else:
                args.update(
                    rrf_score=1 / (60 + rank),
                    dense_rank=rank,
                    dense_score=0.1,
                    dense_rrf_contribution=1 / (60 + rank),
                    sources=("dense",),
                )
                if hit_type is RerankedSearchHit:
                    args.update(
                        reranker_score=float(rank),
                        reranker_model="synthetic",
                        previous_hybrid_rank=rank,
                    )
            hits.append(hit_type(**args))
        before = [h.model_dump() for h in hits]
        assert normalize_ranked_hits(hits) == ranked("Z", "A")
        assert before == [h.model_dump() for h in hits]
        with pytest.raises(ValueError):
            normalize_ranked_hits(list(reversed(hits)))
    assert normalize_ranked_hits([]) == ()
    with pytest.raises(TypeError):
        normalize_ranked_hits([object()])


def test_immutability():
    objects = [
        (case(), "query", "changed"),
        (RelevanceJudgment(chunk_id="A", relevance=1), "relevance", 2),
        (report(), "system_name", "changed"),
        (ranked("A")[0], "rank", 2),
    ]
    for obj, field, value in objects:
        with pytest.raises(ValidationError):
            setattr(obj, field, value)


@pytest.mark.parametrize(
    "grades,k", [([1], 0), ([1], True), ([3], 1), ([-1], 1), ([True], 1)]
)
def test_dcg_rejects_invalid_input(grades, k):
    with pytest.raises(ValueError):
        dcg_at_k(grades, k)


def test_core_boundaries():
    import reporecall.evaluation as package

    # Retrieval evaluation stays pure; the separate RAG provider has its own boundary.
    paths = [
        Path(package.__file__).parent / name
        for name in ("adapters.py", "benchmark.py", "evaluator.py", "metrics.py")
    ]
    allowed = (
        "reporecall.models",
        "reporecall.evaluation",
        "collections",
        "math",
        "pathlib",
        "pydantic",
    )
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").startswith(allowed)
            elif isinstance(node, ast.Import):
                assert all(alias.name.startswith(allowed) for alias in node.names)
