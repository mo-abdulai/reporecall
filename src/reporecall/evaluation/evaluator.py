"""Read-only evaluation of supplied rankings; no retrieval execution."""

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reporecall.evaluation.adapters import validate_ranked_items
from reporecall.evaluation.metrics import metrics_at_k
from reporecall.models.evaluation import (
    PositiveInteger,
    RankedRetrievedItem,
    RetrievalBenchmark,
    RetrievalCaseEvaluation,
    RetrievalEvaluationCase,
    RetrievalEvaluationReport,
)


class RetrievalEvaluationConfig(BaseModel):
    """Nonempty positive cutoffs, deduplicated and sorted deterministically."""

    k_values: tuple[PositiveInteger, ...] = Field(default=(1, 3, 5, 10), min_length=1)
    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("k_values")
    @classmethod
    def canonicalize_cutoffs(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(sorted(set(values)))


class RetrievalEvaluator:
    """Compute per-query metrics and complete macro-averaged benchmark reports."""

    def __init__(self, config: RetrievalEvaluationConfig | None = None) -> None:
        self.config = config or RetrievalEvaluationConfig()

    def evaluate_case(
        self,
        case: RetrievalEvaluationCase,
        ranked_items: Sequence[RankedRetrievedItem],
    ) -> RetrievalCaseEvaluation:
        """Evaluate supplied ranks without rewriting queries or inferring labels."""
        items = validate_ranked_items(ranked_items)
        return RetrievalCaseEvaluation(
            case_id=case.case_id,
            query=case.query,
            ranked_chunk_ids=tuple(item.chunk_id for item in items),
            relevant_count=sum(j.relevance > 0 for j in case.judgments),
            metrics=tuple(metrics_at_k(case, items, k) for k in self.config.k_values),
        )

    def aggregate(
        self,
        *,
        benchmark: RetrievalBenchmark,
        system_name: str,
        evaluations: Sequence[RetrievalCaseEvaluation],
    ) -> RetrievalEvaluationReport:
        """Require complete cases and configured cutoffs; preserve benchmark order."""
        if len({item.case_id for item in evaluations}) != len(evaluations):
            raise ValueError("Duplicate case evaluations.")
        lookup = {item.case_id: item for item in evaluations}
        if set(lookup) != {case.case_id for case in benchmark.cases}:
            raise ValueError("Evaluations must cover exactly the benchmark cases.")
        if any(
            tuple(m.k for m in item.metrics) != self.config.k_values
            for item in evaluations
        ):
            raise ValueError("Evaluations must match configured K values.")
        return RetrievalEvaluationReport(
            benchmark=benchmark,
            system_name=system_name,
            per_case=tuple(lookup[case.case_id] for case in benchmark.cases),
        )

    def evaluate_benchmark(
        self,
        benchmark: RetrievalBenchmark,
        *,
        system_name: str,
        ranked_results: Mapping[str, Sequence[RankedRetrievedItem]],
    ) -> RetrievalEvaluationReport:
        """Evaluate exactly one supplied ranking per case; missing results are errors.

        Empty rankings are successful zero-hit results. Retrieval failures must
        be handled by callers and must never be disguised as empty rankings.
        """
        if set(ranked_results) != {case.case_id for case in benchmark.cases}:
            raise ValueError("Ranked results must cover exactly the benchmark cases.")
        return self.aggregate(
            benchmark=benchmark,
            system_name=system_name,
            evaluations=tuple(
                self.evaluate_case(case, ranked_results[case.case_id])
                for case in benchmark.cases
            ),
        )
