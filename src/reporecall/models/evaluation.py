"""Explicit ground truth and deterministic chunk-level evaluation records."""

from math import fsum
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("Evaluation identifiers and queries must not be blank.")
    return value


type Nonblank = Annotated[str, AfterValidator(_nonblank)]
type PositiveInteger = Annotated[int, Field(gt=0, strict=True)]
type MetricValue = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class RelevanceJudgment(BaseModel):
    """Author-supplied grade: 0 nonrelevant, 1 relevant, 2 highly relevant."""

    chunk_id: Nonblank
    relevance: int = Field(ge=0, le=2, strict=True)
    model_config = ConfigDict(frozen=True, extra="forbid")


class RetrievalEvaluationCase(BaseModel):
    """An exact benchmark query and explicit chunk-level judgments."""

    case_id: Nonblank
    query: Nonblank
    judgments: tuple[RelevanceJudgment, ...] = Field(min_length=1)
    description: str | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("judgments")
    @classmethod
    def validate_judgments(
        cls, values: tuple[RelevanceJudgment, ...]
    ) -> tuple[RelevanceJudgment, ...]:
        if len({item.chunk_id for item in values}) != len(values):
            raise ValueError("Judgment chunk IDs must be unique.")
        if not any(item.relevance > 0 for item in values):
            raise ValueError(
                "An evaluation case requires at least one relevant judgment."
            )
        return tuple(sorted(values, key=lambda item: item.chunk_id))


class RetrievalBenchmark(BaseModel):
    """A nonempty author-supplied benchmark; case order is preserved."""

    benchmark_id: Nonblank
    cases: tuple[RetrievalEvaluationCase, ...] = Field(min_length=1)
    description: str | None = None
    version: str | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("cases")
    @classmethod
    def validate_cases(
        cls, values: tuple[RetrievalEvaluationCase, ...]
    ) -> tuple[RetrievalEvaluationCase, ...]:
        if len({case.case_id for case in values}) != len(values):
            raise ValueError("Benchmark case IDs must be unique.")
        return values


class RankedRetrievedItem(BaseModel):
    """Canonical retrieved identity and its established one-based rank."""

    chunk_id: Nonblank
    rank: PositiveInteger
    model_config = ConfigDict(frozen=True, extra="forbid")


class RetrievalMetricsAtK(BaseModel):
    """Finite metrics at a cutoff; reciprocal_rank is MRR when aggregated."""

    k: PositiveInteger
    precision: MetricValue
    recall: MetricValue
    hit_rate: MetricValue
    reciprocal_rank: MetricValue
    ndcg: MetricValue
    model_config = ConfigDict(frozen=True, extra="forbid")


class RetrievalCaseEvaluation(BaseModel):
    """Per-query measurements and original ranked chunk identities."""

    case_id: Nonblank
    query: Nonblank
    ranked_chunk_ids: tuple[Nonblank, ...]
    relevant_count: PositiveInteger
    metrics: tuple[RetrievalMetricsAtK, ...] = Field(min_length=1)
    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def retrieved_count(self) -> int:
        return len(self.ranked_chunk_ids)

    @model_validator(mode="after")
    def validate_result(self) -> "RetrievalCaseEvaluation":
        if len(set(self.ranked_chunk_ids)) != len(self.ranked_chunk_ids):
            raise ValueError("Retrieved chunk IDs must be unique.")
        cutoffs = tuple(metric.k for metric in self.metrics)
        if cutoffs != tuple(sorted(set(cutoffs))):
            raise ValueError("Metric cutoffs must be unique and ascending.")
        return self


class RetrievalEvaluationReport(BaseModel):
    """Complete benchmark measurements, retaining the supplied labels for audit."""

    benchmark: RetrievalBenchmark
    system_name: Nonblank
    per_case: tuple[RetrievalCaseEvaluation, ...] = Field(min_length=1)
    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def benchmark_id(self) -> str:
        return self.benchmark.benchmark_id

    @property
    def case_count(self) -> int:
        return len(self.per_case)

    @property
    def aggregate_metrics(self) -> tuple[RetrievalMetricsAtK, ...]:
        """Equal-weight macro means; no rounding or missing-case exclusion."""
        count = len(self.per_case)
        return tuple(
            RetrievalMetricsAtK(
                k=metric.k,
                precision=fsum(case.metrics[i].precision for case in self.per_case)
                / count,
                recall=fsum(case.metrics[i].recall for case in self.per_case) / count,
                hit_rate=fsum(case.metrics[i].hit_rate for case in self.per_case)
                / count,
                reciprocal_rank=fsum(
                    case.metrics[i].reciprocal_rank for case in self.per_case
                )
                / count,
                ndcg=fsum(case.metrics[i].ndcg for case in self.per_case) / count,
            )
            for i, metric in enumerate(self.per_case[0].metrics)
        )

    @model_validator(mode="after")
    def validate_coverage(self) -> "RetrievalEvaluationReport":
        if tuple(item.case_id for item in self.per_case) != tuple(
            case.case_id for case in self.benchmark.cases
        ):
            raise ValueError(
                "Report must cover every benchmark case exactly once in benchmark order."
            )
        cutoffs = tuple(m.k for m in self.per_case[0].metrics)
        for case, result in zip(self.benchmark.cases, self.per_case, strict=True):
            if result.query != case.query:
                raise ValueError(
                    "Evaluated query must match the benchmark query exactly."
                )
            if result.relevant_count != sum(j.relevance > 0 for j in case.judgments):
                raise ValueError("Relevant count must match supplied judgments.")
            if tuple(m.k for m in result.metrics) != cutoffs:
                raise ValueError("All cases must use the same metric cutoffs.")
        return self


class RetrievalSystemComparison(BaseModel):
    """Comparable reports ordered by explicit system name, without a winner."""

    reports: tuple[RetrievalEvaluationReport, ...] = Field(min_length=1)
    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("reports")
    @classmethod
    def validate_reports(
        cls, reports: tuple[RetrievalEvaluationReport, ...]
    ) -> tuple[RetrievalEvaluationReport, ...]:
        if len({r.system_name for r in reports}) != len(reports):
            raise ValueError("Compared system names must be unique.")
        benchmark = reports[0].benchmark
        cutoffs = tuple(m.k for m in reports[0].aggregate_metrics)
        if any(r.benchmark != benchmark for r in reports):
            raise ValueError(
                "Compared reports must share the same benchmark and judgments."
            )
        if any(tuple(m.k for m in r.aggregate_metrics) != cutoffs for r in reports):
            raise ValueError("Compared reports must share metric cutoffs.")
        return tuple(sorted(reports, key=lambda r: r.system_name))

    @property
    def benchmark_id(self) -> str:
        return self.reports[0].benchmark_id
