"""Human reference facts, judge signals, and auditable RAG evaluation records."""

from enum import Enum
from math import fsum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reporecall.models.citation_validation import validate_citation_labels
from reporecall.models.context_expansion import ContextExpansionReason
from reporecall.models.evaluation import MetricValue, Nonblank
from reporecall.models.provenance import CitationBundle, CitationIdentifier
from reporecall.models.relationships import ArtifactReference
from reporecall.models.retrieval_documents import RetrievalSectionType

type EvaluationRating = Annotated[int, Field(strict=True, ge=0, le=4)]
type BriefNote = Annotated[str, Field(max_length=500)]


class ExpectedFact(BaseModel):
    """An explicitly supplied human/benchmark reference fact, never judge-created."""

    fact_id: Nonblank
    text: Nonblank
    model_config = ConfigDict(frozen=True, extra="forbid")


class RAGEvaluationCase(BaseModel):
    """Human-authored correctness target with exact query text."""

    case_id: Nonblank
    query: Nonblank
    expected_facts: tuple[ExpectedFact, ...] = Field(min_length=1)
    reference_answer: str | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def validate_facts(self) -> "RAGEvaluationCase":
        if len({f.fact_id for f in self.expected_facts}) != len(self.expected_facts):
            raise ValueError("Expected fact IDs must be unique.")
        return self


class RAGEvaluationBenchmark(BaseModel):
    """Nonempty supplied ground truth in stable author-defined case order."""

    benchmark_id: Nonblank
    cases: tuple[RAGEvaluationCase, ...] = Field(min_length=1)
    version: str | None = None
    description: str | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def validate_cases(self) -> "RAGEvaluationBenchmark":
        if len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError("Benchmark case IDs must be unique.")
        return self


class RAGEvaluationSample(BaseModel):
    """Supplied system output; a blank answer is valid evaluable output."""

    case_id: Nonblank
    answer: str
    citation_bundle: CitationBundle
    answer_model: Nonblank | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")


class ExpectedFactStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    MISSING = "missing"
    CONTRADICTED = "contradicted"


class CitationSupportStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    UNVERIFIABLE = "unverifiable"


class ExpectedFactAssessment(BaseModel):
    """Judge assessment of the answer relative to one human expected fact."""

    fact_id: Nonblank
    status: ExpectedFactStatus
    brief_note: BriefNote | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")


class CitationSupportAssessment(BaseModel):
    """Semantic support signal for a valid label actually cited in the answer."""

    citation_label: str = Field(pattern=r"^[RX][1-9][0-9]*$")
    status: CitationSupportStatus
    brief_note: BriefNote | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")


class JudgeEvidenceItem(BaseModel):
    """Exact chunk text and structural provenance, excluding ranking diagnostics."""

    citation: CitationIdentifier
    content: str
    artifact: ArtifactReference | None
    section_type: RetrievalSectionType
    expanded_from: tuple[str, ...] = ()
    expansion_reasons: tuple[ContextExpansionReason, ...] = ()
    model_config = ConfigDict(frozen=True, extra="forbid")


class RAGJudgeRequest(BaseModel):
    """Data-only semantic evaluation inputs, with the explicit fixed rubric."""

    query: Nonblank
    answer: str
    expected_facts: tuple[ExpectedFact, ...] = Field(min_length=1)
    reference_answer: str | None
    evidence: tuple[JudgeEvidenceItem, ...]
    valid_cited_labels: tuple[str, ...]
    context_truncated: bool
    rubric: Nonblank
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def validate_identity(self) -> "RAGJudgeRequest":
        fact_ids = [f.fact_id for f in self.expected_facts]
        labels = [e.citation.label for e in self.evidence]
        if len(set(fact_ids)) != len(fact_ids) or len(set(labels)) != len(labels):
            raise ValueError("Judge fact and evidence identities must be unique.")
        if len(set(self.valid_cited_labels)) != len(self.valid_cited_labels) or not set(
            self.valid_cited_labels
        ) <= set(labels):
            raise ValueError("Judge may assess only unique supplied evidence labels.")
        return self


class RAGJudgeBackendResult(BaseModel):
    """Untrusted structured semantic verdicts; not human ground truth."""

    answer_relevance: EvaluationRating
    faithfulness: EvaluationRating
    fact_assessments: tuple[ExpectedFactAssessment, ...] = Field(min_length=1)
    citation_assessments: tuple[CitationSupportAssessment, ...]
    brief_note: BriefNote | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def validate_unique_assessments(self) -> "RAGJudgeBackendResult":
        if len({f.fact_id for f in self.fact_assessments}) != len(
            self.fact_assessments
        ):
            raise ValueError("Duplicate fact assessments.")
        if len({c.citation_label for c in self.citation_assessments}) != len(
            self.citation_assessments
        ):
            raise ValueError("Duplicate citation assessments.")
        return self


class RAGJudgeResult(RAGJudgeBackendResult):
    """Semantic instrument output with explicit model and rubric provenance."""

    judge_model: Nonblank
    rubric: Nonblank


class RAGCitationDiagnostics(BaseModel):
    """Deterministic canonical-marker membership, separate from support judgments."""

    cited_labels: tuple[str, ...]
    valid_cited_labels: tuple[str, ...]
    unknown_cited_labels: tuple[str, ...]
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def validate_partition(self) -> "RAGCitationDiagnostics":
        cited, valid, unknown = map(
            set, (self.cited_labels, self.valid_cited_labels, self.unknown_cited_labels)
        )
        if (
            len(cited) != len(self.cited_labels)
            or valid & unknown
            or valid | unknown != cited
        ):
            raise ValueError("Citation diagnostics must partition unique cited labels.")
        if self.valid_cited_labels != tuple(
            x for x in self.cited_labels if x in valid
        ) or self.unknown_cited_labels != tuple(
            x for x in self.cited_labels if x in unknown
        ):
            raise ValueError("Citation diagnostics must preserve occurrence order.")
        return self

    @property
    def citation_validity_rate(self) -> float | None:
        return (
            len(self.valid_cited_labels) / len(self.cited_labels)
            if self.cited_labels
            else None
        )


class RAGSemanticMetrics(BaseModel):
    """Deterministic arithmetic over model judgments, still judge-dependent signals."""

    strict_fact_recall: MetricValue
    covered_or_partial_rate: MetricValue
    contradiction_rate: MetricValue
    missing_fact_rate: MetricValue
    citation_support_rate: MetricValue | None
    citation_supported_count: int = Field(ge=0)
    citation_partial_count: int = Field(ge=0)
    citation_unsupported_count: int = Field(ge=0)
    citation_unverifiable_count: int = Field(ge=0)
    model_config = ConfigDict(frozen=True, extra="forbid")


def semantic_metrics(judge: RAGJudgeResult) -> RAGSemanticMetrics:
    """No fractional credit: strict rates count only fully supported items."""
    facts = [f.status for f in judge.fact_assessments]
    citations = [c.status for c in judge.citation_assessments]
    supported = facts.count(ExpectedFactStatus.SUPPORTED)
    return RAGSemanticMetrics(
        strict_fact_recall=supported / len(facts),
        covered_or_partial_rate=(supported + facts.count(ExpectedFactStatus.PARTIAL))
        / len(facts),
        contradiction_rate=facts.count(ExpectedFactStatus.CONTRADICTED) / len(facts),
        missing_fact_rate=facts.count(ExpectedFactStatus.MISSING) / len(facts),
        citation_support_rate=citations.count(CitationSupportStatus.SUPPORTED)
        / len(citations)
        if citations
        else None,
        citation_supported_count=citations.count(CitationSupportStatus.SUPPORTED),
        citation_partial_count=citations.count(CitationSupportStatus.PARTIAL),
        citation_unsupported_count=citations.count(CitationSupportStatus.UNSUPPORTED),
        citation_unverifiable_count=citations.count(CitationSupportStatus.UNVERIFIABLE),
    )


class RAGEvaluationResult(BaseModel):
    """Auditable case, exact sample, deterministic diagnostics, and judge signals."""

    case: RAGEvaluationCase
    sample: RAGEvaluationSample
    deterministic_citation_validation: RAGCitationDiagnostics
    semantic_judgment: RAGJudgeResult
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def validate_result(self) -> "RAGEvaluationResult":
        if (
            self.case.case_id != self.sample.case_id
            or self.case.query != self.sample.citation_bundle.query
        ):
            raise ValueError("Evaluation case/sample identity and query must agree.")
        parsed = validate_citation_labels(self.sample.answer, self.sample.citation_bundle)
        diagnostics = self.deterministic_citation_validation
        if (
            diagnostics.cited_labels != parsed.referenced_labels
            or diagnostics.unknown_cited_labels != parsed.unknown_labels
        ):
            raise ValueError("Citation diagnostics must match the supplied answer and bundle.")
        if tuple(f.fact_id for f in self.semantic_judgment.fact_assessments) != tuple(
            f.fact_id for f in self.case.expected_facts
        ):
            raise ValueError("Every expected fact must be assessed once in case order.")
        if (
            tuple(c.citation_label for c in self.semantic_judgment.citation_assessments)
            != self.deterministic_citation_validation.valid_cited_labels
        ):
            raise ValueError(
                "Every valid cited label must be assessed once in occurrence order."
            )
        return self

    @property
    def case_id(self) -> str:
        return self.case.case_id

    @property
    def semantic_metrics(self) -> RAGSemanticMetrics:
        return semantic_metrics(self.semantic_judgment)


class RAGAggregateMetrics(BaseModel):
    """Equal-case macro averages; citation means use defined values only."""

    mean_answer_relevance: float = Field(ge=0, le=4, allow_inf_nan=False)
    mean_faithfulness: float = Field(ge=0, le=4, allow_inf_nan=False)
    mean_strict_fact_recall: MetricValue
    mean_covered_or_partial_rate: MetricValue
    mean_contradiction_rate: MetricValue
    mean_missing_fact_rate: MetricValue
    mean_citation_validity_rate: MetricValue | None
    mean_citation_support_rate: MetricValue | None
    citation_validity_case_count: int = Field(ge=0)
    citation_support_case_count: int = Field(ge=0)
    model_config = ConfigDict(frozen=True, extra="forbid")


def _mean(values: list[float]) -> float | None:
    return fsum(values) / len(values) if values else None


class RAGEvaluationReport(BaseModel):
    """Complete benchmark results under one judge model and rubric."""

    benchmark: RAGEvaluationBenchmark
    system_name: Nonblank
    per_case: tuple[RAGEvaluationResult, ...] = Field(min_length=1)
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def validate_report(self) -> "RAGEvaluationReport":
        if tuple(r.case for r in self.per_case) != self.benchmark.cases:
            raise ValueError(
                "Report must contain each benchmark case exactly once in order."
            )
        if (
            len(
                {
                    (r.semantic_judgment.judge_model, r.semantic_judgment.rubric)
                    for r in self.per_case
                }
            )
            != 1
        ):
            raise ValueError(
                "A report requires the same judge model and rubric for every case."
            )
        return self

    @property
    def benchmark_id(self) -> str:
        return self.benchmark.benchmark_id

    @property
    def case_count(self) -> int:
        return len(self.per_case)

    @property
    def aggregate(self) -> RAGAggregateMetrics:
        metrics = [r.semantic_metrics for r in self.per_case]
        validity = [
            r.deterministic_citation_validation.citation_validity_rate
            for r in self.per_case
        ]
        valid_values = [v for v in validity if v is not None]
        support = [
            m.citation_support_rate
            for m in metrics
            if m.citation_support_rate is not None
        ]
        n = len(metrics)
        return RAGAggregateMetrics(
            mean_answer_relevance=fsum(
                r.semantic_judgment.answer_relevance for r in self.per_case
            )
            / n,
            mean_faithfulness=fsum(
                r.semantic_judgment.faithfulness for r in self.per_case
            )
            / n,
            mean_strict_fact_recall=fsum(m.strict_fact_recall for m in metrics) / n,
            mean_covered_or_partial_rate=fsum(
                m.covered_or_partial_rate for m in metrics
            )
            / n,
            mean_contradiction_rate=fsum(m.contradiction_rate for m in metrics) / n,
            mean_missing_fact_rate=fsum(m.missing_fact_rate for m in metrics) / n,
            mean_citation_validity_rate=_mean(valid_values),
            mean_citation_support_rate=_mean(support),
            citation_validity_case_count=len(valid_values),
            citation_support_case_count=len(support),
        )


class RAGSystemComparison(BaseModel):
    """Same-ground-truth and same-judge comparisons in system-name order."""

    reports: tuple[RAGEvaluationReport, ...] = Field(min_length=1)
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def validate_comparison(self) -> "RAGSystemComparison":
        first = self.reports[0]
        if len({r.system_name for r in self.reports}) != len(self.reports):
            raise ValueError("Compared system names must be unique.")
        for report in self.reports:
            if report.benchmark != first.benchmark:
                raise ValueError("Comparison requires identical human ground truth.")
            a, b = (
                report.per_case[0].semantic_judgment,
                first.per_case[0].semantic_judgment,
            )
            if (a.judge_model, a.rubric) != (b.judge_model, b.rubric):
                raise ValueError("Comparison requires the same judge model and rubric.")
        object.__setattr__(
            self, "reports", tuple(sorted(self.reports, key=lambda r: r.system_name))
        )
        return self
