"""Evaluate supplied answers without running retrieval or answer generation."""

from collections.abc import Sequence

from pydantic import ValidationError

from reporecall.evaluation.rag_judge_backend import (
    RAG_JUDGE_RUBRIC,
    RAGEvaluationError,
    RAGEvaluationJudgeBackend,
    RAGJudgeBackendError,
    RAGJudgeOutputError,
)
from reporecall.evaluation.rag_metrics import (
    citation_diagnostics,
    render_judge_evidence,
)
from reporecall.models.rag_evaluation import (
    RAGEvaluationBenchmark,
    RAGEvaluationCase,
    RAGEvaluationReport,
    RAGEvaluationResult,
    RAGEvaluationSample,
    RAGJudgeRequest,
    RAGJudgeResult,
)


def validate_judge_output(
    result: RAGJudgeResult, request: RAGJudgeRequest, model_name: str
) -> RAGJudgeResult:
    """Revalidate backend objects and require exact expected assessment coverage."""
    try:
        validated = RAGJudgeResult.model_validate(result.model_dump())
    except (AttributeError, ValidationError) as exc:
        raise RAGJudgeOutputError("Invalid structured judge output.") from exc
    if validated.judge_model != model_name or validated.rubric != request.rubric:
        raise RAGJudgeOutputError(
            "Judge model/rubric identity does not match the request."
        )
    facts = {f.fact_id: f for f in validated.fact_assessments}
    citations = {c.citation_label: c for c in validated.citation_assessments}
    if set(facts) != {f.fact_id for f in request.expected_facts}:
        raise RAGJudgeOutputError("Judge must assess every expected fact exactly once.")
    if set(citations) != set(request.valid_cited_labels):
        raise RAGJudgeOutputError("Judge must assess exactly the valid cited labels.")
    return validated.model_copy(
        update={
            "fact_assessments": tuple(facts[f.fact_id] for f in request.expected_facts),
            "citation_assessments": tuple(
                citations[label] for label in request.valid_cited_labels
            ),
        }
    )


class RAGEvaluator:
    """Combine deterministic checks and a separately identified semantic judge."""

    def __init__(self, *, judge_backend: RAGEvaluationJudgeBackend) -> None:
        self.judge_backend = judge_backend

    def evaluate_case(
        self, case: RAGEvaluationCase, sample: RAGEvaluationSample
    ) -> RAGEvaluationResult:
        """Validate data, judge supplied output, and retain all evaluation provenance."""
        try:
            RAGEvaluationCase.model_validate(case.model_dump())
            RAGEvaluationSample.model_validate(sample.model_dump())
        except ValidationError as exc:
            raise RAGEvaluationError("Invalid evaluation case or sample.") from exc
        if case.case_id != sample.case_id or case.query != sample.citation_bundle.query:
            raise RAGEvaluationError(
                "Case/sample identity and bundle query must agree."
            )
        model_name = self.judge_backend.model_name
        if not model_name.strip():
            raise RAGEvaluationError("Judge model name must not be blank.")
        diagnostics = citation_diagnostics(sample.answer, sample.citation_bundle)
        request = RAGJudgeRequest(
            query=case.query,
            answer=sample.answer,
            expected_facts=case.expected_facts,
            reference_answer=case.reference_answer,
            evidence=render_judge_evidence(sample.citation_bundle),
            valid_cited_labels=diagnostics.valid_cited_labels,
            context_truncated=sample.citation_bundle.context_truncated,
            rubric=RAG_JUDGE_RUBRIC,
        )
        try:
            raw = self.judge_backend.judge(request)
        except RAGEvaluationError:
            raise
        except Exception as exc:
            # Provider/plugin execution boundary: retain the cause, never invent scores.
            raise RAGJudgeBackendError("Semantic judge execution failed.") from exc
        result = validate_judge_output(raw, request, model_name)
        return RAGEvaluationResult(
            case=case,
            sample=sample,
            deterministic_citation_validation=diagnostics,
            semantic_judgment=result,
        )

    def evaluate_benchmark(
        self,
        benchmark: RAGEvaluationBenchmark,
        *,
        system_name: str,
        samples: Sequence[RAGEvaluationSample],
    ) -> RAGEvaluationReport:
        """Require every case; a failed judge aborts instead of silently skipping it."""
        lookup = {s.case_id: s for s in samples}
        if len(lookup) != len(samples) or set(lookup) != {
            c.case_id for c in benchmark.cases
        }:
            raise RAGEvaluationError(
                "Samples must cover every benchmark case exactly once."
            )
        if not system_name.strip():
            raise RAGEvaluationError("System name must not be blank.")
        return RAGEvaluationReport(
            benchmark=benchmark,
            system_name=system_name,
            per_case=tuple(
                self.evaluate_case(case, lookup[case.case_id])
                for case in benchmark.cases
            ),
        )
