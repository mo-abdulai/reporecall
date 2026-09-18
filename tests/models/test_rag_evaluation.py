"""Validation of explicitly synthetic RAG evaluation records."""

import pytest
from pydantic import ValidationError

from reporecall.evaluation.rag_judge_backend import RAG_JUDGE_RUBRIC
from reporecall.evaluation.rag_metrics import render_judge_evidence
from reporecall.models import (
    ExpectedFact,
    RAGCitationDiagnostics,
    RAGEvaluationBenchmark,
    RAGEvaluationCase,
    RAGJudgeRequest,
    RAGJudgeResult,
)
from tests.evaluation.rag_fixtures import judgment, make_bundle, make_case, make_sample


@pytest.mark.parametrize("field", ["fact_id", "text"])
@pytest.mark.parametrize("value", ["", " \n"])
def test_nonblank_facts(field, value):
    with pytest.raises(ValidationError):
        ExpectedFact(**{"fact_id": "F1", "text": "Synthetic fact.", field: value})


@pytest.mark.parametrize(
    "updates", [{"expected_facts": []}, {"case_id": " "}, {"query": "\n"}]
)
def test_case_constraints(updates):
    with pytest.raises(ValidationError):
        RAGEvaluationCase.model_validate({**make_case().model_dump(), **updates})


def test_duplicate_facts_cases_and_empty_benchmark():
    case = make_case()
    with pytest.raises(ValidationError, match="unique"):
        RAGEvaluationCase.model_validate(
            {**case.model_dump(), "expected_facts": [case.expected_facts[0]] * 2}
        )
    for cases in ((), (case, case)):
        with pytest.raises(ValidationError):
            RAGEvaluationBenchmark(benchmark_id="synthetic", cases=cases)


@pytest.mark.parametrize("field", ["answer_relevance", "faithfulness"])
@pytest.mark.parametrize("value", [-1, 5, 2.5, True, "4"])
def test_strict_ordinal_ratings(field, value):
    with pytest.raises(ValidationError):
        RAGJudgeResult.model_validate({**judgment().model_dump(), field: value})


@pytest.mark.parametrize(
    "updates",
    [
        {"judge_model": " "},
        {"rubric": ""},
        {"brief_note": "x" * 501},
        {"analysis": "forbidden"},
    ],
)
def test_judge_provenance_and_notes(updates):
    with pytest.raises(ValidationError):
        RAGJudgeResult.model_validate({**judgment().model_dump(), **updates})


def test_immutable_inputs_and_exact_text():
    case = make_case()
    assert case.query == "  What happened in the synthetic example?\n"
    for item, field, value in (
        (case, "query", "changed"),
        (case.expected_facts[0], "text", "changed"),
        (make_sample(answer=" \n"), "answer", "changed"),
        (judgment(), "faithfulness", 0),
    ):
        with pytest.raises(ValidationError, match="frozen"):
            setattr(item, field, value)
    assert make_sample(answer=" \n").answer == " \n"


@pytest.mark.parametrize(
    "cited,valid,unknown",
    [
        (("R1", "R1"), ("R1",), ()),
        (("R1",), ("R1",), ("R1",)),
        (("R1", "X1"), ("X1", "R1"), ()),
        (("R1",), (), ()),
        (("R1",), ("R1", "R1"), ()),
    ],
)
def test_diagnostics_partition(cited, valid, unknown):
    with pytest.raises(ValidationError):
        RAGCitationDiagnostics(
            cited_labels=cited, valid_cited_labels=valid, unknown_cited_labels=unknown
        )


@pytest.mark.parametrize(
    "change",
    ["duplicate_facts", "duplicate_evidence", "unknown_label", "duplicate_label"],
)
def test_request_identity_validation(change):
    case = make_case()
    values = {
        "query": case.query,
        "answer": "[R1]",
        "expected_facts": case.expected_facts,
        "reference_answer": None,
        "evidence": render_judge_evidence(make_bundle()),
        "valid_cited_labels": ("R1",),
        "context_truncated": False,
        "rubric": RAG_JUDGE_RUBRIC,
    }
    if change == "duplicate_facts":
        values["expected_facts"] = (case.expected_facts[0],) * 2
    elif change == "duplicate_evidence":
        values["evidence"] = values["evidence"] * 2
    elif change == "unknown_label":
        values["valid_cited_labels"] = ("R99",)
    else:
        values["valid_cited_labels"] = ("R1", "R1")
    with pytest.raises(ValidationError):
        RAGJudgeRequest(**values)
