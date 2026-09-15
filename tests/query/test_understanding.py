import pytest

from reporecall.models import RetrievalSectionType
from reporecall.query import (
    QueryUnderstandingBackendResult,
    QueryUnderstandingService,
    QueryUnderstandingValidationError,
)


class FakeQueryUnderstandingBackend:
    model_name = "fake-query-model"

    def __init__(self, result: QueryUnderstandingBackendResult) -> None:
        self.result = result
        self.calls: list[str] = []

    def understand(self, query: str) -> QueryUnderstandingBackendResult:
        self.calls.append(query)
        return self.result


def understand_with_result(
    query: str,
    result: QueryUnderstandingBackendResult,
):
    backend = FakeQueryUnderstandingBackend(result)
    return QueryUnderstandingService(backend).understand(query), backend


def test_no_filter_query_returns_none_for_metadata_filter():
    query = "database connection leak after retry worker crash"
    understood, backend = understand_with_result(
        query,
        QueryUnderstandingBackendResult(retrieval_query=query),
    )

    assert backend.calls == [query]
    assert understood.original_query == query
    assert understood.retrieval_query == query
    assert understood.metadata_filter is None
    assert understood.extracted_constraints == ()
    assert understood.model_name == "fake-query-model"


def test_language_extraction_uses_metadata_filter_canonicalization():
    understood, _ = understand_with_result(
        "Find Python authentication regressions",
        QueryUnderstandingBackendResult(
            retrieval_query="authentication regressions",
            languages=("Python",),
        ),
    )

    assert understood.retrieval_query == "authentication regressions"
    assert understood.metadata_filter is not None
    assert understood.metadata_filter.languages == ("python",)
    assert understood.extracted_constraints == ("language:python",)


def test_language_path_and_tests_extraction_is_supported():
    understood, _ = understand_with_result(
        "Find Python authentication fixes in src/api that changed tests",
        QueryUnderstandingBackendResult(
            retrieval_query="authentication fixes",
            languages=("Python",),
            path_prefixes=("src/api",),
            has_test_changes=True,
        ),
    )

    assert understood.retrieval_query == "authentication fixes"
    assert understood.metadata_filter is not None
    assert understood.metadata_filter.languages == ("python",)
    assert understood.metadata_filter.path_prefixes == ("src/api",)
    assert understood.metadata_filter.has_test_changes is True
    assert understood.extracted_constraints == (
        "language:python",
        "path_prefix:src/api",
        "has_test_changes:true",
    )


def test_section_type_extraction_accepts_enum_values_and_names():
    understood, _ = understand_with_result(
        "Search only patches for ConnectionResetError",
        QueryUnderstandingBackendResult(
            retrieval_query="ConnectionResetError",
            section_types=("patch",),
        ),
    )

    assert understood.metadata_filter is not None
    assert understood.metadata_filter.section_types == (RetrievalSectionType.PATCH,)
    assert understood.retrieval_query == "ConnectionResetError"

    understood_from_name, _ = understand_with_result(
        "Search issues and review comments for connection leaks",
        QueryUnderstandingBackendResult(
            retrieval_query="connection leaks",
            section_types=("ISSUE", "REVIEW_COMMENT"),
        ),
    )
    assert understood_from_name.metadata_filter is not None
    assert understood_from_name.metadata_filter.section_types == (
        RetrievalSectionType.ISSUE,
        RetrievalSectionType.REVIEW_COMMENT,
    )


def test_explicit_label_extraction_is_supported():
    understood, _ = understand_with_result(
        "Find issues labeled security about authentication",
        QueryUnderstandingBackendResult(
            retrieval_query="authentication",
            section_types=("issue",),
            labels=("security",),
        ),
    )

    assert understood.metadata_filter is not None
    assert understood.metadata_filter.labels == ("security",)
    assert understood.retrieval_query == "authentication"


def test_ambiguous_label_can_remain_semantic_text_without_metadata_filter():
    understood, _ = understand_with_result(
        "Find security authentication bugs",
        QueryUnderstandingBackendResult(
            retrieval_query="security authentication bugs",
        ),
    )

    assert understood.metadata_filter is None
    assert understood.retrieval_query == "security authentication bugs"


def test_boolean_false_constraint_is_preserved():
    understood, _ = understand_with_result(
        "Find authentication changes that did not change tests",
        QueryUnderstandingBackendResult(
            retrieval_query="authentication changes",
            has_test_changes=False,
        ),
    )

    assert understood.metadata_filter is not None
    assert understood.metadata_filter.has_test_changes is False
    assert understood.extracted_constraints == ("has_test_changes:false",)


def test_multiple_field_and_multiple_value_or_constraints():
    understood, _ = understand_with_result(
        "Find Python or Go patches labeled bug under src/database",
        QueryUnderstandingBackendResult(
            retrieval_query="authentication",
            languages=("Python", "Go"),
            section_types=("patch",),
            labels=("bug",),
            path_prefixes=("src/database",),
        ),
    )

    assert understood.metadata_filter is not None
    assert understood.metadata_filter.languages == ("go", "python")
    assert understood.metadata_filter.section_types == (RetrievalSectionType.PATCH,)
    assert understood.metadata_filter.labels == ("bug",)
    assert understood.metadata_filter.path_prefixes == ("src/database",)


def test_issue_pr_sha_repository_extension_and_change_flags_are_supported():
    sha = "9a0b27473cfb40769d1b06ac827241fb89025def"
    understood, _ = understand_with_result(
        "Show retry cleanup from PR #487 in owner/project",
        QueryUnderstandingBackendResult(
            retrieval_query="retry cleanup",
            repositories=("owner/project",),
            issue_numbers=(421,),
            pull_request_numbers=(487,),
            commit_shas=(sha,),
            extensions=("py",),
            directories=("src/database",),
            has_documentation_changes=True,
            has_configuration_changes=True,
            has_dependency_changes=True,
        ),
    )

    assert understood.metadata_filter is not None
    assert understood.metadata_filter.repositories[0].owner == "owner"
    assert understood.metadata_filter.repositories[0].name == "project"
    assert understood.metadata_filter.issue_numbers == (421,)
    assert understood.metadata_filter.pull_request_numbers == (487,)
    assert understood.metadata_filter.commit_shas == (sha,)
    assert understood.metadata_filter.extensions == (".py",)
    assert understood.metadata_filter.directories == ("src/database",)
    assert understood.metadata_filter.has_documentation_changes is True
    assert understood.metadata_filter.has_configuration_changes is True
    assert understood.metadata_filter.has_dependency_changes is True


@pytest.mark.parametrize(
    ("result", "expected_field", "expected_value"),
    [
        (
            QueryUnderstandingBackendResult(
                retrieval_query="large changes",
                min_changed_lines=100,
            ),
            "min_changed_lines",
            100,
        ),
        (
            QueryUnderstandingBackendResult(
                retrieval_query="small patches",
                max_changed_lines=100,
            ),
            "max_changed_lines",
            100,
        ),
        (
            QueryUnderstandingBackendResult(
                retrieval_query="small patches",
                max_changed_lines=99,
            ),
            "max_changed_lines",
            99,
        ),
        (
            QueryUnderstandingBackendResult(
                retrieval_query="large patches",
                min_changed_lines=101,
            ),
            "min_changed_lines",
            101,
        ),
    ],
)
def test_numeric_range_constraints(
    result: QueryUnderstandingBackendResult,
    expected_field: str,
    expected_value: int,
):
    understood, _ = understand_with_result("query", result)

    assert understood.metadata_filter is not None
    assert getattr(understood.metadata_filter, expected_field) == expected_value


def test_blank_query_fails_without_calling_backend():
    backend = FakeQueryUnderstandingBackend(
        QueryUnderstandingBackendResult(retrieval_query="unused")
    )
    service = QueryUnderstandingService(backend)

    with pytest.raises(QueryUnderstandingValidationError, match="must not be blank"):
        service.understand("   ")

    assert backend.calls == []


def test_empty_backend_retrieval_query_falls_back_to_original_query():
    understood, _ = understand_with_result(
        "Python only",
        QueryUnderstandingBackendResult(
            retrieval_query="",
            languages=("Python",),
        ),
    )

    assert understood.retrieval_query == "Python only"
    assert understood.metadata_filter is not None
    assert understood.metadata_filter.languages == ("python",)


@pytest.mark.parametrize(
    "result",
    [
        QueryUnderstandingBackendResult(
            retrieval_query="invalid issue",
            issue_numbers=(0,),
        ),
        QueryUnderstandingBackendResult(
            retrieval_query="bad range",
            min_changed_lines=100,
            max_changed_lines=50,
        ),
        QueryUnderstandingBackendResult(
            retrieval_query="bad section",
            section_types=("CODE",),
        ),
        QueryUnderstandingBackendResult(
            retrieval_query="bad repository",
            repositories=("not-a-repository",),
        ),
    ],
)
def test_invalid_backend_metadata_fails_domain_validation(
    result: QueryUnderstandingBackendResult,
):
    backend = FakeQueryUnderstandingBackend(result)

    with pytest.raises(QueryUnderstandingValidationError, match="domain validation"):
        QueryUnderstandingService(backend).understand("query")


def test_duplicate_values_are_canonicalized_by_metadata_filter():
    understood, _ = understand_with_result(
        "Find Python authentication regressions",
        QueryUnderstandingBackendResult(
            retrieval_query="authentication regressions",
            languages=("Python", "python", "PYTHON"),
        ),
    )

    assert understood.metadata_filter is not None
    assert understood.metadata_filter.languages == ("python",)


def test_retrieval_query_preserves_technical_identifiers():
    understood, _ = understand_with_result(
        "Find Python ConnectionResetError in retry_worker()",
        QueryUnderstandingBackendResult(
            retrieval_query="ConnectionResetError in retry_worker()",
            languages=("Python",),
        ),
    )

    assert "ConnectionResetError" in understood.retrieval_query
    assert "retry_worker()" in understood.retrieval_query


def test_service_is_deterministic_with_deterministic_backend():
    result = QueryUnderstandingBackendResult(
        retrieval_query="authentication fixes",
        languages=("Python",),
        has_test_changes=True,
    )
    first, _ = understand_with_result("Find Python authentication fixes", result)
    second, _ = understand_with_result("Find Python authentication fixes", result)

    assert first == second
