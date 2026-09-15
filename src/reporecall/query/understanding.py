from collections.abc import Iterable

from pydantic import ValidationError

from reporecall.github.models import GitHubRepository
from reporecall.models import EventActor, EventActorType, MetadataFilter
from reporecall.models.query_understanding import UnderstoodQuery
from reporecall.models.retrieval_documents import RetrievalSectionType
from reporecall.query.backend import (
    QueryUnderstandingBackend,
    QueryUnderstandingBackendResult,
)
from reporecall.query.exceptions import QueryUnderstandingValidationError


class QueryUnderstandingService:
    """Validate backend query interpretation into RepoRecall domain models."""

    def __init__(self, backend: QueryUnderstandingBackend) -> None:
        self._backend = backend

    def understand(self, query: str) -> UnderstoodQuery:
        """Interpret search text without invoking retrieval or generation."""

        if not query.strip():
            raise QueryUnderstandingValidationError("Query must not be blank.")

        backend_result = self._backend.understand(query)
        metadata_filter = self._build_metadata_filter(backend_result)
        retrieval_query = backend_result.retrieval_query.strip() or query.strip()

        return UnderstoodQuery(
            original_query=query,
            retrieval_query=retrieval_query,
            metadata_filter=metadata_filter,
            extracted_constraints=_extracted_constraints(metadata_filter),
            model_name=self._backend.model_name,
        )

    def _build_metadata_filter(
        self,
        backend_result: QueryUnderstandingBackendResult,
    ) -> MetadataFilter | None:
        try:
            metadata_filter = MetadataFilter(
                repositories=tuple(
                    GitHubRepository.parse(value)
                    for value in backend_result.repositories
                ),
                section_types=tuple(
                    _parse_section_type(value)
                    for value in backend_result.section_types
                ),
                issue_numbers=backend_result.issue_numbers,
                pull_request_numbers=backend_result.pull_request_numbers,
                commit_shas=backend_result.commit_shas,
                labels=backend_result.labels,
                milestones=backend_result.milestones,
                languages=backend_result.languages,
                extensions=backend_result.extensions,
                directories=backend_result.directories,
                path_prefixes=backend_result.path_prefixes,
                actors=tuple(
                    EventActor(
                        actor_type=EventActorType(actor.actor_type),
                        identifier=actor.identifier,
                        name=actor.name,
                        email=actor.email,
                    )
                    for actor in backend_result.actors
                ),
                has_test_changes=backend_result.has_test_changes,
                has_documentation_changes=(
                    backend_result.has_documentation_changes
                ),
                has_configuration_changes=(
                    backend_result.has_configuration_changes
                ),
                has_dependency_changes=backend_result.has_dependency_changes,
                min_added_lines=backend_result.min_added_lines,
                max_added_lines=backend_result.max_added_lines,
                min_deleted_lines=backend_result.min_deleted_lines,
                max_deleted_lines=backend_result.max_deleted_lines,
                min_changed_lines=backend_result.min_changed_lines,
                max_changed_lines=backend_result.max_changed_lines,
            )
        except (ValueError, ValidationError) as exc:
            raise QueryUnderstandingValidationError(
                "Query-understanding output failed domain validation."
            ) from exc

        if metadata_filter.is_empty:
            return None
        return metadata_filter


def _parse_section_type(value: str) -> RetrievalSectionType:
    candidate = value.strip()
    if not candidate:
        raise ValueError("Section type must not be blank.")
    try:
        return RetrievalSectionType(candidate)
    except ValueError:
        try:
            return RetrievalSectionType[candidate.upper()]
        except KeyError as exc:
            raise ValueError(f"Unsupported retrieval section type: {value!r}.") from exc


def _extracted_constraints(metadata_filter: MetadataFilter | None) -> tuple[str, ...]:
    if metadata_filter is None:
        return ()

    constraints: list[str] = []
    constraints.extend(
        f"repository:{repository.owner}/{repository.name}"
        for repository in metadata_filter.repositories
    )
    constraints.extend(
        f"section_type:{section_type.value}"
        for section_type in metadata_filter.section_types
    )
    constraints.extend(_labeled_values("issue", metadata_filter.issue_numbers))
    constraints.extend(
        _labeled_values("pull_request", metadata_filter.pull_request_numbers)
    )
    constraints.extend(_labeled_values("commit_sha", metadata_filter.commit_shas))
    constraints.extend(_labeled_values("label", metadata_filter.labels))
    constraints.extend(_labeled_values("milestone", metadata_filter.milestones))
    constraints.extend(_labeled_values("language", metadata_filter.languages))
    constraints.extend(_labeled_values("extension", metadata_filter.extensions))
    constraints.extend(_labeled_values("directory", metadata_filter.directories))
    constraints.extend(_labeled_values("path_prefix", metadata_filter.path_prefixes))
    constraints.extend(
        f"actor:{actor.actor_type.value}:{actor.identifier}"
        for actor in metadata_filter.actors
    )

    boolean_constraints = (
        ("has_test_changes", metadata_filter.has_test_changes),
        (
            "has_documentation_changes",
            metadata_filter.has_documentation_changes,
        ),
        (
            "has_configuration_changes",
            metadata_filter.has_configuration_changes,
        ),
        ("has_dependency_changes", metadata_filter.has_dependency_changes),
    )
    for boolean_label, boolean_value in boolean_constraints:
        if boolean_value is not None:
            constraints.append(f"{boolean_label}:{str(boolean_value).lower()}")

    numeric_constraints = (
        ("min_added_lines", metadata_filter.min_added_lines),
        ("max_added_lines", metadata_filter.max_added_lines),
        ("min_deleted_lines", metadata_filter.min_deleted_lines),
        ("max_deleted_lines", metadata_filter.max_deleted_lines),
        ("min_changed_lines", metadata_filter.min_changed_lines),
        ("max_changed_lines", metadata_filter.max_changed_lines),
    )
    for numeric_label, numeric_value in numeric_constraints:
        if numeric_value is not None:
            constraints.append(f"{numeric_label}:{numeric_value}")

    return tuple(constraints)


def _labeled_values(label: str, values: Iterable[object]) -> list[str]:
    return [f"{label}:{value}" for value in values]
