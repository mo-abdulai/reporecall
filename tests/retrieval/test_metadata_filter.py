from collections.abc import Sequence

import pytest

from reporecall.github import GitHubRepository
from reporecall.models import (
    EventActor,
    EventActorType,
    EventMetadata,
    MetadataFilter,
    RetrievalChunk,
    RetrievalSectionType,
)
from reporecall.retrieval import MetadataFilterMatcher, filter_chunk_ids


def test_language_filter_uses_or_within_the_field():
    chunks = [
        _chunk("a", languages=("Python",)),
        _chunk("b", languages=("TypeScript",)),
        _chunk("c", languages=("Python", "YAML")),
        _chunk("d", languages=("Go",)),
    ]

    assert filter_chunk_ids(chunks, MetadataFilter(languages=("Python",))) == (
        "a",
        "c",
    )
    assert filter_chunk_ids(
        chunks,
        MetadataFilter(languages=("Python", "Go")),
    ) == ("a", "c", "d")


def test_different_fields_use_and_semantics():
    chunks = [
        _chunk("a", languages=("Python",), labels=("bug",)),
        _chunk("b", languages=("Python",), labels=("feature",)),
        _chunk("c", languages=("TypeScript",), labels=("bug",)),
    ]

    candidates = filter_chunk_ids(
        chunks,
        MetadataFilter(languages=("Python",), labels=("bug",)),
    )

    assert candidates == ("a",)


def test_repository_and_section_type_use_structured_chunk_fields():
    first_repository = GitHubRepository(owner="owner", name="first")
    second_repository = GitHubRepository(owner="owner", name="second")
    chunks = [
        _chunk(
            "issue",
            repository=first_repository,
            section_type=RetrievalSectionType.ISSUE,
        ),
        _chunk(
            "patch",
            repository=first_repository,
            section_type=RetrievalSectionType.PATCH,
        ),
        _chunk(
            "other-repo",
            repository=second_repository,
            section_type=RetrievalSectionType.PATCH,
        ),
    ]

    candidates = filter_chunk_ids(
        chunks,
        MetadataFilter(
            repositories=(first_repository,),
            section_types=(RetrievalSectionType.PATCH,),
        ),
    )

    assert candidates == ("patch",)


def test_issue_pr_and_commit_filters_ignore_textual_occurrences():
    structural = _chunk(
        "structural",
        content="No identifiers are written here.",
        issue_numbers=(10,),
        pull_request_numbers=(20,),
        commit_shas=("abc123",),
    )
    textual = _chunk(
        "textual",
        content="See Issue #10, PR #20, and commit abc123.",
    )

    assert filter_chunk_ids(
        [structural, textual],
        MetadataFilter(issue_numbers=(10,)),
    ) == ("structural",)
    assert filter_chunk_ids(
        [structural, textual],
        MetadataFilter(pull_request_numbers=(20,)),
    ) == ("structural",)
    assert filter_chunk_ids(
        [structural, textual],
        MetadataFilter(commit_shas=("abc123",)),
    ) == ("structural",)


def test_label_matching_is_case_insensitive_without_semantic_expansion():
    chunks = [
        _chunk("bug", labels=("Bug",)),
        _chunk("regression", labels=("regression",)),
    ]

    assert filter_chunk_ids(chunks, MetadataFilter(labels=("BUG",))) == ("bug",)


def test_milestone_matching_is_exact():
    chunks = [
        _chunk("exact", milestones=("Release 1",)),
        _chunk("different-case", milestones=("release 1",)),
    ]

    assert filter_chunk_ids(
        chunks,
        MetadataFilter(milestones=("Release 1",)),
    ) == ("exact",)


def test_extension_matching_accepts_optional_leading_dot_and_ignores_case():
    chunks = [
        _chunk("python", extensions=(".py",)),
        _chunk("typescript", extensions=(".ts",)),
    ]

    assert filter_chunk_ids(chunks, MetadataFilter(extensions=("PY",))) == (
        "python",
    )


def test_directory_matching_is_exact_against_metadata_directories():
    chunk = _chunk(
        "database",
        directories=("src", "src/database", "src/database/models"),
    )

    matcher = MetadataFilterMatcher()

    assert matcher.matches(chunk, MetadataFilter(directories=("src/database",)))
    assert not matcher.matches(chunk, MetadataFilter(directories=("database",)))


def test_path_prefix_matching_is_case_sensitive_and_component_aware():
    matching = _chunk(
        "matching",
        changed_paths=("src/database/session.py", "src/api/routes.py"),
    )
    sibling = _chunk("sibling", changed_paths=("src/database_old.py",))
    different_case = _chunk(
        "different-case",
        changed_paths=("Src/database/session.py",),
    )

    candidates = filter_chunk_ids(
        [matching, sibling, different_case],
        MetadataFilter(path_prefixes=("src/database/",)),
    )

    assert candidates == ("matching",)


def test_actor_filtering_uses_exact_namespaced_identity():
    github_actor = EventActor(
        actor_type=EventActorType.GITHUB_USER,
        identifier="alice",
    )
    git_actor = EventActor(
        actor_type=EventActorType.GIT_AUTHOR,
        identifier="alice",
    )
    chunks = [
        _chunk("github", authors=(github_actor,)),
        _chunk("git", participants=(git_actor,)),
    ]

    assert filter_chunk_ids(
        chunks,
        MetadataFilter(actors=(github_actor,)),
    ) == ("github",)


@pytest.mark.parametrize(
    ("filter_field", "metadata_path_field", "matching_path"),
    [
        ("has_test_changes", "test_paths", "tests/test_api.py"),
        ("has_documentation_changes", "documentation_paths", "docs/api.md"),
        ("has_configuration_changes", "configuration_paths", "pyproject.toml"),
        ("has_dependency_changes", "dependency_paths", "uv.lock"),
    ],
)
def test_file_category_filters_support_true_and_false(
    filter_field: str,
    metadata_path_field: str,
    matching_path: str,
):
    changed = _chunk("changed", **{metadata_path_field: (matching_path,)})
    unchanged = _chunk("unchanged")

    assert filter_chunk_ids(
        [changed, unchanged],
        MetadataFilter.model_validate({filter_field: True}),
    ) == ("changed",)
    assert filter_chunk_ids(
        [changed, unchanged],
        MetadataFilter.model_validate({filter_field: False}),
    ) == ("unchanged",)


def test_numeric_ranges_are_inclusive_and_apply_across_statistics():
    chunk = _chunk("changes", added_lines=50, deleted_lines=25)
    matcher = MetadataFilterMatcher()

    assert matcher.matches(
        chunk,
        MetadataFilter(
            min_added_lines=50,
            max_added_lines=50,
            min_deleted_lines=20,
            max_deleted_lines=30,
            min_changed_lines=75,
            max_changed_lines=75,
        ),
    )
    assert not matcher.matches(chunk, MetadataFilter(min_changed_lines=76))
    assert not matcher.matches(chunk, MetadataFilter(max_changed_lines=74))


def test_empty_filter_matches_all_and_candidate_ids_are_unique_and_stable():
    first = _chunk("first")
    second = _chunk("second")

    candidates = filter_chunk_ids(
        [second, first, second],
        MetadataFilter(),
    )

    assert candidates == ("second", "first")


def test_matching_does_not_mutate_filter_or_chunk():
    chunk = _chunk("chunk", languages=("Python",), labels=("Bug",))
    metadata_filter = MetadataFilter(languages=("python",), labels=("bug",))
    original_chunk = chunk.model_dump()
    original_filter = metadata_filter.model_dump()

    assert MetadataFilterMatcher().matches(chunk, metadata_filter)
    assert chunk.model_dump() == original_chunk
    assert metadata_filter.model_dump() == original_filter


def _chunk(
    chunk_id: str,
    *,
    repository: GitHubRepository | None = None,
    section_type: RetrievalSectionType = RetrievalSectionType.ISSUE,
    content: str = "source content",
    issue_numbers: Sequence[int] = (),
    pull_request_numbers: Sequence[int] = (),
    commit_shas: Sequence[str] = (),
    labels: Sequence[str] = (),
    milestones: Sequence[str] = (),
    languages: Sequence[str] = (),
    extensions: Sequence[str] = (),
    directories: Sequence[str] = (),
    changed_paths: Sequence[str] = (),
    authors: Sequence[EventActor] = (),
    participants: Sequence[EventActor] = (),
    test_paths: Sequence[str] = (),
    documentation_paths: Sequence[str] = (),
    configuration_paths: Sequence[str] = (),
    dependency_paths: Sequence[str] = (),
    added_lines: int = 0,
    deleted_lines: int = 0,
) -> RetrievalChunk:
    active_repository = repository or GitHubRepository(owner="owner", name="repo")
    event_id = f"event-{chunk_id}"
    metadata = EventMetadata(
        event_id=event_id,
        repository=active_repository,
        issue_numbers=tuple(issue_numbers),
        pull_request_numbers=tuple(pull_request_numbers),
        commit_shas=tuple(commit_shas),
        changed_paths=tuple(changed_paths),
        authors=tuple(authors),
        participants=tuple(participants),
        labels=tuple(labels),
        milestones=tuple(milestones),
        languages=tuple(languages),
        file_extensions=tuple(extensions),
        directories=tuple(directories),
        added_lines=added_lines,
        deleted_lines=deleted_lines,
        changed_lines=added_lines + deleted_lines,
        has_tests=bool(test_paths),
        test_paths=tuple(test_paths),
        has_documentation_changes=bool(documentation_paths),
        documentation_paths=tuple(documentation_paths),
        has_configuration_changes=bool(configuration_paths),
        configuration_paths=tuple(configuration_paths),
        has_dependency_changes=bool(dependency_paths),
        dependency_paths=tuple(dependency_paths),
    )
    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        event_id=event_id,
        repository=active_repository,
        section_id=f"section-{chunk_id}",
        section_type=section_type,
        chunk_index=0,
        content=content,
        text=f"Repository content\n{content}",
        metadata=metadata,
    )
