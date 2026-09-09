import pytest
from pydantic import ValidationError

from reporecall.github import GitHubRepository
from reporecall.models import (
    EventActor,
    EventActorType,
    MetadataFilter,
    RetrievalSectionType,
)


def test_filter_is_immutable_serializable_and_normalizes_values():
    actor = EventActor(
        actor_type=EventActorType.GITHUB_USER,
        identifier="alice",
    )
    repository = GitHubRepository(owner="owner", name="repo")
    metadata_filter = MetadataFilter(
        repositories=(repository, repository),
        section_types=(RetrievalSectionType.PATCH, RetrievalSectionType.ISSUE),
        issue_numbers=(20, 10, 10),
        pull_request_numbers=(30, 30),
        commit_shas=(" abc123 ", "abc123"),
        labels=("Bug", " bug "),
        milestones=(" Release 1 ", "Release 1"),
        languages=("Python", "PYTHON", "Go"),
        extensions=("py", ".PY", ".ts"),
        directories=("src\\database/", "src/database"),
        path_prefixes=("./src/database/", "src/database"),
        actors=(actor, actor),
    )

    assert metadata_filter.repositories == (repository,)
    assert metadata_filter.section_types == (
        RetrievalSectionType.ISSUE,
        RetrievalSectionType.PATCH,
    )
    assert metadata_filter.issue_numbers == (10, 20)
    assert metadata_filter.pull_request_numbers == (30,)
    assert metadata_filter.commit_shas == ("abc123",)
    assert metadata_filter.labels == ("bug",)
    assert metadata_filter.milestones == ("Release 1",)
    assert metadata_filter.languages == ("go", "python")
    assert metadata_filter.extensions == (".py", ".ts")
    assert metadata_filter.directories == ("src/database",)
    assert metadata_filter.path_prefixes == ("src/database",)
    assert metadata_filter.actors == (actor,)
    assert MetadataFilter.model_validate(metadata_filter.model_dump()) == metadata_filter

    with pytest.raises(ValidationError, match="frozen"):
        metadata_filter.labels = ("feature",)


def test_empty_filter_and_false_boolean_have_distinct_meanings():
    assert MetadataFilter().is_empty is True
    assert MetadataFilter(has_test_changes=False).is_empty is False
    assert MetadataFilter(min_changed_lines=0).is_empty is False


@pytest.mark.parametrize("field", ["issue_numbers", "pull_request_numbers"])
@pytest.mark.parametrize("value", [0, -1])
def test_identity_numbers_must_be_positive(field: str, value: int):
    with pytest.raises(ValidationError, match="must be positive"):
        MetadataFilter.model_validate({field: (value,)})


@pytest.mark.parametrize(
    "field",
    ["commit_shas", "labels", "milestones", "languages", "extensions"],
)
def test_text_filter_values_must_not_be_blank(field: str):
    with pytest.raises(ValidationError, match="must not be blank"):
        MetadataFilter.model_validate({field: ("   ",)})


@pytest.mark.parametrize("field", ["directories", "path_prefixes"])
@pytest.mark.parametrize("value", ["", " ", ".", "/src", "../src", "src/../tests"])
def test_path_filter_values_must_be_repository_relative(field: str, value: str):
    with pytest.raises(ValidationError, match="must not be blank|repository-relative"):
        MetadataFilter.model_validate({field: (value,)})


@pytest.mark.parametrize(
    "field",
    [
        "min_added_lines",
        "max_added_lines",
        "min_deleted_lines",
        "max_deleted_lines",
        "min_changed_lines",
        "max_changed_lines",
    ],
)
def test_numeric_bounds_must_be_nonnegative(field: str):
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        MetadataFilter.model_validate({field: -1})


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [
        ("min_added_lines", "max_added_lines"),
        ("min_deleted_lines", "max_deleted_lines"),
        ("min_changed_lines", "max_changed_lines"),
    ],
)
def test_numeric_minimum_cannot_exceed_maximum(minimum: str, maximum: str):
    with pytest.raises(ValidationError, match="Minimum .* cannot exceed"):
        MetadataFilter.model_validate({minimum: 2, maximum: 1})


def test_actor_normalization_preserves_source_namespaces():
    github_actor = EventActor(
        actor_type=EventActorType.GITHUB_USER,
        identifier="alice",
    )
    git_actor = EventActor(
        actor_type=EventActorType.GIT_AUTHOR,
        identifier="alice",
    )

    metadata_filter = MetadataFilter(actors=(github_actor, git_actor, github_actor))

    assert metadata_filter.actors == (git_actor, github_actor)
