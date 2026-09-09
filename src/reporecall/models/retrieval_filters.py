import posixpath

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reporecall.github.models import GitHubRepository
from reporecall.models.event_metadata import EventActor
from reporecall.models.retrieval_documents import RetrievalSectionType


class MetadataFilter(BaseModel):
    """Structured retrieval constraints with AND-across-fields semantics.

    Multiple values within one field use OR semantics. Empty fields do not
    constrain matching.
    """

    repositories: tuple[GitHubRepository, ...] = ()
    section_types: tuple[RetrievalSectionType, ...] = ()

    issue_numbers: tuple[int, ...] = ()
    pull_request_numbers: tuple[int, ...] = ()
    commit_shas: tuple[str, ...] = ()

    labels: tuple[str, ...] = ()
    milestones: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    extensions: tuple[str, ...] = ()

    directories: tuple[str, ...] = ()
    path_prefixes: tuple[str, ...] = ()
    actors: tuple[EventActor, ...] = ()

    has_test_changes: bool | None = None
    has_documentation_changes: bool | None = None
    has_configuration_changes: bool | None = None
    has_dependency_changes: bool | None = None

    min_added_lines: int | None = Field(default=None, ge=0)
    max_added_lines: int | None = Field(default=None, ge=0)
    min_deleted_lines: int | None = Field(default=None, ge=0)
    max_deleted_lines: int | None = Field(default=None, ge=0)
    min_changed_lines: int | None = Field(default=None, ge=0)
    max_changed_lines: int | None = Field(default=None, ge=0)

    model_config = ConfigDict(frozen=True)

    @field_validator("repositories")
    @classmethod
    def normalize_repositories(
        cls,
        values: tuple[GitHubRepository, ...],
    ) -> tuple[GitHubRepository, ...]:
        by_identity = {(item.owner, item.name): item for item in values}
        return tuple(by_identity[key] for key in sorted(by_identity))

    @field_validator("section_types")
    @classmethod
    def normalize_section_types(
        cls,
        values: tuple[RetrievalSectionType, ...],
    ) -> tuple[RetrievalSectionType, ...]:
        return tuple(sorted(set(values), key=lambda item: item.value))

    @field_validator("issue_numbers", "pull_request_numbers")
    @classmethod
    def normalize_positive_numbers(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(value <= 0 for value in values):
            raise ValueError("Issue and pull request filter numbers must be positive.")
        return tuple(sorted(set(values)))

    @field_validator("commit_shas")
    @classmethod
    def normalize_commit_shas(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _normalize_strings(values, field_name="commit SHA")

    @field_validator("labels")
    @classmethod
    def normalize_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _normalize_strings(values, field_name="label", casefold=True)

    @field_validator("milestones")
    @classmethod
    def normalize_milestones(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _normalize_strings(values, field_name="milestone")

    @field_validator("languages")
    @classmethod
    def normalize_languages(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _normalize_strings(values, field_name="language", casefold=True)

    @field_validator("extensions")
    @classmethod
    def normalize_extensions(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized: set[str] = set()
        for value in values:
            extension = value.strip().casefold()
            if not extension or extension == ".":
                raise ValueError("Extension filter values must not be blank.")
            normalized.add(extension if extension.startswith(".") else f".{extension}")
        return tuple(sorted(normalized))

    @field_validator("directories", "path_prefixes")
    @classmethod
    def normalize_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted({_normalize_filter_path(value) for value in values}))

    @field_validator("actors")
    @classmethod
    def normalize_actors(cls, values: tuple[EventActor, ...]) -> tuple[EventActor, ...]:
        by_identity = {
            (actor.actor_type.value, actor.identifier, actor.name, actor.email): actor
            for actor in values
        }
        keys = sorted(
            by_identity,
            key=lambda item: tuple(value or "" for value in item),
        )
        return tuple(by_identity[key] for key in keys)

    @model_validator(mode="after")
    def validate_ranges(self) -> "MetadataFilter":
        ranges = (
            (self.min_added_lines, self.max_added_lines, "added lines"),
            (self.min_deleted_lines, self.max_deleted_lines, "deleted lines"),
            (self.min_changed_lines, self.max_changed_lines, "changed lines"),
        )
        for minimum, maximum, label in ranges:
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError(f"Minimum {label} cannot exceed maximum {label}.")
        return self

    @property
    def is_empty(self) -> bool:
        """Return whether this filter imposes no metadata constraints."""

        collection_fields = (
            self.repositories,
            self.section_types,
            self.issue_numbers,
            self.pull_request_numbers,
            self.commit_shas,
            self.labels,
            self.milestones,
            self.languages,
            self.extensions,
            self.directories,
            self.path_prefixes,
            self.actors,
        )
        optional_fields = (
            self.has_test_changes,
            self.has_documentation_changes,
            self.has_configuration_changes,
            self.has_dependency_changes,
            self.min_added_lines,
            self.max_added_lines,
            self.min_deleted_lines,
            self.max_deleted_lines,
            self.min_changed_lines,
            self.max_changed_lines,
        )
        return not any(collection_fields) and all(
            value is None for value in optional_fields
        )


def _normalize_strings(
    values: tuple[str, ...],
    *,
    field_name: str,
    casefold: bool = False,
) -> tuple[str, ...]:
    normalized: set[str] = set()
    for value in values:
        candidate = value.strip()
        if not candidate:
            raise ValueError(f"{field_name.capitalize()} filter values must not be blank.")
        normalized.add(candidate.casefold() if casefold else candidate)
    return tuple(sorted(normalized, key=lambda item: (item.casefold(), item)))


def _normalize_filter_path(value: str) -> str:
    candidate = value.strip().replace("\\", "/")
    if not candidate:
        raise ValueError("Directory and path-prefix filters must not be blank.")
    if candidate.startswith("/") or ".." in candidate.split("/"):
        raise ValueError("Directory and path-prefix filters must be repository-relative.")

    normalized = posixpath.normpath(candidate)
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized in {"", "."}:
        raise ValueError("Directory and path-prefix filters must not be blank.")
    return normalized
