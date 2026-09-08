from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reporecall.github.models import GitHubRepository


class EventActorType(str, Enum):
    """Source namespaces for deterministic event actor identities."""

    GITHUB_USER = "github_user"
    GIT_AUTHOR = "git_author"


class EventActor(BaseModel):
    """An explicit actor identity preserved without cross-source reconciliation."""

    actor_type: EventActorType
    identifier: str = Field(min_length=1)
    name: str | None = None
    email: str | None = None

    model_config = ConfigDict(frozen=True)


class EventMetadata(BaseModel):
    """Immutable deterministic metadata derived from one engineering event."""

    event_id: str = Field(min_length=1)
    repository: GitHubRepository

    issue_numbers: tuple[int, ...] = ()
    pull_request_numbers: tuple[int, ...] = ()
    commit_shas: tuple[str, ...] = ()
    changed_paths: tuple[str, ...] = ()

    authors: tuple[EventActor, ...] = ()
    participants: tuple[EventActor, ...] = ()

    labels: tuple[str, ...] = ()
    milestones: tuple[str, ...] = ()

    languages: tuple[str, ...] = ()
    file_extensions: tuple[str, ...] = ()
    directories: tuple[str, ...] = ()

    added_lines: int = Field(default=0, ge=0)
    deleted_lines: int = Field(default=0, ge=0)
    changed_lines: int = Field(default=0, ge=0)

    has_tests: bool = False
    test_paths: tuple[str, ...] = ()

    has_documentation_changes: bool = False
    documentation_paths: tuple[str, ...] = ()

    has_configuration_changes: bool = False
    configuration_paths: tuple[str, ...] = ()

    has_dependency_changes: bool = False
    dependency_paths: tuple[str, ...] = ()

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_derived_values(self) -> "EventMetadata":
        if self.changed_lines != self.added_lines + self.deleted_lines:
            raise ValueError("changed_lines must equal added_lines plus deleted_lines.")

        path_flags = (
            (self.has_tests, self.test_paths, "has_tests"),
            (
                self.has_documentation_changes,
                self.documentation_paths,
                "has_documentation_changes",
            ),
            (
                self.has_configuration_changes,
                self.configuration_paths,
                "has_configuration_changes",
            ),
            (
                self.has_dependency_changes,
                self.dependency_paths,
                "has_dependency_changes",
            ),
        )
        for flag, paths, field_name in path_flags:
            if flag is not bool(paths):
                raise ValueError(f"{field_name} must match whether its path collection is empty.")
        return self
