from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class QueryUnderstandingActor(BaseModel):
    """Provider-neutral actor identity before domain validation."""

    actor_type: str = Field(min_length=1)
    identifier: str = Field(min_length=1)
    name: str | None = None
    email: str | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


class QueryUnderstandingBackendResult(BaseModel):
    """Structured provider output before RepoRecall domain conversion."""

    retrieval_query: str

    repositories: tuple[str, ...] = ()
    section_types: tuple[str, ...] = ()

    issue_numbers: tuple[int, ...] = ()
    pull_request_numbers: tuple[int, ...] = ()
    commit_shas: tuple[str, ...] = ()

    labels: tuple[str, ...] = ()
    milestones: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    extensions: tuple[str, ...] = ()

    directories: tuple[str, ...] = ()
    path_prefixes: tuple[str, ...] = ()
    actors: tuple[QueryUnderstandingActor, ...] = ()

    has_test_changes: bool | None = None
    has_documentation_changes: bool | None = None
    has_configuration_changes: bool | None = None
    has_dependency_changes: bool | None = None

    min_added_lines: int | None = None
    max_added_lines: int | None = None
    min_deleted_lines: int | None = None
    max_deleted_lines: int | None = None
    min_changed_lines: int | None = None
    max_changed_lines: int | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


class QueryUnderstandingBackend(Protocol):
    """Backend capable of structured search-intent interpretation."""

    @property
    def model_name(self) -> str:
        """Return the concrete interpretation model identity."""

    def understand(self, query: str) -> QueryUnderstandingBackendResult:
        """Interpret a nonblank natural-language query."""
