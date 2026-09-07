from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from reporecall.github.models import GitHubRepository
from reporecall.models.records import GitHubUser


class GitHubTimelineEvidenceType(str, Enum):
    """GitHub timeline event types retained as relationship evidence."""

    CROSS_REFERENCED = "cross-referenced"
    REFERENCED = "referenced"
    CLOSED = "closed"


class GitHubTimelineRelationshipEvidence(BaseModel):
    """Normalized direct relationship evidence from a GitHub timeline event."""

    repository: GitHubRepository
    target_number: int = Field(gt=0)
    event_type: GitHubTimelineEvidenceType
    created_at: datetime
    actor: GitHubUser | None = None
    commit_repository: GitHubRepository | None = None
    commit_sha: str | None = Field(default=None, min_length=1)
    commit_url: str | None = Field(default=None, min_length=1)
    source_repository: GitHubRepository | None = None
    source_number: int | None = Field(default=None, gt=0)
    source_is_pull_request: bool | None = None

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("GitHub timeline evidence timestamps must be timezone-aware.")
        return value

    @field_validator("commit_sha")
    @classmethod
    def normalize_commit_sha(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Commit SHA must not be empty.")
        return normalized

    @model_validator(mode="after")
    def require_event_specific_fields(self) -> "GitHubTimelineRelationshipEvidence":
        if self.event_type is GitHubTimelineEvidenceType.CROSS_REFERENCED:
            if (
                self.source_repository is None
                or self.source_number is None
                or self.source_is_pull_request is None
            ):
                raise ValueError("Cross-reference evidence requires a complete source artifact.")
            return self

        if self.commit_repository is None or self.commit_sha is None or self.commit_url is None:
            raise ValueError("Commit timeline evidence requires repository, SHA, and URL fields.")
        return self


class GitHubCommitPullRequestAssociation(BaseModel):
    """Authoritative GitHub association between a commit and a pull request."""

    repository: GitHubRepository
    commit_sha: str = Field(min_length=1)
    pull_request_number: int = Field(gt=0)
    pull_request_url: str = Field(min_length=1)

    @field_validator("commit_sha")
    @classmethod
    def normalize_commit_sha(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Commit SHA must not be empty.")
        return normalized
