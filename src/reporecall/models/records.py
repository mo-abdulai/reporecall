from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from reporecall.github.models import GitHubRepository


class FileChangeType(str, Enum):
    """Raw Git file-level change categories supported by RepoRecall."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


class ChangedFile(BaseModel):
    """A file changed by a Git commit."""

    path: str
    change_type: FileChangeType
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    patch: str | None
    old_path: str | None = None


class GitCommit(BaseModel):
    """Normalized raw Git commit data."""

    sha: str
    message: str
    author_name: str
    author_email: str | None
    authored_at: datetime
    committed_at: datetime
    parent_shas: list[str]
    changed_files: list[ChangedFile]


class IssueState(str, Enum):
    """GitHub issue states supported by RepoRecall."""

    OPEN = "open"
    CLOSED = "closed"


class GitHubUser(BaseModel):
    """Minimal GitHub user identity retained for source attribution."""

    login: str
    id: int | None = None
    html_url: str | None = None


class GitHubIssueLabel(BaseModel):
    """Structured label metadata attached to a GitHub issue."""

    name: str
    color: str | None = None
    description: str | None = None


class GitHubMilestone(BaseModel):
    """Minimal GitHub milestone identity retained for deterministic metadata."""

    number: int = Field(gt=0)
    title: str = Field(min_length=1)
    html_url: str | None = None


class GitHubIssue(BaseModel):
    """Normalized GitHub issue data from a repository issues endpoint."""

    repository: GitHubRepository
    number: int
    title: str
    body: str | None
    state: IssueState
    author: GitHubUser | None
    labels: list[GitHubIssueLabel]
    milestone: GitHubMilestone | None = None
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None
    html_url: str
    comments_count: int = Field(ge=0)
    locked: bool

    @field_validator("created_at", "updated_at", "closed_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("GitHub issue timestamps must be timezone-aware.")
        return value


class GitHubIssueComment(BaseModel):
    """Normalized issue-style GitHub comment data."""

    repository: GitHubRepository
    id: int
    issue_number: int
    author: GitHubUser | None
    body: str | None
    created_at: datetime
    updated_at: datetime
    html_url: str
    author_association: str | None = None

    @field_validator("created_at", "updated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("GitHub issue comment timestamps must be timezone-aware.")
        return value


class PullRequestState(str, Enum):
    """GitHub pull request states supported as listing filters."""

    OPEN = "open"
    CLOSED = "closed"


class ReviewState(str, Enum):
    """GitHub pull request review states normalized by RepoRecall."""

    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    COMMENTED = "commented"
    DISMISSED = "dismissed"
    PENDING = "pending"

    @classmethod
    def _missing_(cls, value: object) -> "ReviewState | None":
        if isinstance(value, str):
            normalized = value.lower()
            for member in cls:
                if member.value == normalized:
                    return member
        return None


class GitHubPullRequestFileStatus(str, Enum):
    """GitHub pull request file status values normalized by RepoRecall."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    COPIED = "copied"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


class GitHubBranchReference(BaseModel):
    """Minimal source or target branch metadata for a pull request."""

    repository: GitHubRepository | None
    ref: str
    sha: str
    label: str | None = None


class GitHubPullRequest(BaseModel):
    """Normalized GitHub pull request metadata."""

    repository: GitHubRepository
    number: int
    title: str
    body: str | None
    state: PullRequestState
    author: GitHubUser | None
    labels: list[GitHubIssueLabel]
    milestone: GitHubMilestone | None = None
    draft: bool
    locked: bool
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None
    merged_at: datetime | None
    html_url: str
    head: GitHubBranchReference
    base: GitHubBranchReference
    merge_commit_sha: str | None
    commits_count: int | None = Field(default=None, ge=0)
    changed_files_count: int | None = Field(default=None, ge=0)
    additions: int | None = Field(default=None, ge=0)
    deletions: int | None = Field(default=None, ge=0)
    comments_count: int | None = Field(default=None, ge=0)
    review_comments_count: int | None = Field(default=None, ge=0)
    maintainer_can_modify: bool | None = None

    @property
    def is_merged(self) -> bool:
        return self.merged_at is not None

    @field_validator("created_at", "updated_at", "closed_at", "merged_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("GitHub pull request timestamps must be timezone-aware.")
        return value


class GitHubPullRequestReview(BaseModel):
    """Normalized GitHub pull request review metadata."""

    repository: GitHubRepository
    id: int
    pull_request_number: int
    author: GitHubUser | None
    body: str | None
    state: ReviewState
    submitted_at: datetime | None
    commit_sha: str | None
    html_url: str | None

    @field_validator("submitted_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("GitHub pull request review timestamps must be timezone-aware.")
        return value


class GitHubPullRequestReviewComment(BaseModel):
    """A line-level comment attached to a GitHub pull request review."""

    repository: GitHubRepository
    id: int
    pull_request_number: int
    review_id: int | None
    author: GitHubUser | None
    body: str | None
    created_at: datetime
    updated_at: datetime
    html_url: str
    commit_sha: str | None
    original_commit_sha: str | None
    path: str
    line: int | None = None
    original_line: int | None = None
    side: str | None = None
    start_line: int | None = None
    start_side: str | None = None
    author_association: str | None = None

    @field_validator("created_at", "updated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("GitHub pull request review comment timestamps must be timezone-aware.")
        return value


class GitHubPullRequestFile(BaseModel):
    """A file changed by a GitHub pull request."""

    filename: str
    status: GitHubPullRequestFileStatus
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    changes: int = Field(ge=0)
    patch: str | None = None
    previous_filename: str | None = None
    raw_url: str | None = None
    blob_url: str | None = None


class GitHubCommitReference(BaseModel):
    """Lightweight commit reference returned by a GitHub pull request."""

    sha: str
    html_url: str | None
    message: str
    author_name: str | None
    author_email: str | None
    authored_at: datetime | None

    @field_validator("authored_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("GitHub commit reference timestamps must be timezone-aware.")
        return value
