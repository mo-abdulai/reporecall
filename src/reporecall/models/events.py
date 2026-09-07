from pydantic import BaseModel, Field, model_validator

from reporecall.github.models import GitHubRepository
from reporecall.models.records import (
    GitCommit,
    GitHubCommitReference,
    GitHubIssue,
    GitHubIssueComment,
    GitHubPullRequest,
    GitHubPullRequestFile,
    GitHubPullRequestReview,
    GitHubPullRequestReviewComment,
)
from reporecall.models.relationships import (
    ArtifactReference,
    ArtifactType,
    EngineeringRelationship,
)

_CORE_ANCHOR_TYPES = {
    ArtifactType.PULL_REQUEST,
    ArtifactType.ISSUE,
    ArtifactType.LOCAL_GIT_COMMIT,
    ArtifactType.GITHUB_COMMIT_REFERENCE,
}


class EngineeringEvent(BaseModel):
    """A repository-local structural grouping of connected engineering records."""

    event_id: str = Field(min_length=1)
    repository: GitHubRepository
    anchor: ArtifactReference
    issues: list[GitHubIssue] = Field(default_factory=list)
    issue_comments: list[GitHubIssueComment] = Field(default_factory=list)
    pull_requests: list[GitHubPullRequest] = Field(default_factory=list)
    pull_request_comments: list[GitHubIssueComment] = Field(default_factory=list)
    pull_request_reviews: list[GitHubPullRequestReview] = Field(default_factory=list)
    pull_request_review_comments: list[GitHubPullRequestReviewComment] = Field(
        default_factory=list
    )
    pull_request_files: list[GitHubPullRequestFile] = Field(default_factory=list)
    github_commit_references: list[GitHubCommitReference] = Field(default_factory=list)
    local_commits: list[GitCommit] = Field(default_factory=list)
    relationships: list[EngineeringRelationship] = Field(default_factory=list)
    contextual_relationships: list[EngineeringRelationship] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_repository_identity(self) -> "EngineeringEvent":
        if self.anchor.artifact_type not in _CORE_ANCHOR_TYPES:
            raise ValueError("Engineering event anchors must be core artifacts.")
        if self.anchor.repository != self.repository:
            raise ValueError("Engineering event anchor must belong to the event repository.")

        record_repositories = [
            *(record.repository for record in self.issues),
            *(record.repository for record in self.issue_comments),
            *(record.repository for record in self.pull_requests),
            *(record.repository for record in self.pull_request_comments),
            *(record.repository for record in self.pull_request_reviews),
            *(record.repository for record in self.pull_request_review_comments),
        ]
        if any(repository != self.repository for repository in record_repositories):
            raise ValueError("Engineering event records must belong to one repository.")
        if any(
            relationship.source.repository != self.repository
            or relationship.target.repository != self.repository
            for relationship in self.relationships
        ):
            raise ValueError("Internal event relationships must remain repository-local.")
        return self

    @property
    def issue_numbers(self) -> list[int]:
        """Return unique issue numbers in deterministic order."""

        return sorted({issue.number for issue in self.issues})

    @property
    def pull_request_numbers(self) -> list[int]:
        """Return unique pull request numbers in deterministic order."""

        return sorted({pull_request.number for pull_request in self.pull_requests})

    @property
    def commit_shas(self) -> list[str]:
        """Return unique GitHub and local commit SHAs in deterministic order."""

        return sorted(
            {
                *(commit.sha for commit in self.github_commit_references),
                *(commit.sha for commit in self.local_commits),
            }
        )

    @property
    def changed_paths(self) -> list[str]:
        """Return unique PR and local changed paths in deterministic order."""

        return sorted(
            {
                *(file.filename for file in self.pull_request_files),
                *(
                    file.path
                    for commit in self.local_commits
                    for file in commit.changed_files
                ),
            }
        )
