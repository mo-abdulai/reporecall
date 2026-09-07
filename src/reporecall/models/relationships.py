from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from reporecall.github.models import GitHubRepository


class ArtifactType(str, Enum):
    """Artifact categories that can participate in deterministic relationships."""

    ISSUE = "issue"
    ISSUE_COMMENT = "issue_comment"
    PULL_REQUEST = "pull_request"
    PULL_REQUEST_FILE = "pull_request_file"
    PULL_REQUEST_COMMENT = "pull_request_comment"
    PULL_REQUEST_REVIEW = "pull_request_review"
    PULL_REQUEST_REVIEW_COMMENT = "pull_request_review_comment"
    GITHUB_COMMIT_REFERENCE = "github_commit_reference"
    LOCAL_GIT_COMMIT = "local_git_commit"
    LOCAL_CHANGED_FILE = "local_changed_file"


class ArtifactReference(BaseModel):
    """Stable lightweight identity for an engineering-history artifact."""

    artifact_type: ArtifactType
    repository: GitHubRepository
    identifier: str = Field(min_length=1)

    model_config = ConfigDict(frozen=True)


class RelationshipType(str, Enum):
    """Deterministic relationship semantics between two artifacts."""

    ISSUE_HAS_COMMENT = "issue_has_comment"
    PULL_REQUEST_HAS_COMMENT = "pull_request_has_comment"
    PULL_REQUEST_HAS_REVIEW = "pull_request_has_review"
    REVIEW_HAS_COMMENT = "review_has_comment"
    PULL_REQUEST_HAS_REVIEW_COMMENT = "pull_request_has_review_comment"
    PULL_REQUEST_HAS_COMMIT = "pull_request_has_commit"
    GITHUB_COMMIT_MATCHES_LOCAL_COMMIT = "github_commit_matches_local_commit"
    PULL_REQUEST_CHANGES_FILE = "pull_request_changes_file"
    LOCAL_COMMIT_CHANGES_FILE = "local_commit_changes_file"
    PULL_REQUEST_REFERENCES_ISSUE = "pull_request_references_issue"
    PULL_REQUEST_CLOSES_ISSUE = "pull_request_closes_issue"
    COMMIT_REFERENCES_ISSUE = "commit_references_issue"
    COMMIT_CLOSES_ISSUE = "commit_closes_issue"


class RelationshipEvidenceType(str, Enum):
    """Evidence categories used to explain deterministic relationships."""

    PARENT_IDENTIFIER = "parent_identifier"
    PR_COMMIT_LIST = "pr_commit_list"
    EXACT_SHA_MATCH = "exact_sha_match"
    PR_FILE_LIST = "pr_file_list"
    LOCAL_COMMIT_FILE_LIST = "local_commit_file_list"
    CLOSING_KEYWORD = "closing_keyword"
    TEXT_REFERENCE = "text_reference"
    GITHUB_TIMELINE_CROSS_REFERENCE = "github_timeline_cross_reference"
    GITHUB_TIMELINE_COMMIT_REFERENCE = "github_timeline_commit_reference"
    GITHUB_TIMELINE_CLOSED_EVENT = "github_timeline_closed_event"
    GITHUB_COMMIT_ASSOCIATED_PULL_REQUEST = "github_commit_associated_pull_request"


class EngineeringRelationship(BaseModel):
    """A deterministic relationship between two RepoRecall artifact references."""

    source: ArtifactReference
    target: ArtifactReference
    relationship_type: RelationshipType
    evidence_type: RelationshipEvidenceType
    evidence: str | None = None
    source_field: str | None = None

    model_config = ConfigDict(frozen=True)

    @property
    def identity_key(self) -> tuple[ArtifactReference, RelationshipType, ArtifactReference, RelationshipEvidenceType]:
        """Return the deterministic identity used for deduplication."""

        return (self.source, self.relationship_type, self.target, self.evidence_type)
