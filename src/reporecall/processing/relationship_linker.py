from pydantic import BaseModel, Field

from reporecall.github import GitHubRepository
from reporecall.models import (
    ArtifactReference,
    ArtifactType,
    ChangedFile,
    EngineeringRelationship,
    GitCommit,
    GitHubCommitReference,
    GitHubIssue,
    GitHubIssueComment,
    GitHubPullRequest,
    GitHubPullRequestFile,
    GitHubPullRequestReview,
    GitHubPullRequestReviewComment,
    RelationshipEvidenceType,
    RelationshipType,
)
from reporecall.processing.reference_parser import IssueReferenceKind, ReferenceParser


class RelationshipInput(BaseModel):
    """Normalized records supplied to the deterministic relationship linker."""

    repository: GitHubRepository
    issues: list[GitHubIssue] = Field(default_factory=list)
    issue_comments: list[GitHubIssueComment] = Field(default_factory=list)
    pull_requests: list[GitHubPullRequest] = Field(default_factory=list)
    pull_request_comments: list[GitHubIssueComment] = Field(default_factory=list)
    pull_request_reviews: list[GitHubPullRequestReview] = Field(default_factory=list)
    pull_request_review_comments: list[GitHubPullRequestReviewComment] = Field(default_factory=list)
    pull_request_commits: dict[int, list[GitHubCommitReference]] = Field(default_factory=dict)
    pull_request_files: dict[int, list[GitHubPullRequestFile]] = Field(default_factory=dict)
    local_commits: list[GitCommit] = Field(default_factory=list)


class RelationshipLinker:
    """Create deterministic relationships from normalized RepoRecall records."""

    def __init__(self, reference_parser: ReferenceParser | None = None) -> None:
        self.reference_parser = reference_parser or ReferenceParser()

    def link(self, records: RelationshipInput) -> list[EngineeringRelationship]:
        """Return stable, deduplicated relationships with concise evidence."""

        builder = _RelationshipBuilder()
        issues = {(issue.repository, issue.number): issue for issue in records.issues}
        pull_requests = {
            (pull_request.repository, pull_request.number): pull_request for pull_request in records.pull_requests
        }
        reviews = {(review.repository, review.id): review for review in records.pull_request_reviews}
        local_commits = {commit.sha: commit for commit in records.local_commits}

        self._link_issue_comments(records.issue_comments, issues, pull_requests, builder)
        self._link_pull_request_comments(records.pull_request_comments, pull_requests, builder)
        self._link_pull_request_reviews(records.pull_request_reviews, pull_requests, builder)
        self._link_review_comments(records.pull_request_review_comments, pull_requests, reviews, builder)
        self._link_pull_request_commits(records, pull_requests, local_commits, builder)
        self._link_pull_request_files(records, pull_requests, builder)
        self._link_local_commit_files(records, builder)
        self._link_pull_request_issue_references(records.pull_requests, issues, builder)
        self._link_commit_issue_references(records, issues, builder)

        return builder.relationships()

    def _link_issue_comments(
        self,
        comments: list[GitHubIssueComment],
        issues: dict[tuple[GitHubRepository, int], GitHubIssue],
        pull_requests: dict[tuple[GitHubRepository, int], GitHubPullRequest],
        builder: "_RelationshipBuilder",
    ) -> None:
        for comment in comments:
            key = (comment.repository, comment.issue_number)
            if key in pull_requests:
                continue
            issue = issues.get(key)
            if issue is None:
                continue
            builder.add(
                source=_issue_ref(issue),
                target=_issue_comment_ref(comment),
                relationship_type=RelationshipType.ISSUE_HAS_COMMENT,
                evidence_type=RelationshipEvidenceType.PARENT_IDENTIFIER,
                evidence=f"comment parent issue #{comment.issue_number}",
            )

    def _link_pull_request_comments(
        self,
        comments: list[GitHubIssueComment],
        pull_requests: dict[tuple[GitHubRepository, int], GitHubPullRequest],
        builder: "_RelationshipBuilder",
    ) -> None:
        for comment in comments:
            pull_request = pull_requests.get((comment.repository, comment.issue_number))
            if pull_request is None:
                continue
            builder.add(
                source=_pull_request_ref(pull_request),
                target=_pull_request_comment_ref(comment),
                relationship_type=RelationshipType.PULL_REQUEST_HAS_COMMENT,
                evidence_type=RelationshipEvidenceType.PARENT_IDENTIFIER,
                evidence=f"comment parent pull request #{comment.issue_number}",
            )

    def _link_pull_request_reviews(
        self,
        reviews: list[GitHubPullRequestReview],
        pull_requests: dict[tuple[GitHubRepository, int], GitHubPullRequest],
        builder: "_RelationshipBuilder",
    ) -> None:
        for review in reviews:
            pull_request = pull_requests.get((review.repository, review.pull_request_number))
            if pull_request is None:
                continue
            builder.add(
                source=_pull_request_ref(pull_request),
                target=_review_ref(review),
                relationship_type=RelationshipType.PULL_REQUEST_HAS_REVIEW,
                evidence_type=RelationshipEvidenceType.PARENT_IDENTIFIER,
                evidence=f"review parent pull request #{review.pull_request_number}",
            )

    def _link_review_comments(
        self,
        comments: list[GitHubPullRequestReviewComment],
        pull_requests: dict[tuple[GitHubRepository, int], GitHubPullRequest],
        reviews: dict[tuple[GitHubRepository, int], GitHubPullRequestReview],
        builder: "_RelationshipBuilder",
    ) -> None:
        for comment in comments:
            if comment.review_id is not None:
                review = reviews.get((comment.repository, comment.review_id))
                if review is not None:
                    builder.add(
                        source=_review_ref(review),
                        target=_review_comment_ref(comment),
                        relationship_type=RelationshipType.REVIEW_HAS_COMMENT,
                        evidence_type=RelationshipEvidenceType.PARENT_IDENTIFIER,
                        evidence=f"review comment parent review {comment.review_id}",
                    )

            pull_request = pull_requests.get((comment.repository, comment.pull_request_number))
            if pull_request is not None:
                builder.add(
                    source=_pull_request_ref(pull_request),
                    target=_review_comment_ref(comment),
                    relationship_type=RelationshipType.PULL_REQUEST_HAS_REVIEW_COMMENT,
                    evidence_type=RelationshipEvidenceType.PARENT_IDENTIFIER,
                    evidence=f"review comment parent pull request #{comment.pull_request_number}",
                )

    def _link_pull_request_commits(
        self,
        records: RelationshipInput,
        pull_requests: dict[tuple[GitHubRepository, int], GitHubPullRequest],
        local_commits: dict[str, GitCommit],
        builder: "_RelationshipBuilder",
    ) -> None:
        for pull_request_number, commit_refs in records.pull_request_commits.items():
            pull_request = pull_requests.get((records.repository, pull_request_number))
            if pull_request is None:
                continue

            for commit_ref in commit_refs:
                builder.add(
                    source=_pull_request_ref(pull_request),
                    target=_github_commit_ref(records.repository, commit_ref),
                    relationship_type=RelationshipType.PULL_REQUEST_HAS_COMMIT,
                    evidence_type=RelationshipEvidenceType.PR_COMMIT_LIST,
                    evidence=f"GitHub PR commit list included {commit_ref.sha}",
                )

                local_commit = local_commits.get(commit_ref.sha)
                if local_commit is not None:
                    builder.add(
                        source=_github_commit_ref(records.repository, commit_ref),
                        target=_local_commit_ref(records.repository, local_commit),
                        relationship_type=RelationshipType.GITHUB_COMMIT_MATCHES_LOCAL_COMMIT,
                        evidence_type=RelationshipEvidenceType.EXACT_SHA_MATCH,
                        evidence=f"GitHub commit SHA exactly matched local commit SHA {commit_ref.sha}",
                    )

    def _link_pull_request_files(
        self,
        records: RelationshipInput,
        pull_requests: dict[tuple[GitHubRepository, int], GitHubPullRequest],
        builder: "_RelationshipBuilder",
    ) -> None:
        for pull_request_number, files in records.pull_request_files.items():
            pull_request = pull_requests.get((records.repository, pull_request_number))
            if pull_request is None:
                continue

            for file in files:
                builder.add(
                    source=_pull_request_ref(pull_request),
                    target=_pull_request_file_ref(records.repository, pull_request_number, file),
                    relationship_type=RelationshipType.PULL_REQUEST_CHANGES_FILE,
                    evidence_type=RelationshipEvidenceType.PR_FILE_LIST,
                    evidence=f"GitHub PR file list included {file.filename}",
                )

    def _link_local_commit_files(
        self,
        records: RelationshipInput,
        builder: "_RelationshipBuilder",
    ) -> None:
        for commit in records.local_commits:
            for file in commit.changed_files:
                builder.add(
                    source=_local_commit_ref(records.repository, commit),
                    target=_local_changed_file_ref(records.repository, commit, file),
                    relationship_type=RelationshipType.LOCAL_COMMIT_CHANGES_FILE,
                    evidence_type=RelationshipEvidenceType.LOCAL_COMMIT_FILE_LIST,
                    evidence=f"local commit file list included {file.path}",
                )

    def _link_pull_request_issue_references(
        self,
        pull_requests: list[GitHubPullRequest],
        issues: dict[tuple[GitHubRepository, int], GitHubIssue],
        builder: "_RelationshipBuilder",
    ) -> None:
        for pull_request in pull_requests:
            for reference in self.reference_parser.parse_issue_references(
                pull_request.body,
                default_repository=pull_request.repository,
            ):
                issue = issues.get((reference.repository, reference.issue_number))
                if issue is None:
                    continue
                relationship_type = (
                    RelationshipType.PULL_REQUEST_CLOSES_ISSUE
                    if reference.kind is IssueReferenceKind.CLOSES
                    else RelationshipType.PULL_REQUEST_REFERENCES_ISSUE
                )
                evidence_type = (
                    RelationshipEvidenceType.CLOSING_KEYWORD
                    if reference.kind is IssueReferenceKind.CLOSES
                    else RelationshipEvidenceType.TEXT_REFERENCE
                )
                builder.add(
                    source=_pull_request_ref(pull_request),
                    target=_issue_ref(issue),
                    relationship_type=relationship_type,
                    evidence_type=evidence_type,
                    evidence=reference.evidence,
                    source_field="body",
                )

    def _link_commit_issue_references(
        self,
        records: RelationshipInput,
        issues: dict[tuple[GitHubRepository, int], GitHubIssue],
        builder: "_RelationshipBuilder",
    ) -> None:
        for commit in records.local_commits:
            for reference in self.reference_parser.parse_issue_references(
                commit.message,
                default_repository=records.repository,
            ):
                issue = issues.get((reference.repository, reference.issue_number))
                if issue is None:
                    continue
                relationship_type = (
                    RelationshipType.COMMIT_CLOSES_ISSUE
                    if reference.kind is IssueReferenceKind.CLOSES
                    else RelationshipType.COMMIT_REFERENCES_ISSUE
                )
                evidence_type = (
                    RelationshipEvidenceType.CLOSING_KEYWORD
                    if reference.kind is IssueReferenceKind.CLOSES
                    else RelationshipEvidenceType.TEXT_REFERENCE
                )
                builder.add(
                    source=_local_commit_ref(records.repository, commit),
                    target=_issue_ref(issue),
                    relationship_type=relationship_type,
                    evidence_type=evidence_type,
                    evidence=reference.evidence,
                    source_field="message",
                )


class _RelationshipBuilder:
    def __init__(self) -> None:
        self._relationships_by_key: dict[
            tuple[ArtifactReference, RelationshipType, ArtifactReference, RelationshipEvidenceType],
            EngineeringRelationship,
        ] = {}

    def add(
        self,
        *,
        source: ArtifactReference,
        target: ArtifactReference,
        relationship_type: RelationshipType,
        evidence_type: RelationshipEvidenceType,
        evidence: str | None = None,
        source_field: str | None = None,
    ) -> None:
        relationship = EngineeringRelationship(
            source=source,
            target=target,
            relationship_type=relationship_type,
            evidence_type=evidence_type,
            evidence=evidence,
            source_field=source_field,
        )
        self._relationships_by_key.setdefault(relationship.identity_key, relationship)

    def relationships(self) -> list[EngineeringRelationship]:
        return sorted(
            self._relationships_by_key.values(),
            key=lambda relationship: (
                relationship.source.repository.owner,
                relationship.source.repository.name,
                relationship.source.artifact_type.value,
                relationship.source.identifier,
                relationship.relationship_type.value,
                relationship.target.repository.owner,
                relationship.target.repository.name,
                relationship.target.artifact_type.value,
                relationship.target.identifier,
                relationship.evidence_type.value,
                relationship.evidence or "",
            ),
        )


def _issue_ref(issue: GitHubIssue) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.ISSUE,
        repository=issue.repository,
        identifier=str(issue.number),
    )


def _issue_comment_ref(comment: GitHubIssueComment) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.ISSUE_COMMENT,
        repository=comment.repository,
        identifier=f"issue-comment:{comment.id}",
    )


def _pull_request_ref(pull_request: GitHubPullRequest) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.PULL_REQUEST,
        repository=pull_request.repository,
        identifier=str(pull_request.number),
    )


def _pull_request_comment_ref(comment: GitHubIssueComment) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.PULL_REQUEST_COMMENT,
        repository=comment.repository,
        identifier=f"pr-comment:{comment.id}",
    )


def _review_ref(review: GitHubPullRequestReview) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.PULL_REQUEST_REVIEW,
        repository=review.repository,
        identifier=f"review:{review.id}",
    )


def _review_comment_ref(comment: GitHubPullRequestReviewComment) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.PULL_REQUEST_REVIEW_COMMENT,
        repository=comment.repository,
        identifier=f"review-comment:{comment.id}",
    )


def _github_commit_ref(repository: GitHubRepository, commit: GitHubCommitReference) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.GITHUB_COMMIT_REFERENCE,
        repository=repository,
        identifier=commit.sha,
    )


def _local_commit_ref(repository: GitHubRepository, commit: GitCommit) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.LOCAL_GIT_COMMIT,
        repository=repository,
        identifier=commit.sha,
    )


def _pull_request_file_ref(
    repository: GitHubRepository,
    pull_request_number: int,
    file: GitHubPullRequestFile,
) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.PULL_REQUEST_FILE,
        repository=repository,
        identifier=f"pr:{pull_request_number}:file:{file.filename}",
    )


def _local_changed_file_ref(
    repository: GitHubRepository,
    commit: GitCommit,
    file: ChangedFile,
) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.LOCAL_CHANGED_FILE,
        repository=repository,
        identifier=f"{commit.sha}:{file.path}",
    )
