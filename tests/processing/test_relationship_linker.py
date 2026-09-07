from datetime import UTC, datetime

from reporecall.github import GitHubRepository
from reporecall.models import (
    ArtifactType,
    ChangedFile,
    FileChangeType,
    GitCommit,
    GitHubBranchReference,
    GitHubCommitReference,
    GitHubIssue,
    GitHubIssueComment,
    GitHubPullRequest,
    GitHubPullRequestFile,
    GitHubPullRequestFileStatus,
    GitHubPullRequestReview,
    GitHubPullRequestReviewComment,
    IssueState,
    PullRequestState,
    RelationshipEvidenceType,
    RelationshipType,
    ReviewState,
)
from reporecall.processing import RelationshipInput, RelationshipLinker


def test_parent_identifier_relationships():
    repository = _repository()
    issue = _issue(10)
    issue_comment = _issue_comment(100, issue_number=10)
    pull_request = _pull_request(20)
    pr_comment = _issue_comment(101, issue_number=20)
    review = _review(200, pull_request_number=20)
    review_comment = _review_comment(300, pull_request_number=20, review_id=200)

    relationships = RelationshipLinker().link(
        RelationshipInput(
            repository=repository,
            issues=[issue],
            issue_comments=[issue_comment],
            pull_requests=[pull_request],
            pull_request_comments=[pr_comment],
            pull_request_reviews=[review],
            pull_request_review_comments=[review_comment],
        )
    )

    assert _has_relationship(
        relationships,
        source_type=ArtifactType.ISSUE,
        source_id="10",
        relationship_type=RelationshipType.ISSUE_HAS_COMMENT,
        target_type=ArtifactType.ISSUE_COMMENT,
        target_id="issue-comment:100",
        evidence_type=RelationshipEvidenceType.PARENT_IDENTIFIER,
    )
    assert _has_relationship(
        relationships,
        source_type=ArtifactType.PULL_REQUEST,
        source_id="20",
        relationship_type=RelationshipType.PULL_REQUEST_HAS_COMMENT,
        target_type=ArtifactType.PULL_REQUEST_COMMENT,
        target_id="pr-comment:101",
        evidence_type=RelationshipEvidenceType.PARENT_IDENTIFIER,
    )
    assert _has_relationship(
        relationships,
        source_type=ArtifactType.PULL_REQUEST,
        source_id="20",
        relationship_type=RelationshipType.PULL_REQUEST_HAS_REVIEW,
        target_type=ArtifactType.PULL_REQUEST_REVIEW,
        target_id="review:200",
        evidence_type=RelationshipEvidenceType.PARENT_IDENTIFIER,
    )
    assert _has_relationship(
        relationships,
        source_type=ArtifactType.PULL_REQUEST_REVIEW,
        source_id="review:200",
        relationship_type=RelationshipType.REVIEW_HAS_COMMENT,
        target_type=ArtifactType.PULL_REQUEST_REVIEW_COMMENT,
        target_id="review-comment:300",
        evidence_type=RelationshipEvidenceType.PARENT_IDENTIFIER,
    )
    assert _has_relationship(
        relationships,
        source_type=ArtifactType.PULL_REQUEST,
        source_id="20",
        relationship_type=RelationshipType.PULL_REQUEST_HAS_REVIEW_COMMENT,
        target_type=ArtifactType.PULL_REQUEST_REVIEW_COMMENT,
        target_id="review-comment:300",
        evidence_type=RelationshipEvidenceType.PARENT_IDENTIFIER,
    )


def test_missing_parent_records_do_not_create_relationships():
    relationships = RelationshipLinker().link(
        RelationshipInput(
            repository=_repository(),
            issue_comments=[_issue_comment(100, issue_number=10)],
            pull_request_reviews=[_review(200, pull_request_number=20)],
            pull_request_review_comments=[_review_comment(300, pull_request_number=20, review_id=201)],
        )
    )

    assert relationships == []


def test_pull_request_commit_relationships_and_exact_local_commit_matching():
    matching_ref = _commit_reference("abc123", message="Fix session cleanup")
    different_ref = _commit_reference("def456", message="Same message")
    local_match = _local_commit("abc123", message="Different local message")
    same_message_different_sha = _local_commit("zzz999", message="Same message")

    relationships = RelationshipLinker().link(
        RelationshipInput(
            repository=_repository(),
            pull_requests=[_pull_request(100)],
            pull_request_commits={100: [matching_ref, different_ref]},
            local_commits=[local_match, same_message_different_sha],
        )
    )

    assert _has_relationship(
        relationships,
        source_type=ArtifactType.PULL_REQUEST,
        source_id="100",
        relationship_type=RelationshipType.PULL_REQUEST_HAS_COMMIT,
        target_type=ArtifactType.GITHUB_COMMIT_REFERENCE,
        target_id="abc123",
        evidence_type=RelationshipEvidenceType.PR_COMMIT_LIST,
    )
    assert _has_relationship(
        relationships,
        source_type=ArtifactType.PULL_REQUEST,
        source_id="100",
        relationship_type=RelationshipType.PULL_REQUEST_HAS_COMMIT,
        target_type=ArtifactType.GITHUB_COMMIT_REFERENCE,
        target_id="def456",
        evidence_type=RelationshipEvidenceType.PR_COMMIT_LIST,
    )
    assert _has_relationship(
        relationships,
        source_type=ArtifactType.GITHUB_COMMIT_REFERENCE,
        source_id="abc123",
        relationship_type=RelationshipType.GITHUB_COMMIT_MATCHES_LOCAL_COMMIT,
        target_type=ArtifactType.LOCAL_GIT_COMMIT,
        target_id="abc123",
        evidence_type=RelationshipEvidenceType.EXACT_SHA_MATCH,
    )
    assert not _has_relationship(
        relationships,
        source_type=ArtifactType.GITHUB_COMMIT_REFERENCE,
        source_id="def456",
        relationship_type=RelationshipType.GITHUB_COMMIT_MATCHES_LOCAL_COMMIT,
        target_type=ArtifactType.LOCAL_GIT_COMMIT,
        target_id="zzz999",
        evidence_type=RelationshipEvidenceType.EXACT_SHA_MATCH,
    )


def test_pull_request_file_relationship_identifiers_do_not_collide_for_same_filename():
    file = _pull_request_file("src/app.py")
    relationships = RelationshipLinker().link(
        RelationshipInput(
            repository=_repository(),
            pull_requests=[_pull_request(100), _pull_request(101)],
            pull_request_files={100: [file], 101: [file]},
        )
    )

    assert _has_relationship(
        relationships,
        source_type=ArtifactType.PULL_REQUEST,
        source_id="100",
        relationship_type=RelationshipType.PULL_REQUEST_CHANGES_FILE,
        target_type=ArtifactType.PULL_REQUEST_FILE,
        target_id="pr:100:file:src/app.py",
        evidence_type=RelationshipEvidenceType.PR_FILE_LIST,
    )
    assert _has_relationship(
        relationships,
        source_type=ArtifactType.PULL_REQUEST,
        source_id="101",
        relationship_type=RelationshipType.PULL_REQUEST_CHANGES_FILE,
        target_type=ArtifactType.PULL_REQUEST_FILE,
        target_id="pr:101:file:src/app.py",
        evidence_type=RelationshipEvidenceType.PR_FILE_LIST,
    )


def test_local_changed_file_relationship_identifiers_do_not_collide_for_same_filename():
    first = _local_commit("abc123", changed_files=[_changed_file("src/app.py")])
    second = _local_commit("def456", changed_files=[_changed_file("src/app.py")])

    relationships = RelationshipLinker().link(
        RelationshipInput(repository=_repository(), local_commits=[first, second])
    )

    assert _has_relationship(
        relationships,
        source_type=ArtifactType.LOCAL_GIT_COMMIT,
        source_id="abc123",
        relationship_type=RelationshipType.LOCAL_COMMIT_CHANGES_FILE,
        target_type=ArtifactType.LOCAL_CHANGED_FILE,
        target_id="abc123:src/app.py",
        evidence_type=RelationshipEvidenceType.LOCAL_COMMIT_FILE_LIST,
    )
    assert _has_relationship(
        relationships,
        source_type=ArtifactType.LOCAL_GIT_COMMIT,
        source_id="def456",
        relationship_type=RelationshipType.LOCAL_COMMIT_CHANGES_FILE,
        target_type=ArtifactType.LOCAL_CHANGED_FILE,
        target_id="def456:src/app.py",
        evidence_type=RelationshipEvidenceType.LOCAL_COMMIT_FILE_LIST,
    )


def test_pull_request_body_closing_plain_multiple_and_duplicate_issue_references():
    relationships = RelationshipLinker().link(
        RelationshipInput(
            repository=_repository(),
            issues=[_issue(10), _issue(20), _issue(30)],
            pull_requests=[
                _pull_request(
                    100,
                    body="Fixes #10. Also fixes #10. closes #20. Related to #30.",
                )
            ],
        )
    )

    assert _relationship_count(relationships, RelationshipType.PULL_REQUEST_CLOSES_ISSUE) == 2
    assert _relationship_count(relationships, RelationshipType.PULL_REQUEST_REFERENCES_ISSUE) == 1
    assert _has_relationship(
        relationships,
        source_type=ArtifactType.PULL_REQUEST,
        source_id="100",
        relationship_type=RelationshipType.PULL_REQUEST_CLOSES_ISSUE,
        target_type=ArtifactType.ISSUE,
        target_id="10",
        evidence_type=RelationshipEvidenceType.CLOSING_KEYWORD,
    )
    assert _has_relationship(
        relationships,
        source_type=ArtifactType.PULL_REQUEST,
        source_id="100",
        relationship_type=RelationshipType.PULL_REQUEST_REFERENCES_ISSUE,
        target_type=ArtifactType.ISSUE,
        target_id="30",
        evidence_type=RelationshipEvidenceType.TEXT_REFERENCE,
    )


def test_missing_and_cross_repository_issue_references_are_not_fabricated():
    relationships = RelationshipLinker().link(
        RelationshipInput(
            repository=_repository(),
            issues=[_issue(123)],
            pull_requests=[
                _pull_request(100, body="Fixes #999. Fixes other/project#123."),
            ],
        )
    )

    assert relationships == []


def test_commit_message_issue_references_keep_closing_and_plain_meanings_distinct():
    relationships = RelationshipLinker().link(
        RelationshipInput(
            repository=_repository(),
            issues=[_issue(50), _issue(51)],
            local_commits=[
                _local_commit("abc123", message="Fixes #50"),
                _local_commit("def456", message="See #51"),
            ],
        )
    )

    assert _has_relationship(
        relationships,
        source_type=ArtifactType.LOCAL_GIT_COMMIT,
        source_id="abc123",
        relationship_type=RelationshipType.COMMIT_CLOSES_ISSUE,
        target_type=ArtifactType.ISSUE,
        target_id="50",
        evidence_type=RelationshipEvidenceType.CLOSING_KEYWORD,
    )
    assert _has_relationship(
        relationships,
        source_type=ArtifactType.LOCAL_GIT_COMMIT,
        source_id="def456",
        relationship_type=RelationshipType.COMMIT_REFERENCES_ISSUE,
        target_type=ArtifactType.ISSUE,
        target_id="51",
        evidence_type=RelationshipEvidenceType.TEXT_REFERENCE,
    )


def test_linker_output_is_deterministic():
    records = RelationshipInput(
        repository=_repository(),
        issues=[_issue(10)],
        issue_comments=[_issue_comment(100, issue_number=10)],
        pull_requests=[_pull_request(20, body="Fixes #10")],
        pull_request_commits={20: [_commit_reference("abc123")]},
        local_commits=[_local_commit("abc123")],
    )
    linker = RelationshipLinker()

    first = linker.link(records)
    second = linker.link(records)

    assert first == second


def _has_relationship(
    relationships,
    *,
    source_type: ArtifactType,
    source_id: str,
    relationship_type: RelationshipType,
    target_type: ArtifactType,
    target_id: str,
    evidence_type: RelationshipEvidenceType,
) -> bool:
    return any(
        relationship.source.artifact_type is source_type
        and relationship.source.identifier == source_id
        and relationship.relationship_type is relationship_type
        and relationship.target.artifact_type is target_type
        and relationship.target.identifier == target_id
        and relationship.evidence_type is evidence_type
        for relationship in relationships
    )


def _relationship_count(relationships, relationship_type: RelationshipType) -> int:
    return sum(1 for relationship in relationships if relationship.relationship_type is relationship_type)


def _repository() -> GitHubRepository:
    return GitHubRepository(owner="owner", name="repo")


def _issue(number: int, *, repository: GitHubRepository | None = None) -> GitHubIssue:
    repository = repository or _repository()
    return GitHubIssue(
        repository=repository,
        number=number,
        title=f"Issue {number}",
        body=None,
        state=IssueState.OPEN,
        author=None,
        labels=[],
        created_at=_dt(1),
        updated_at=_dt(2),
        closed_at=None,
        html_url=f"https://github.com/{repository.owner}/{repository.name}/issues/{number}",
        comments_count=0,
        locked=False,
    )


def _issue_comment(comment_id: int, *, issue_number: int) -> GitHubIssueComment:
    return GitHubIssueComment(
        repository=_repository(),
        id=comment_id,
        issue_number=issue_number,
        author=None,
        body="Comment",
        created_at=_dt(1),
        updated_at=_dt(2),
        html_url=f"https://github.com/owner/repo/issues/{issue_number}#issuecomment-{comment_id}",
    )


def _pull_request(number: int, *, body: str | None = None) -> GitHubPullRequest:
    repository = _repository()
    return GitHubPullRequest(
        repository=repository,
        number=number,
        title=f"PR {number}",
        body=body,
        state=PullRequestState.OPEN,
        author=None,
        labels=[],
        draft=False,
        locked=False,
        created_at=_dt(1),
        updated_at=_dt(2),
        closed_at=None,
        merged_at=None,
        html_url=f"https://github.com/owner/repo/pull/{number}",
        head=GitHubBranchReference(
            repository=repository,
            ref="feature",
            sha="headsha",
            label="owner:feature",
        ),
        base=GitHubBranchReference(
            repository=repository,
            ref="main",
            sha="basesha",
            label="owner:main",
        ),
        merge_commit_sha=None,
    )


def _review(review_id: int, *, pull_request_number: int) -> GitHubPullRequestReview:
    return GitHubPullRequestReview(
        repository=_repository(),
        id=review_id,
        pull_request_number=pull_request_number,
        author=None,
        body="Looks good",
        state=ReviewState.APPROVED,
        submitted_at=_dt(3),
        commit_sha="abc123",
        html_url=f"https://github.com/owner/repo/pull/{pull_request_number}#pullrequestreview-{review_id}",
    )


def _review_comment(
    comment_id: int,
    *,
    pull_request_number: int,
    review_id: int | None,
) -> GitHubPullRequestReviewComment:
    return GitHubPullRequestReviewComment(
        repository=_repository(),
        id=comment_id,
        pull_request_number=pull_request_number,
        review_id=review_id,
        author=None,
        body="Review comment",
        created_at=_dt(4),
        updated_at=_dt(5),
        html_url=f"https://github.com/owner/repo/pull/{pull_request_number}#discussion_r{comment_id}",
        commit_sha="abc123",
        original_commit_sha="def456",
        path="src/app.py",
    )


def _commit_reference(sha: str, *, message: str = "Commit message") -> GitHubCommitReference:
    return GitHubCommitReference(
        sha=sha,
        html_url=f"https://github.com/owner/repo/commit/{sha}",
        message=message,
        author_name=None,
        author_email=None,
        authored_at=None,
    )


def _local_commit(
    sha: str,
    *,
    message: str = "Commit message",
    changed_files: list[ChangedFile] | None = None,
) -> GitCommit:
    return GitCommit(
        sha=sha,
        message=message,
        author_name="Repo Tester",
        author_email="tester@example.com",
        authored_at=_dt(1),
        committed_at=_dt(1),
        parent_shas=[],
        changed_files=changed_files if changed_files is not None else [],
    )


def _pull_request_file(filename: str) -> GitHubPullRequestFile:
    return GitHubPullRequestFile(
        filename=filename,
        status=GitHubPullRequestFileStatus.MODIFIED,
        additions=1,
        deletions=1,
        changes=2,
    )


def _changed_file(path: str) -> ChangedFile:
    return ChangedFile(
        path=path,
        old_path=None,
        change_type=FileChangeType.MODIFIED,
        additions=1,
        deletions=1,
        patch=None,
    )


def _dt(day: int) -> datetime:
    return datetime(2026, 1, day, 12, 0, tzinfo=UTC)
