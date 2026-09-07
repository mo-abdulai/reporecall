from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from reporecall.github import GitHubRepository
from reporecall.models import (
    GitHubBranchReference,
    GitHubCommitReference,
    GitHubIssueLabel,
    GitHubPullRequest,
    GitHubPullRequestFile,
    GitHubPullRequestFileStatus,
    GitHubPullRequestReview,
    GitHubPullRequestReviewComment,
    GitHubUser,
    PullRequestState,
    ReviewState,
)


def test_open_pull_request_with_metadata_and_fork_head_repository():
    pull_request = _pull_request(
        state=PullRequestState.OPEN,
        head_repository=GitHubRepository(owner="contributor", name="repo"),
        labels=[
            GitHubIssueLabel(name="bug", color="d73a4a", description="Something is broken."),
            GitHubIssueLabel(name="database", color="0075ca", description=None),
        ],
    )

    assert pull_request.state is PullRequestState.OPEN
    assert pull_request.author is not None
    assert pull_request.author.login == "octocat"
    assert [label.name for label in pull_request.labels] == ["bug", "database"]
    assert pull_request.head.repository == GitHubRepository(owner="contributor", name="repo")
    assert pull_request.base.repository == GitHubRepository(owner="owner", name="repo")
    assert pull_request.created_at.tzinfo is not None
    assert pull_request.is_merged is False


def test_closed_unmerged_pull_request_with_nullable_body_author_and_counts():
    pull_request = _pull_request(
        state=PullRequestState.CLOSED,
        body=None,
        include_author=False,
        closed_at=datetime(2026, 1, 3, 12, 0, tzinfo=UTC),
        merged_at=None,
        commits_count=None,
        changed_files_count=None,
        additions=None,
        deletions=None,
    )

    assert pull_request.body is None
    assert pull_request.author is None
    assert pull_request.closed_at is not None
    assert pull_request.merged_at is None
    assert pull_request.is_merged is False
    assert pull_request.commits_count is None
    assert pull_request.changed_files_count is None


def test_closed_merged_draft_pull_request():
    pull_request = _pull_request(
        state=PullRequestState.CLOSED,
        draft=True,
        closed_at=datetime(2026, 1, 3, 12, 0, tzinfo=UTC),
        merged_at=datetime(2026, 1, 3, 12, 5, tzinfo=UTC),
    )

    assert pull_request.draft is True
    assert pull_request.is_merged is True


def test_pull_request_rejects_invalid_state_and_naive_timestamp():
    with pytest.raises(ValidationError):
        _pull_request(state="merged")

    naive_created_at = datetime(2026, 1, 1, 12, 0, tzinfo=UTC).replace(tzinfo=None)
    with pytest.raises(ValidationError):
        _pull_request(created_at=naive_created_at)


def test_pull_request_file_and_commit_reference_models():
    changed_file = GitHubPullRequestFile(
        filename="src/app.py",
        status=GitHubPullRequestFileStatus.MODIFIED,
        additions=5,
        deletions=2,
        changes=7,
        patch=None,
        previous_filename=None,
        raw_url="https://raw.githubusercontent.com/owner/repo/main/src/app.py",
        blob_url="https://github.com/owner/repo/blob/main/src/app.py",
    )
    commit = GitHubCommitReference(
        sha="abc123",
        html_url="https://github.com/owner/repo/commit/abc123",
        message="Fix issue",
        author_name="Repo Tester",
        author_email="tester@example.com",
        authored_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )

    assert changed_file.patch is None
    assert commit.authored_at is not None
    assert commit.authored_at.tzinfo is not None


def test_pull_request_review_normalizes_uppercase_state_and_nullable_fields():
    review = GitHubPullRequestReview(
        repository=GitHubRepository(owner="owner", name="repo"),
        id=11,
        pull_request_number=123,
        author=None,
        body=None,
        state="APPROVED",
        submitted_at=datetime(2026, 1, 3, 12, 0, tzinfo=UTC),
        commit_sha="abc123",
        html_url=None,
    )

    assert review.state is ReviewState.APPROVED
    assert review.author is None
    assert review.body is None
    assert review.commit_sha == "abc123"
    assert review.submitted_at is not None
    assert review.submitted_at.tzinfo is not None


def test_pull_request_review_comment_preserves_code_location_metadata():
    comment = GitHubPullRequestReviewComment(
        repository=GitHubRepository(owner="owner", name="repo"),
        id=22,
        pull_request_number=123,
        review_id=11,
        author=GitHubUser(login="reviewer", id=2, html_url="https://github.com/reviewer"),
        body="This should be closed.",
        created_at=datetime(2026, 1, 4, 12, 0, tzinfo=UTC),
        updated_at=datetime(2026, 1, 5, 12, 0, tzinfo=UTC),
        html_url="https://github.com/owner/repo/pull/123#discussion_r22",
        commit_sha="abc123",
        original_commit_sha="def456",
        path="src/session.py",
        line=42,
        original_line=40,
        side="RIGHT",
        start_line=39,
        start_side="RIGHT",
    )

    assert comment.review_id == 11
    assert comment.path == "src/session.py"
    assert comment.line == 42
    assert comment.original_commit_sha == "def456"
    assert comment.author is not None
    assert comment.author.login == "reviewer"


def _pull_request(
    *,
    state: PullRequestState | str = PullRequestState.OPEN,
    body: str | None = "Fixes a bug.",
    author: GitHubUser | None = None,
    include_author: bool = True,
    labels: list[GitHubIssueLabel] | None = None,
    draft: bool = False,
    created_at: datetime = datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    updated_at: datetime = datetime(2026, 1, 2, 12, 0, tzinfo=UTC),
    closed_at: datetime | None = None,
    merged_at: datetime | None = None,
    head_repository: GitHubRepository | None = None,
    commits_count: int | None = 2,
    changed_files_count: int | None = 3,
    additions: int | None = 10,
    deletions: int | None = 4,
) -> GitHubPullRequest:
    repository = GitHubRepository(owner="owner", name="repo")
    pull_request_author = author
    if pull_request_author is None and include_author:
        pull_request_author = GitHubUser(login="octocat", id=1, html_url="https://github.com/octocat")
    return GitHubPullRequest(
        repository=repository,
        number=123,
        title="Fix bug",
        body=body,
        state=state,
        author=pull_request_author,
        labels=labels if labels is not None else [],
        draft=draft,
        locked=False,
        created_at=created_at,
        updated_at=updated_at,
        closed_at=closed_at,
        merged_at=merged_at,
        html_url="https://github.com/owner/repo/pull/123",
        head=GitHubBranchReference(
            repository=head_repository,
            ref="fix/session-cleanup",
            sha="headsha",
            label="contributor:fix/session-cleanup",
        ),
        base=GitHubBranchReference(
            repository=repository,
            ref="main",
            sha="basesha",
            label="owner:main",
        ),
        merge_commit_sha="mergesha",
        commits_count=commits_count,
        changed_files_count=changed_files_count,
        additions=additions,
        deletions=deletions,
        comments_count=1,
        review_comments_count=2,
        maintainer_can_modify=True,
    )
