from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from reporecall.github import GitHubRepository
from reporecall.models import (
    GitHubIssue,
    GitHubIssueComment,
    GitHubIssueLabel,
    GitHubUser,
    IssueState,
)


def test_github_issue_constructs_open_issue_with_metadata():
    issue = GitHubIssue(
        repository=GitHubRepository(owner="fastapi", name="fastapi"),
        number=123,
        title="Fix database leak",
        body="Connection cleanup fails.",
        state=IssueState.OPEN,
        author=GitHubUser(login="octocat", id=1, html_url="https://github.com/octocat"),
        labels=[
            GitHubIssueLabel(name="bug", color="d73a4a", description="Something is broken."),
            GitHubIssueLabel(name="database", color="0075ca", description=None),
        ],
        created_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        updated_at=datetime(2026, 1, 2, 12, 0, tzinfo=UTC),
        closed_at=None,
        html_url="https://github.com/fastapi/fastapi/issues/123",
        comments_count=2,
        locked=False,
    )

    assert issue.repository.owner == "fastapi"
    assert issue.state is IssueState.OPEN
    assert issue.author is not None
    assert issue.author.login == "octocat"
    assert [label.name for label in issue.labels] == ["bug", "database"]
    assert issue.created_at.tzinfo is not None


def test_github_issue_constructs_closed_issue_with_nullable_body_and_author():
    issue = GitHubIssue(
        repository=GitHubRepository(owner="django", name="django"),
        number=456,
        title="Closed issue",
        body=None,
        state=IssueState.CLOSED,
        author=None,
        labels=[],
        created_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        updated_at=datetime(2026, 1, 2, 12, 0, tzinfo=UTC),
        closed_at=datetime(2026, 1, 3, 12, 0, tzinfo=UTC),
        html_url="https://github.com/django/django/issues/456",
        comments_count=0,
        locked=True,
    )

    assert issue.body is None
    assert issue.author is None
    assert issue.closed_at is not None
    assert issue.closed_at.tzinfo is not None


def test_github_issue_comment_preserves_parent_repository_author_and_url():
    comment = GitHubIssueComment(
        repository=GitHubRepository(owner="fastapi", name="fastapi"),
        id=987,
        issue_number=123,
        author=GitHubUser(login="octocat", id=1, html_url="https://github.com/octocat"),
        body=None,
        created_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        updated_at=datetime(2026, 1, 2, 12, 0, tzinfo=UTC),
        html_url="https://github.com/fastapi/fastapi/issues/123#issuecomment-987",
        author_association="CONTRIBUTOR",
    )

    assert comment.repository == GitHubRepository(owner="fastapi", name="fastapi")
    assert comment.issue_number == 123
    assert comment.author is not None
    assert comment.author.login == "octocat"
    assert comment.body is None
    assert comment.created_at.tzinfo is not None
    assert comment.html_url.endswith("#issuecomment-987")


def test_github_issue_rejects_invalid_state():
    with pytest.raises(ValidationError):
        GitHubIssue(
            repository=GitHubRepository(owner="fastapi", name="fastapi"),
            number=123,
            title="Invalid state",
            body=None,
            state="merged",
            author=None,
            labels=[],
            created_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            updated_at=datetime(2026, 1, 2, 12, 0, tzinfo=UTC),
            closed_at=None,
            html_url="https://github.com/fastapi/fastapi/issues/123",
            comments_count=0,
            locked=False,
        )


def test_github_issue_comment_rejects_naive_timestamps():
    naive_created_at = datetime(2026, 1, 1, 12, 0, tzinfo=UTC).replace(tzinfo=None)

    with pytest.raises(ValidationError):
        GitHubIssueComment(
            repository=GitHubRepository(owner="fastapi", name="fastapi"),
            id=987,
            issue_number=123,
            author=None,
            body="Looks related.",
            created_at=naive_created_at,
            updated_at=datetime(2026, 1, 2, 12, 0, tzinfo=UTC),
            html_url="https://github.com/fastapi/fastapi/issues/123#issuecomment-987",
        )


def test_github_issue_rejects_naive_timestamps():
    naive_created_at = datetime(2026, 1, 1, 12, 0, tzinfo=UTC).replace(tzinfo=None)

    with pytest.raises(ValidationError):
        GitHubIssue(
            repository=GitHubRepository(owner="fastapi", name="fastapi"),
            number=123,
            title="Naive timestamp",
            body=None,
            state=IssueState.OPEN,
            author=None,
            labels=[],
            created_at=naive_created_at,
            updated_at=datetime(2026, 1, 2, 12, 0, tzinfo=UTC),
            closed_at=None,
            html_url="https://github.com/fastapi/fastapi/issues/123",
            comments_count=0,
            locked=False,
        )
