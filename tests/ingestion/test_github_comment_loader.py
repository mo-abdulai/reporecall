import httpx
import pytest

from reporecall.github import GitHubClient, GitHubRepository, GitHubResponseError
from reporecall.ingestion import (
    GitHubCommentLoader,
    GitHubIssueLoader,
    InvalidIssueNumberError,
    InvalidPullRequestNumberError,
)


def test_get_issue_comments_normalizes_one_comment():
    requests = _capture_requests([httpx.Response(200, json=[_comment(100, body="Looks related.")])])

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubCommentLoader(client, _repository())
        comments = loader.get_issue_comments(123)

    assert requests.seen[0].url.path == "/repos/owner/repo/issues/123/comments"
    assert len(comments) == 1
    assert comments[0].repository == _repository()
    assert comments[0].id == 100
    assert comments[0].issue_number == 123
    assert comments[0].author is not None
    assert comments[0].author.login == "octocat"
    assert comments[0].body == "Looks related."
    assert comments[0].created_at.tzinfo is not None
    assert comments[0].updated_at.tzinfo is not None
    assert comments[0].html_url == "https://github.com/owner/repo/issues/123#issuecomment-100"
    assert comments[0].author_association == "CONTRIBUTOR"


def test_get_issue_comments_preserves_multiple_comments_and_pages():
    responses = [
        httpx.Response(
            200,
            json=[_comment(100), _comment(101)],
            headers={"Link": '<https://api.github.test/repos/owner/repo/issues/123/comments?page=2>; rel="next"'},
        ),
        httpx.Response(200, json=[_comment(102)]),
    ]
    requests = _capture_requests(responses)

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubCommentLoader(client, _repository())
        comments = loader.get_issue_comments(123)

    assert [comment.id for comment in comments] == [100, 101, 102]
    assert len(requests.seen) == 2


def test_get_issue_comments_supports_empty_results_nullable_author_and_body():
    requests = _capture_requests(
        [
            httpx.Response(200, json=[]),
            httpx.Response(200, json=[_comment(100, body=None, include_user=False)]),
        ]
    )

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubCommentLoader(client, _repository())
        assert loader.get_issue_comments(123) == []
        comments = loader.get_issue_comments(124)

    assert comments[0].body is None
    assert comments[0].author is None


@pytest.mark.parametrize("number", [0, -1])
def test_get_issue_comments_rejects_invalid_number_without_request(number: int):
    requests = _capture_requests([])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(InvalidIssueNumberError),
    ):
        loader = GitHubCommentLoader(client, _repository())
        loader.get_issue_comments(number)

    assert requests.seen == []


@pytest.mark.parametrize("case", ["missing_id", "invalid_timestamp", "bad_author"])
def test_malformed_issue_comment_response_fails_clearly(case: str):
    payloads = {
        "missing_id": {"created_at": "2026-01-01T12:00:00Z", "updated_at": "2026-01-02T12:00:00Z", "html_url": "x"},
        "invalid_timestamp": _comment(100, created_at="not-a-date"),
        "bad_author": _comment(100, user="octocat"),
    }
    requests = _capture_requests([httpx.Response(200, json=[payloads[case]])])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(GitHubResponseError),
    ):
        loader = GitHubCommentLoader(client, _repository())
        loader.get_issue_comments(123)


def test_get_pull_request_comments_uses_issue_comment_endpoint_and_parent_pr_number():
    responses = [
        httpx.Response(
            200,
            json=[_comment(200)],
            headers={"Link": '<https://api.github.test/repos/owner/repo/issues/456/comments?page=2>; rel="next"'},
        ),
        httpx.Response(200, json=[_comment(201)]),
    ]
    requests = _capture_requests(responses)

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubCommentLoader(client, _repository())
        comments = loader.get_pull_request_comments(456)

    assert [request.url.path for request in requests.seen] == [
        "/repos/owner/repo/issues/456/comments",
        "/repos/owner/repo/issues/456/comments",
    ]
    assert [comment.issue_number for comment in comments] == [456, 456]
    assert [comment.id for comment in comments] == [200, 201]


def test_get_pull_request_comments_supports_empty_results():
    requests = _capture_requests([httpx.Response(200, json=[])])

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubCommentLoader(client, _repository())
        assert loader.get_pull_request_comments(456) == []


@pytest.mark.parametrize("number", [0, -1])
def test_get_pull_request_comments_rejects_invalid_number_without_request(number: int):
    requests = _capture_requests([])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(InvalidPullRequestNumberError),
    ):
        loader = GitHubCommentLoader(client, _repository())
        loader.get_pull_request_comments(number)

    assert requests.seen == []


def test_get_issues_does_not_fetch_issue_comment_endpoints():
    requests = _capture_requests([httpx.Response(200, json=[_issue(1)])])

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubIssueLoader(client, _repository())
        loader.get_issues()

    assert [request.url.path for request in requests.seen] == ["/repos/owner/repo/issues"]


class CapturedRequests:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = responses
        self.seen: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self._handler)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        return self._responses.pop(0)


def _capture_requests(responses: list[httpx.Response]) -> CapturedRequests:
    return CapturedRequests(responses)


def _repository() -> GitHubRepository:
    return GitHubRepository(owner="owner", name="repo")


def _comment(
    comment_id: int,
    *,
    body: str | None = "Comment body",
    user: dict[str, object] | str | None = None,
    include_user: bool = True,
    created_at: str = "2026-01-01T12:00:00Z",
) -> dict[str, object]:
    if user is None and include_user:
        user = {"login": "octocat", "id": 1, "html_url": "https://github.com/octocat"}
    return {
        "id": comment_id,
        "user": user,
        "body": body,
        "created_at": created_at,
        "updated_at": "2026-01-02T12:00:00Z",
        "html_url": f"https://github.com/owner/repo/issues/123#issuecomment-{comment_id}",
        "author_association": "CONTRIBUTOR",
    }


def _issue(number: int) -> dict[str, object]:
    return {
        "number": number,
        "title": "Fix bug",
        "body": None,
        "state": "open",
        "user": {"login": "octocat", "id": 1, "html_url": "https://github.com/octocat"},
        "labels": [],
        "created_at": "2026-01-01T12:00:00Z",
        "updated_at": "2026-01-02T12:00:00Z",
        "closed_at": None,
        "html_url": f"https://github.com/owner/repo/issues/{number}",
        "comments": 3,
        "locked": False,
    }
