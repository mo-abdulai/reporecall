import httpx
import pytest

from reporecall.github import GitHubClient, GitHubRepository, GitHubResponseError
from reporecall.ingestion import (
    GitHubIssueLoader,
    InvalidIssueLimitError,
    InvalidIssueNumberError,
    NotAnIssueError,
)
from reporecall.models import IssueState


def test_get_issues_excludes_pull_requests_from_listing():
    requests = _capture_requests(
        [
            httpx.Response(
                200,
                json=[
                    _issue(1, "First"),
                    _issue(2, "Second"),
                    _pull_request(3, "PR"),
                    _issue(4, "Third"),
                ],
            )
        ]
    )

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubIssueLoader(client, _repository())
        issues = loader.get_issues()

    assert [issue.number for issue in issues] == [1, 2, 4]
    assert len(requests.seen) == 1


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (IssueState.OPEN, "open"),
        (IssueState.CLOSED, "closed"),
        (None, "all"),
    ],
)
def test_get_issues_sends_state_filter(state: IssueState | None, expected: str):
    requests = _capture_requests([httpx.Response(200, json=[])])

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubIssueLoader(client, _repository())
        loader.get_issues(state=state)

    assert requests.seen[0].url.path == "/repos/owner/repo/issues"
    assert requests.seen[0].url.params["state"] == expected


def test_get_issues_preserves_paginated_api_order():
    responses = [
        httpx.Response(
            200,
            json=[_issue(1, "First"), _pull_request(2, "PR"), _issue(3, "Second")],
            headers={"Link": '<https://api.github.test/repos/owner/repo/issues?page=2>; rel="next"'},
        ),
        httpx.Response(200, json=[_issue(4, "Third"), _issue(5, "Fourth")]),
    ]
    requests = _capture_requests(responses)

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubIssueLoader(client, _repository())
        issues = loader.get_issues()

    assert [issue.number for issue in issues] == [1, 3, 4, 5]
    assert len(requests.seen) == 2


def test_get_issues_limit_counts_actual_issues_after_pr_exclusion():
    responses = [
        httpx.Response(
            200,
            json=[
                _pull_request(1, "PR 1"),
                _pull_request(2, "PR 2"),
                _pull_request(3, "PR 3"),
                _issue(4, "First"),
                _issue(5, "Second"),
            ],
            headers={"Link": '<https://api.github.test/repos/owner/repo/issues?page=2>; rel="next"'},
        ),
        httpx.Response(
            200,
            json=[
                _issue(6, "Third"),
                _issue(7, "Fourth"),
                _issue(8, "Fifth"),
                _issue(9, "Sixth"),
                _issue(10, "Seventh"),
            ],
        ),
    ]
    requests = _capture_requests(responses)

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubIssueLoader(client, _repository())
        issues = loader.get_issues(limit=4)

    assert [issue.number for issue in issues] == [4, 5, 6, 7]


@pytest.mark.parametrize(("limit", "expected"), [(1, [1]), (10, [1, 2]), (None, [1, 2])])
def test_get_issues_limit_variants(limit: int | None, expected: list[int]):
    requests = _capture_requests([httpx.Response(200, json=[_issue(1, "First"), _issue(2, "Second")])])

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubIssueLoader(client, _repository())
        issues = loader.get_issues(limit=limit)

    assert [issue.number for issue in issues] == expected


@pytest.mark.parametrize("limit", [0, -1])
def test_get_issues_rejects_invalid_limit_without_request(limit: int):
    requests = _capture_requests([])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(InvalidIssueLimitError),
    ):
        loader = GitHubIssueLoader(client, _repository())
        loader.get_issues(limit=limit)

    assert requests.seen == []


def test_get_issue_normalizes_single_issue():
    requests = _capture_requests(
        [
            httpx.Response(
                200,
                json=_issue(
                    123,
                    "Fix auth bug",
                    body="Details",
                    state="closed",
                    closed_at="2026-01-03T12:00:00Z",
                    labels=[
                        {"name": "bug", "color": "d73a4a", "description": "Something is broken."},
                        {"name": "auth", "color": "5319e7", "description": None},
                    ],
                    milestone={
                        "number": 7,
                        "title": "Authentication cleanup",
                        "html_url": "https://github.com/owner/repo/milestone/7",
                    },
                ),
            )
        ]
    )

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubIssueLoader(client, _repository())
        issue = loader.get_issue(123)

    assert requests.seen[0].url.path == "/repos/owner/repo/issues/123"
    assert issue.repository == _repository()
    assert issue.number == 123
    assert issue.title == "Fix auth bug"
    assert issue.body == "Details"
    assert issue.state is IssueState.CLOSED
    assert issue.author is not None
    assert issue.author.login == "octocat"
    assert [label.name for label in issue.labels] == ["bug", "auth"]
    assert issue.milestone is not None
    assert issue.milestone.number == 7
    assert issue.milestone.title == "Authentication cleanup"
    assert issue.created_at.tzinfo is not None
    assert issue.html_url == "https://github.com/owner/repo/issues/123"


def test_get_issue_rejects_pull_request_response():
    requests = _capture_requests([httpx.Response(200, json=_pull_request(123, "PR"))])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(NotAnIssueError),
    ):
        loader = GitHubIssueLoader(client, _repository())
        loader.get_issue(123)


@pytest.mark.parametrize("number", [0, -1])
def test_get_issue_rejects_invalid_number_without_request(number: int):
    requests = _capture_requests([])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(InvalidIssueNumberError),
    ):
        loader = GitHubIssueLoader(client, _repository())
        loader.get_issue(number)

    assert requests.seen == []


@pytest.mark.parametrize(
    "case",
    ["missing_number", "invalid_timestamp", "bad_labels", "bad_milestone"],
)
def test_malformed_issue_response_fails_clearly(case: str):
    payloads = {
        "missing_number": {"title": "Missing number"},
        "invalid_timestamp": _issue(1, "Invalid timestamp", created_at="not-a-date"),
        "bad_labels": _issue(1, "Bad labels", labels=["bug"]),
        "bad_milestone": _issue(1, "Bad milestone", milestone="v1"),
    }
    requests = _capture_requests([httpx.Response(200, json=payloads[case])])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(GitHubResponseError),
    ):
        loader = GitHubIssueLoader(client, _repository())
        loader.get_issue(1)


class CapturedRequests:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = responses
        self.seen: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self._handler)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        response = self._responses.pop(0)
        return response


def _capture_requests(responses: list[httpx.Response]) -> CapturedRequests:
    return CapturedRequests(responses)


def _repository() -> GitHubRepository:
    return GitHubRepository(owner="owner", name="repo")


def _issue(
    number: int,
    title: str,
    *,
    body: str | None = None,
    state: str = "open",
    created_at: str = "2026-01-01T12:00:00Z",
    updated_at: str = "2026-01-02T12:00:00Z",
    closed_at: str | None = None,
    labels: list[dict[str, object]] | None = None,
    milestone: object = None,
) -> dict[str, object]:
    return {
        "number": number,
        "title": title,
        "body": body,
        "state": state,
        "user": {"login": "octocat", "id": 1, "html_url": "https://github.com/octocat"},
        "labels": labels if labels is not None else [],
        "milestone": milestone,
        "created_at": created_at,
        "updated_at": updated_at,
        "closed_at": closed_at,
        "html_url": f"https://github.com/owner/repo/issues/{number}",
        "comments": 3,
        "locked": False,
    }


def _pull_request(number: int, title: str) -> dict[str, object]:
    item = _issue(number, title)
    item["pull_request"] = {"url": f"https://api.github.test/repos/owner/repo/pulls/{number}"}
    return item
