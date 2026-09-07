import httpx
import pytest

from reporecall.github import GitHubClient, GitHubRepository, GitHubResponseError
from reporecall.ingestion import (
    GitHubPullRequestLoader,
    GitHubReviewLoader,
    InvalidPullRequestNumberError,
)
from reporecall.models import ReviewState


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("APPROVED", ReviewState.APPROVED),
        ("CHANGES_REQUESTED", ReviewState.CHANGES_REQUESTED),
        ("COMMENTED", ReviewState.COMMENTED),
        ("DISMISSED", ReviewState.DISMISSED),
    ],
)
def test_get_pull_request_reviews_normalizes_review_states(state: str, expected: ReviewState):
    requests = _capture_requests([httpx.Response(200, json=[_review(10, state=state)])])

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubReviewLoader(client, _repository())
        reviews = loader.get_pull_request_reviews(456)

    assert requests.seen[0].url.path == "/repos/owner/repo/pulls/456/reviews"
    assert reviews[0].repository == _repository()
    assert reviews[0].id == 10
    assert reviews[0].pull_request_number == 456
    assert reviews[0].state is expected
    assert reviews[0].author is not None
    assert reviews[0].author.login == "reviewer"
    assert reviews[0].commit_sha == "abc123"
    assert reviews[0].submitted_at is not None
    assert reviews[0].submitted_at.tzinfo is not None
    assert reviews[0].html_url == "https://github.com/owner/repo/pull/456#pullrequestreview-10"


def test_get_pull_request_reviews_supports_nullable_fields_pagination_and_empty_results():
    responses = [
        httpx.Response(200, json=[]),
        httpx.Response(
            200,
            json=[_review(10, body=None, include_user=False)],
            headers={"Link": '<https://api.github.test/repos/owner/repo/pulls/456/reviews?page=2>; rel="next"'},
        ),
        httpx.Response(200, json=[_review(11, state="PENDING", submitted_at=None)]),
    ]
    requests = _capture_requests(responses)

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubReviewLoader(client, _repository())
        assert loader.get_pull_request_reviews(455) == []
        reviews = loader.get_pull_request_reviews(456)

    assert [review.id for review in reviews] == [10, 11]
    assert reviews[0].body is None
    assert reviews[0].author is None
    assert reviews[1].state is ReviewState.PENDING
    assert reviews[1].submitted_at is None
    assert len(requests.seen) == 3


@pytest.mark.parametrize("number", [0, -1])
def test_get_pull_request_reviews_rejects_invalid_number_without_request(number: int):
    requests = _capture_requests([])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(InvalidPullRequestNumberError),
    ):
        loader = GitHubReviewLoader(client, _repository())
        loader.get_pull_request_reviews(number)

    assert requests.seen == []


@pytest.mark.parametrize("case", ["missing_id", "invalid_state", "invalid_timestamp", "bad_author"])
def test_malformed_pull_request_review_response_fails_clearly(case: str):
    payloads = {
        "missing_id": {"state": "APPROVED"},
        "invalid_state": _review(10, state="STALE"),
        "invalid_timestamp": _review(10, submitted_at="not-a-date"),
        "bad_author": _review(10, user="reviewer"),
    }
    requests = _capture_requests([httpx.Response(200, json=[payloads[case]])])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(GitHubResponseError),
    ):
        loader = GitHubReviewLoader(client, _repository())
        loader.get_pull_request_reviews(456)


def test_get_pull_request_review_comments_normalizes_code_location_metadata():
    requests = _capture_requests([httpx.Response(200, json=[_review_comment(20)])])

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubReviewLoader(client, _repository())
        comments = loader.get_pull_request_review_comments(456)

    assert requests.seen[0].url.path == "/repos/owner/repo/pulls/456/comments"
    assert comments[0].repository == _repository()
    assert comments[0].id == 20
    assert comments[0].pull_request_number == 456
    assert comments[0].review_id == 10
    assert comments[0].author is not None
    assert comments[0].author.login == "reviewer"
    assert comments[0].body == "Consider simplifying this."
    assert comments[0].created_at.tzinfo is not None
    assert comments[0].updated_at.tzinfo is not None
    assert comments[0].html_url == "https://github.com/owner/repo/pull/456#discussion_r20"
    assert comments[0].commit_sha == "abc123"
    assert comments[0].original_commit_sha == "def456"
    assert comments[0].path == "src/session.py"
    assert comments[0].line == 42
    assert comments[0].original_line == 40
    assert comments[0].side == "RIGHT"
    assert comments[0].start_line == 39
    assert comments[0].start_side == "RIGHT"


def test_get_pull_request_review_comments_supports_optional_locations_pagination_and_empty_results():
    responses = [
        httpx.Response(200, json=[]),
        httpx.Response(
            200,
            json=[_review_comment(20, line=None, original_line=None, side=None)],
            headers={"Link": '<https://api.github.test/repos/owner/repo/pulls/456/comments?page=2>; rel="next"'},
        ),
        httpx.Response(200, json=[_review_comment(21, include_user=False)]),
    ]
    requests = _capture_requests(responses)

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubReviewLoader(client, _repository())
        assert loader.get_pull_request_review_comments(455) == []
        comments = loader.get_pull_request_review_comments(456)

    assert [comment.id for comment in comments] == [20, 21]
    assert comments[0].line is None
    assert comments[0].original_line is None
    assert comments[0].side is None
    assert comments[1].author is None
    assert len(requests.seen) == 3


@pytest.mark.parametrize("number", [0, -1])
def test_get_pull_request_review_comments_rejects_invalid_number_without_request(number: int):
    requests = _capture_requests([])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(InvalidPullRequestNumberError),
    ):
        loader = GitHubReviewLoader(client, _repository())
        loader.get_pull_request_review_comments(number)

    assert requests.seen == []


@pytest.mark.parametrize(
    "case",
    ["missing_id", "missing_review_id", "missing_path", "invalid_timestamp", "bad_author"],
)
def test_malformed_pull_request_review_comment_response_fails_clearly(case: str):
    payloads = {
        "missing_id": _without(_review_comment(20), "id"),
        "missing_review_id": _without(_review_comment(20), "pull_request_review_id"),
        "missing_path": _without(_review_comment(20), "path"),
        "invalid_timestamp": _review_comment(20, created_at="not-a-date"),
        "bad_author": _review_comment(20, user="reviewer"),
    }
    requests = _capture_requests([httpx.Response(200, json=[payloads[case]])])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(GitHubResponseError),
    ):
        loader = GitHubReviewLoader(client, _repository())
        loader.get_pull_request_review_comments(456)


def test_get_pull_requests_does_not_fetch_discussion_endpoints():
    requests = _capture_requests([httpx.Response(200, json=[_pull_request(456)])])

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubPullRequestLoader(client, _repository())
        loader.get_pull_requests()

    assert [request.url.path for request in requests.seen] == ["/repos/owner/repo/pulls"]


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


def _review(
    review_id: int,
    *,
    state: str = "APPROVED",
    body: str | None = "Looks good.",
    user: dict[str, object] | str | None = None,
    include_user: bool = True,
    submitted_at: str | None = "2026-01-03T12:00:00Z",
) -> dict[str, object]:
    if user is None and include_user:
        user = {"login": "reviewer", "id": 2, "html_url": "https://github.com/reviewer"}
    return {
        "id": review_id,
        "user": user,
        "body": body,
        "state": state,
        "submitted_at": submitted_at,
        "commit_id": "abc123",
        "html_url": f"https://github.com/owner/repo/pull/456#pullrequestreview-{review_id}",
    }


def _review_comment(
    comment_id: int,
    *,
    user: dict[str, object] | str | None = None,
    include_user: bool = True,
    created_at: str = "2026-01-04T12:00:00Z",
    line: int | None = 42,
    original_line: int | None = 40,
    side: str | None = "RIGHT",
) -> dict[str, object]:
    if user is None and include_user:
        user = {"login": "reviewer", "id": 2, "html_url": "https://github.com/reviewer"}
    return {
        "id": comment_id,
        "pull_request_review_id": 10,
        "user": user,
        "body": "Consider simplifying this.",
        "created_at": created_at,
        "updated_at": "2026-01-05T12:00:00Z",
        "html_url": f"https://github.com/owner/repo/pull/456#discussion_r{comment_id}",
        "commit_id": "abc123",
        "original_commit_id": "def456",
        "path": "src/session.py",
        "line": line,
        "original_line": original_line,
        "side": side,
        "start_line": 39,
        "start_side": "RIGHT",
        "author_association": "MEMBER",
    }


def _pull_request(number: int) -> dict[str, object]:
    return {
        "number": number,
        "title": "Fix bug",
        "body": None,
        "state": "open",
        "user": {"login": "octocat", "id": 1, "html_url": "https://github.com/octocat"},
        "labels": [],
        "draft": False,
        "locked": False,
        "created_at": "2026-01-01T12:00:00Z",
        "updated_at": "2026-01-02T12:00:00Z",
        "closed_at": None,
        "merged_at": None,
        "html_url": f"https://github.com/owner/repo/pull/{number}",
        "head": {"ref": "feature", "sha": "headsha", "label": "owner:feature", "repo": _repo_payload()},
        "base": {"ref": "main", "sha": "basesha", "label": "owner:main", "repo": _repo_payload()},
        "merge_commit_sha": "mergesha",
        "comments": 1,
        "review_comments": 2,
    }


def _repo_payload() -> dict[str, object]:
    return {"name": "repo", "owner": {"login": "owner"}}


def _without(payload: dict[str, object], key: str) -> dict[str, object]:
    copy = dict(payload)
    del copy[key]
    return copy
