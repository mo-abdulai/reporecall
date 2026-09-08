import httpx
import pytest

from reporecall.github import GitHubClient, GitHubRepository, GitHubResponseError
from reporecall.ingestion import (
    GitHubPullRequestLoader,
    InvalidPullRequestLimitError,
    InvalidPullRequestNumberError,
)
from reporecall.models import GitHubPullRequestFileStatus, PullRequestState


def test_get_pull_requests_returns_normalized_prs_in_api_order():
    requests = _capture_requests([httpx.Response(200, json=[_pull_request(1, "First"), _pull_request(2, "Second")])])

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubPullRequestLoader(client, _repository())
        pull_requests = loader.get_pull_requests()

    assert [pull_request.number for pull_request in pull_requests] == [1, 2]
    assert all(pull_request.repository == _repository() for pull_request in pull_requests)


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (PullRequestState.OPEN, "open"),
        (PullRequestState.CLOSED, "closed"),
        (None, "all"),
    ],
)
def test_get_pull_requests_sends_state_filter(state: PullRequestState | None, expected: str):
    requests = _capture_requests([httpx.Response(200, json=[])])

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubPullRequestLoader(client, _repository())
        loader.get_pull_requests(state=state)

    assert requests.seen[0].url.path == "/repos/owner/repo/pulls"
    assert requests.seen[0].url.params["state"] == expected


def test_get_pull_requests_follows_pagination():
    responses = [
        httpx.Response(
            200,
            json=[_pull_request(1, "First")],
            headers={"Link": '<https://api.github.test/repos/owner/repo/pulls?page=2>; rel="next"'},
        ),
        httpx.Response(
            200,
            json=[_pull_request(2, "Second")],
            headers={"Link": '<https://api.github.test/repos/owner/repo/pulls?page=3>; rel="next"'},
        ),
        httpx.Response(200, json=[_pull_request(3, "Third")]),
    ]
    requests = _capture_requests(responses)

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubPullRequestLoader(client, _repository())
        pull_requests = loader.get_pull_requests()

    assert [pull_request.number for pull_request in pull_requests] == [1, 2, 3]
    assert len(requests.seen) == 3


@pytest.mark.parametrize(("limit", "expected"), [(1, [1]), (5, [1, 2, 3]), (10, [1, 2, 3]), (None, [1, 2, 3])])
def test_get_pull_requests_limit_variants(limit: int | None, expected: list[int]):
    requests = _capture_requests(
        [httpx.Response(200, json=[_pull_request(1, "First"), _pull_request(2, "Second"), _pull_request(3, "Third")])]
    )

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubPullRequestLoader(client, _repository())
        pull_requests = loader.get_pull_requests(limit=limit)

    assert [pull_request.number for pull_request in pull_requests] == expected


@pytest.mark.parametrize("limit", [0, -1])
def test_get_pull_requests_rejects_invalid_limit_without_request(limit: int):
    requests = _capture_requests([])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(InvalidPullRequestLimitError),
    ):
        loader = GitHubPullRequestLoader(client, _repository())
        loader.get_pull_requests(limit=limit)

    assert requests.seen == []


def test_get_pull_requests_does_not_fetch_files_or_commits():
    requests = _capture_requests([httpx.Response(200, json=[_pull_request(1, "First"), _pull_request(2, "Second")])])

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubPullRequestLoader(client, _repository())
        loader.get_pull_requests(limit=10)

    assert [request.url.path for request in requests.seen] == ["/repos/owner/repo/pulls"]


def test_empty_pull_request_results_are_valid():
    requests = _capture_requests([httpx.Response(200, json=[]), httpx.Response(200, json=[]), httpx.Response(200, json=[])])

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubPullRequestLoader(client, _repository())
        assert loader.get_pull_requests() == []
        assert loader.get_pull_request_files(123) == []
        assert loader.get_pull_request_commits(123) == []


def test_get_pull_request_normalizes_single_detail_response():
    requests = _capture_requests(
        [
            httpx.Response(
                200,
                json=_pull_request(
                    123,
                    "Fix auth bug",
                    body="Details",
                    state="closed",
                    closed_at="2026-01-03T12:00:00Z",
                    merged_at="2026-01-03T12:05:00Z",
                    draft=True,
                    head_repo_owner="contributor",
                    head_repo_name="fork",
                    labels=[{"name": "bug", "color": "d73a4a", "description": "Something is broken."}],
                    milestone={
                        "number": 8,
                        "title": "Session stability",
                        "html_url": "https://github.com/owner/repo/milestone/8",
                    },
                ),
            )
        ]
    )

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubPullRequestLoader(client, _repository())
        pull_request = loader.get_pull_request(123)

    assert requests.seen[0].url.path == "/repos/owner/repo/pulls/123"
    assert pull_request.number == 123
    assert pull_request.title == "Fix auth bug"
    assert pull_request.body == "Details"
    assert pull_request.state is PullRequestState.CLOSED
    assert pull_request.draft is True
    assert pull_request.author is not None
    assert pull_request.author.login == "octocat"
    assert [label.name for label in pull_request.labels] == ["bug"]
    assert pull_request.milestone is not None
    assert pull_request.milestone.number == 8
    assert pull_request.milestone.title == "Session stability"
    assert pull_request.merged_at is not None
    assert pull_request.is_merged is True
    assert pull_request.merge_commit_sha == "mergesha"
    assert pull_request.head.repository == GitHubRepository(owner="contributor", name="fork")
    assert pull_request.base.repository == _repository()
    assert pull_request.commits_count == 2
    assert pull_request.changed_files_count == 3
    assert pull_request.html_url == "https://github.com/owner/repo/pull/123"


@pytest.mark.parametrize("number", [0, -1])
def test_get_pull_request_rejects_invalid_number_without_request(number: int):
    requests = _capture_requests([])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(InvalidPullRequestNumberError),
    ):
        loader = GitHubPullRequestLoader(client, _repository())
        loader.get_pull_request(number)

    assert requests.seen == []


@pytest.mark.parametrize("method_name", ["get_pull_request_files", "get_pull_request_commits"])
@pytest.mark.parametrize("number", [0, -1])
def test_enrichment_methods_reject_invalid_number_without_request(method_name: str, number: int):
    requests = _capture_requests([])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(InvalidPullRequestNumberError),
    ):
        loader = GitHubPullRequestLoader(client, _repository())
        method = getattr(loader, method_name)
        method(number)

    assert requests.seen == []


def test_get_pull_request_files_normalizes_statuses_and_paginates():
    responses = [
        httpx.Response(
            200,
            json=[
                _file("src/new.py", "added", patch="+print('new')"),
                _file("src/app.py", "modified", patch=None),
            ],
            headers={"Link": '<https://api.github.test/repos/owner/repo/pulls/123/files?page=2>; rel="next"'},
        ),
        httpx.Response(
            200,
            json=[
                _file("src/old.py", "removed"),
                _file("src/new_name.py", "renamed", previous_filename="src/old_name.py"),
            ],
        ),
    ]
    requests = _capture_requests(responses)

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubPullRequestLoader(client, _repository())
        files = loader.get_pull_request_files(123)

    assert [file.filename for file in files] == ["src/new.py", "src/app.py", "src/old.py", "src/new_name.py"]
    assert files[1].patch is None
    assert files[2].status is GitHubPullRequestFileStatus.DELETED
    assert files[3].previous_filename == "src/old_name.py"
    assert len(requests.seen) == 2


def test_get_pull_request_commits_normalizes_references_and_paginates():
    responses = [
        httpx.Response(
            200,
            json=[_commit("abc123", "First commit")],
            headers={"Link": '<https://api.github.test/repos/owner/repo/pulls/123/commits?page=2>; rel="next"'},
        ),
        httpx.Response(200, json=[_commit("def456", "Second commit", include_author=False)]),
    ]
    requests = _capture_requests(responses)

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        loader = GitHubPullRequestLoader(client, _repository())
        commits = loader.get_pull_request_commits(123)

    assert [commit.sha for commit in commits] == ["abc123", "def456"]
    assert commits[0].message == "First commit"
    assert commits[0].author_name == "Repo Tester"
    assert commits[0].author_email == "tester@example.com"
    assert commits[0].authored_at is not None
    assert commits[0].html_url == "https://github.com/owner/repo/commit/abc123"
    assert commits[1].author_name is None
    assert commits[1].authored_at is None
    assert len(requests.seen) == 2


@pytest.mark.parametrize(
    "case",
    [
        "missing_number",
        "invalid_timestamp",
        "missing_head_sha",
        "bad_labels",
        "bad_milestone",
    ],
)
def test_malformed_pull_request_response_fails_clearly(case: str):
    payloads = {
        "missing_number": {"title": "Missing number"},
        "invalid_timestamp": _pull_request(1, "Bad timestamp", created_at="not-a-date"),
        "missing_head_sha": _pull_request(1, "Missing head sha", head_sha=None),
        "bad_labels": _pull_request(1, "Bad labels", labels=["bug"]),
        "bad_milestone": _pull_request(1, "Bad milestone", milestone="v1"),
    }
    requests = _capture_requests([httpx.Response(200, json=payloads[case])])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(GitHubResponseError),
    ):
        loader = GitHubPullRequestLoader(client, _repository())
        loader.get_pull_request(1)


@pytest.mark.parametrize("case", ["missing_filename", "invalid_additions", "invalid_status"])
def test_malformed_pull_request_file_response_fails_clearly(case: str):
    payloads = {
        "missing_filename": {"status": "added", "additions": 1, "deletions": 0, "changes": 1},
        "invalid_additions": _file("src/app.py", "added", additions=-1),
        "invalid_status": _file("src/app.py", "moved"),
    }
    requests = _capture_requests([httpx.Response(200, json=[payloads[case]])])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(GitHubResponseError),
    ):
        loader = GitHubPullRequestLoader(client, _repository())
        loader.get_pull_request_files(1)


@pytest.mark.parametrize("case", ["missing_sha", "missing_commit", "invalid_timestamp"])
def test_malformed_pull_request_commit_response_fails_clearly(case: str):
    payloads = {
        "missing_sha": {"commit": {"message": "Missing sha"}},
        "missing_commit": {"sha": "abc123", "html_url": None},
        "invalid_timestamp": _commit(
            "abc123",
            "Invalid timestamp",
            author={"name": "Repo Tester", "email": "tester@example.com", "date": "x"},
        ),
    }
    requests = _capture_requests([httpx.Response(200, json=[payloads[case]])])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(GitHubResponseError),
    ):
        loader = GitHubPullRequestLoader(client, _repository())
        loader.get_pull_request_commits(1)


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


def _pull_request(
    number: int,
    title: str,
    *,
    body: str | None = None,
    state: str = "open",
    draft: bool = False,
    created_at: str = "2026-01-01T12:00:00Z",
    updated_at: str = "2026-01-02T12:00:00Z",
    closed_at: str | None = None,
    merged_at: str | None = None,
    labels: list[dict[str, object]] | None = None,
    milestone: object = None,
    head_repo_owner: str | None = "owner",
    head_repo_name: str | None = "repo",
    head_sha: str | None = "headsha",
) -> dict[str, object]:
    head: dict[str, object] = {
        "ref": "fix/session-cleanup",
        "sha": head_sha,
        "label": f"{head_repo_owner}:fix/session-cleanup",
        "repo": _repo_payload(head_repo_owner, head_repo_name) if head_repo_owner is not None else None,
    }
    return {
        "number": number,
        "title": title,
        "body": body,
        "state": state,
        "user": {"login": "octocat", "id": 1, "html_url": "https://github.com/octocat"},
        "labels": labels if labels is not None else [],
        "milestone": milestone,
        "draft": draft,
        "locked": False,
        "created_at": created_at,
        "updated_at": updated_at,
        "closed_at": closed_at,
        "merged_at": merged_at,
        "html_url": f"https://github.com/owner/repo/pull/{number}",
        "head": head,
        "base": {
            "ref": "main",
            "sha": "basesha",
            "label": "owner:main",
            "repo": _repo_payload("owner", "repo"),
        },
        "merge_commit_sha": "mergesha",
        "commits": 2,
        "changed_files": 3,
        "additions": 10,
        "deletions": 4,
        "comments": 1,
        "review_comments": 2,
        "maintainer_can_modify": True,
    }


def _repo_payload(owner: str | None, name: str | None) -> dict[str, object]:
    return {
        "name": name,
        "owner": {"login": owner},
    }


def _file(
    filename: str,
    status: str,
    *,
    additions: int = 5,
    deletions: int = 2,
    changes: int = 7,
    patch: str | None = "@@ patch",
    previous_filename: str | None = None,
) -> dict[str, object]:
    return {
        "filename": filename,
        "status": status,
        "additions": additions,
        "deletions": deletions,
        "changes": changes,
        "patch": patch,
        "previous_filename": previous_filename,
        "raw_url": f"https://raw.githubusercontent.com/owner/repo/main/{filename}",
        "blob_url": f"https://github.com/owner/repo/blob/main/{filename}",
    }


def _commit(
    sha: str,
    message: str,
    *,
    author: dict[str, object] | None = None,
    include_author: bool = True,
) -> dict[str, object]:
    commit_author = author
    if commit_author is None and include_author:
        commit_author = {
            "name": "Repo Tester",
            "email": "tester@example.com",
            "date": "2026-01-01T12:00:00Z",
        }
    return {
        "sha": sha,
        "html_url": f"https://github.com/owner/repo/commit/{sha}",
        "commit": {
            "message": message,
            "author": commit_author,
        },
    }
