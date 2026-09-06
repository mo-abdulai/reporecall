from datetime import UTC, datetime

import httpx
import pytest

from reporecall.github import (
    GitHubAPIError,
    GitHubAuthenticationError,
    GitHubClient,
    GitHubNetworkError,
    GitHubNotFoundError,
    GitHubRateLimitError,
    GitHubRepository,
    GitHubResponseError,
)
from reporecall.github import client as github_client_module


def test_request_has_standard_headers_without_token(monkeypatch):
    requests = _capture_requests([httpx.Response(200, json={"ok": True})])
    monkeypatch.setattr(github_client_module.settings, "github_token", None)

    client = GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport)
    client.get("/user")
    client.close()

    request = requests.seen[0]
    assert request.headers["Accept"] == "application/vnd.github+json"
    assert request.headers["User-Agent"] == "RepoRecall"
    assert request.headers["X-GitHub-Api-Version"] == "2022-11-28"
    assert "Authorization" not in request.headers


def test_request_uses_direct_token():
    requests = _capture_requests([httpx.Response(200, json={"ok": True})])

    client = GitHubClient(
        token="direct-token",
        base_url="https://api.github.test",
        transport=requests.transport,
    )
    client.get("/user")
    client.close()

    assert requests.seen[0].headers["Authorization"] == "Bearer direct-token"


def test_request_uses_token_from_settings(monkeypatch):
    requests = _capture_requests([httpx.Response(200, json={"ok": True})])
    monkeypatch.setattr(github_client_module.settings, "github_token", "settings-token")

    client = GitHubClient(base_url="https://api.github.test", transport=requests.transport)
    client.get("/user")
    client.close()

    assert requests.seen[0].headers["Authorization"] == "Bearer settings-token"


def test_successful_get_returns_decoded_json_object():
    requests = _capture_requests([httpx.Response(200, json={"login": "octocat"})])

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        data = client.get("/users/octocat")

    assert data == {"login": "octocat"}


def test_empty_successful_response_returns_none():
    requests = _capture_requests([httpx.Response(204)])

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        data = client.get("/empty")

    assert data is None


def test_get_repository_requests_expected_endpoint():
    requests = _capture_requests([httpx.Response(200, json={"full_name": "fastapi/fastapi"})])
    repository = GitHubRepository(owner="fastapi", name="fastapi")

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        data = client.get_repository(repository)

    assert data == {"full_name": "fastapi/fastapi"}
    assert requests.seen[0].url.path == "/repos/fastapi/fastapi"


def test_get_repository_rejects_non_object_response():
    requests = _capture_requests([httpx.Response(200, json=[])])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(GitHubResponseError),
    ):
        client.get_repository(GitHubRepository(owner="fastapi", name="fastapi"))


def test_pagination_follows_next_links_and_preserves_initial_params():
    responses = [
        httpx.Response(
            200,
            json=[{"number": 1}],
            headers={
                "Link": '<https://api.github.test/repos/o/r/issues?page=2>; rel="next"',
            },
        ),
        httpx.Response(
            200,
            json=[{"number": 2}],
            headers={
                "Link": '<https://api.github.test/repos/o/r/issues?page=3>; rel="next"',
            },
        ),
        httpx.Response(200, json=[{"number": 3}]),
    ]
    requests = _capture_requests(responses)

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        items = client.paginate("/repos/o/r/issues", params={"state": "all"}, per_page=50)

    assert items == [{"number": 1}, {"number": 2}, {"number": 3}]
    assert requests.seen[0].url.params["state"] == "all"
    assert requests.seen[0].url.params["per_page"] == "50"
    assert "state" not in requests.seen[1].url.params


def test_pagination_respects_max_pages():
    responses = [
        httpx.Response(
            200,
            json=[{"number": 1}],
            headers={
                "Link": '<https://api.github.test/repos/o/r/issues?page=2>; rel="next"',
            },
        ),
        httpx.Response(200, json=[{"number": 2}]),
    ]
    requests = _capture_requests(responses)

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        items = client.paginate("/repos/o/r/issues", max_pages=1)

    assert items == [{"number": 1}]
    assert len(requests.seen) == 1


def test_pagination_rejects_non_array_response():
    requests = _capture_requests([httpx.Response(200, json={"items": []})])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(GitHubResponseError),
    ):
        client.paginate("/repos/o/r/issues")


def test_pagination_rejects_repeated_next_url():
    repeated_url = "https://api.github.test/repos/o/r/issues?page=1"
    requests = _capture_requests(
        [
            httpx.Response(
                200,
                json=[],
                headers={"Link": f'<{repeated_url}>; rel="next"'},
            ),
            httpx.Response(200, json=[]),
        ]
    )

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(GitHubResponseError),
    ):
        client.paginate(repeated_url)


def test_authentication_error():
    requests = _capture_requests([httpx.Response(401, json={"message": "Bad credentials"})])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(GitHubAuthenticationError) as exc_info,
    ):
        client.get("/user")

    assert exc_info.value.status_code == 401
    assert "Bad credentials" in str(exc_info.value)


def test_not_found_error():
    requests = _capture_requests([httpx.Response(404, json={"message": "Not Found"})])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(GitHubNotFoundError) as exc_info,
    ):
        client.get("/repos/o/missing")

    assert exc_info.value.status_code == 404


def test_rate_limit_error_and_metadata():
    reset_timestamp = 1_700_000_000
    requests = _capture_requests(
        [
            httpx.Response(
                403,
                json={"message": "API rate limit exceeded"},
                headers={
                    "X-RateLimit-Limit": "60",
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Used": "60",
                    "X-RateLimit-Reset": str(reset_timestamp),
                    "X-RateLimit-Resource": "core",
                },
            )
        ]
    )

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(GitHubRateLimitError) as exc_info,
    ):
        client.get("/rate-limited")

    assert client.rate_limit is not None
    assert client.rate_limit.limit == 60
    assert client.rate_limit.remaining == 0
    assert client.rate_limit.used == 60
    assert client.rate_limit.reset_at == datetime.fromtimestamp(reset_timestamp, tz=UTC)
    assert client.rate_limit.resource == "core"
    assert exc_info.value.reset_at == datetime.fromtimestamp(reset_timestamp, tz=UTC)


def test_server_error():
    requests = _capture_requests([httpx.Response(500, json={"message": "Server Error"})])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(GitHubAPIError) as exc_info,
    ):
        client.get("/server-error")

    assert exc_info.value.status_code == 500
    assert "Server Error" in str(exc_info.value)


def test_network_failure_preserves_cause():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    with (
        GitHubClient(
            token=None,
            base_url="https://api.github.test",
            transport=httpx.MockTransport(handler),
        ) as client,
        pytest.raises(GitHubNetworkError) as exc_info,
    ):
        client.get("/timeout")

    assert isinstance(exc_info.value.__cause__, httpx.TimeoutException)


def test_invalid_json_response():
    requests = _capture_requests(
        [httpx.Response(200, content=b"not-json", headers={"Content-Type": "application/json"})]
    )

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(GitHubResponseError),
    ):
        client.get("/invalid-json")


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
