from unittest.mock import Mock, call

import httpx
import pytest

from reporecall.github import GitHubClient, GitHubRepository, GitHubResponseError
from reporecall.ingestion import (
    GitHubIssueLoader,
    GitHubPullRequestLoader,
    GitHubRelationshipEvidenceLoader,
    InvalidIssueNumberError,
)
from reporecall.models import GitHubTimelineEvidenceType


def test_timeline_loader_filters_events_and_follows_pagination():
    responses = [
        httpx.Response(
            200,
            json=[
                _cross_reference(20, is_pull_request=True),
                {"event": "labeled", "label": {"name": "bug"}},
                _commit_event("referenced", "abc123"),
            ],
            headers={
                "Link": '<https://api.github.test/repos/owner/repo/issues/10/timeline?page=2>; rel="next"'
            },
        ),
        httpx.Response(
            200,
            json=[
                _commit_event("closed", "def456"),
                _commit_event("closed", None),
                {"event": "assigned", "assignee": {"login": "octocat"}},
            ],
        ),
    ]
    requests = _capture_requests(responses)

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        evidence = GitHubRelationshipEvidenceLoader(
            client,
            _repository(),
        ).get_timeline_relationship_evidence(10)

    assert [item.event_type for item in evidence] == [
        GitHubTimelineEvidenceType.CROSS_REFERENCED,
        GitHubTimelineEvidenceType.REFERENCED,
        GitHubTimelineEvidenceType.CLOSED,
    ]
    assert evidence[0].source_repository == _repository()
    assert evidence[0].source_number == 20
    assert evidence[0].source_is_pull_request is True
    assert evidence[0].actor is not None
    assert evidence[0].actor.login == "octocat"
    assert evidence[1].commit_sha == "abc123"
    assert evidence[1].commit_repository == _repository()
    assert evidence[2].commit_sha == "def456"
    assert evidence[2].commit_repository == _repository()
    assert [request.url.path for request in requests.seen] == [
        "/repos/owner/repo/issues/10/timeline",
        "/repos/owner/repo/issues/10/timeline",
    ]


def test_cross_reference_detects_normal_issue_structurally():
    requests = _capture_requests(
        [httpx.Response(200, json=[_cross_reference(30, is_pull_request=False)])]
    )

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        evidence = GitHubRelationshipEvidenceLoader(
            client,
            _repository(),
        ).get_timeline_relationship_evidence(10)

    assert evidence[0].source_number == 30
    assert evidence[0].source_is_pull_request is False


def test_cross_reference_preserves_cross_repository_source():
    requests = _capture_requests(
        [
            httpx.Response(
                200,
                json=[
                    _cross_reference(
                        20,
                        is_pull_request=True,
                        source_owner="other",
                        source_repository="project",
                    )
                ],
            )
        ]
    )

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        evidence = GitHubRelationshipEvidenceLoader(
            client,
            _repository(),
        ).get_timeline_relationship_evidence(10)

    assert evidence[0].source_repository == GitHubRepository(owner="other", name="project")


def test_closed_event_without_commit_is_not_relationship_evidence():
    requests = _capture_requests([httpx.Response(200, json=[_commit_event("closed", None)])])

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        evidence = GitHubRelationshipEvidenceLoader(
            client,
            _repository(),
        ).get_timeline_relationship_evidence(10)

    assert evidence == []


def test_unsupported_timeline_events_are_skipped_without_parsing():
    requests = _capture_requests(
        [
            httpx.Response(
                200,
                json=[
                    {"event": "labeled"},
                    {"event": "assigned"},
                    {"event": "renamed"},
                    {"event": "commented"},
                ],
            )
        ]
    )

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        evidence = GitHubRelationshipEvidenceLoader(
            client,
            _repository(),
        ).get_timeline_relationship_evidence(10)

    assert evidence == []


@pytest.mark.parametrize(
    "payload",
    [
        {"event": "cross-referenced", "created_at": "2026-01-01T12:00:00Z"},
        {
            "event": "referenced",
            "created_at": "2026-01-01T12:00:00Z",
            "commit_url": "https://api.github.test/repos/owner/repo/commits/abc123",
        },
        {
            "event": "cross-referenced",
            "created_at": "not-a-date",
            "actor": None,
            "source": {
                "type": "issue",
                "issue": {
                    "number": 20,
                    "repository_url": "https://api.github.test/repos/owner/repo",
                    "pull_request": {
                        "url": "https://api.github.test/repos/owner/repo/pulls/20"
                    },
                },
            },
        },
    ],
)
def test_malformed_supported_timeline_event_fails_clearly(payload: dict[str, object]):
    requests = _capture_requests([httpx.Response(200, json=[payload])])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(GitHubResponseError),
    ):
        GitHubRelationshipEvidenceLoader(
            client,
            _repository(),
        ).get_timeline_relationship_evidence(10)


def test_commit_pull_request_associations_are_normalized():
    requests = _capture_requests(
        [
            httpx.Response(
                200,
                json=[
                    {"number": 20, "html_url": "https://github.com/owner/repo/pull/20"},
                ],
                headers={
                    "Link": '<https://api.github.test/repos/owner/repo/commits/abc123/pulls?page=2>; rel="next"'
                },
            ),
            httpx.Response(
                200,
                json=[{"number": 30, "html_url": "https://github.com/owner/repo/pull/30"}],
            ),
        ]
    )

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        associations = GitHubRelationshipEvidenceLoader(
            client,
            _repository(),
        ).get_pull_requests_for_commit("abc123")

    assert [association.pull_request_number for association in associations] == [20, 30]
    assert all(association.repository == _repository() for association in associations)
    assert all(association.commit_sha == "abc123" for association in associations)
    assert requests.seen[0].url.path == "/repos/owner/repo/commits/abc123/pulls"
    assert len(requests.seen) == 2


def test_empty_commit_pull_request_associations_are_valid():
    requests = _capture_requests([httpx.Response(200, json=[])])

    with GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client:
        associations = GitHubRelationshipEvidenceLoader(
            client,
            _repository(),
        ).get_pull_requests_for_commit("abc123")

    assert associations == []


def test_standard_listing_loaders_do_not_fetch_relationship_evidence():
    client = Mock(spec=GitHubClient)
    client.paginate.return_value = []

    GitHubIssueLoader(client, _repository()).get_issues()
    pull_requests = GitHubPullRequestLoader(client, _repository())
    pull_requests.get_pull_requests()
    pull_requests.get_pull_request_commits(20)

    assert client.paginate.call_args_list == [
        call("/repos/owner/repo/issues", params={"state": "all"}),
        call("/repos/owner/repo/pulls", params={"state": "all"}),
        call("/repos/owner/repo/pulls/20/commits"),
    ]


@pytest.mark.parametrize("number", [0, -1])
def test_timeline_loader_rejects_invalid_number_without_request(number: int):
    requests = _capture_requests([])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(InvalidIssueNumberError),
    ):
        GitHubRelationshipEvidenceLoader(
            client,
            _repository(),
        ).get_timeline_relationship_evidence(number)

    assert requests.seen == []


@pytest.mark.parametrize("commit_sha", ["", "   "])
def test_commit_association_loader_rejects_empty_sha_without_request(commit_sha: str):
    requests = _capture_requests([])

    with (
        GitHubClient(token=None, base_url="https://api.github.test", transport=requests.transport) as client,
        pytest.raises(ValueError, match="Commit SHA"),
    ):
        GitHubRelationshipEvidenceLoader(
            client,
            _repository(),
        ).get_pull_requests_for_commit(commit_sha)

    assert requests.seen == []


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


def _cross_reference(
    source_number: int,
    *,
    is_pull_request: bool,
    source_owner: str = "owner",
    source_repository: str = "repo",
    created_at: str = "2026-01-01T12:00:00Z",
) -> dict[str, object]:
    source_issue: dict[str, object] = {
        "number": source_number,
        "repository_url": f"https://api.github.test/repos/{source_owner}/{source_repository}",
    }
    if is_pull_request:
        source_issue["pull_request"] = {
            "url": f"https://api.github.test/repos/{source_owner}/{source_repository}/pulls/{source_number}"
        }
    return {
        "event": "cross-referenced",
        "created_at": created_at,
        "actor": {
            "login": "octocat",
            "id": 1,
            "html_url": "https://github.com/octocat",
        },
        "source": {"type": "issue", "issue": source_issue},
    }


def _commit_event(event: str, commit_sha: str | None) -> dict[str, object]:
    return {
        "event": event,
        "created_at": "2026-01-02T12:00:00Z",
        "actor": None,
        "commit_id": commit_sha,
        "commit_url": (
            f"https://api.github.test/repos/owner/repo/commits/{commit_sha}"
            if commit_sha is not None
            else None
        ),
        "commit_repository": (
            {"full_name": "owner/repo"} if event == "referenced" else None
        ),
    }
