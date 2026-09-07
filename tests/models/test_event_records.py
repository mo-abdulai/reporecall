from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from reporecall.github import GitHubRepository
from reporecall.models import (
    ArtifactReference,
    ArtifactType,
    ChangedFile,
    EngineeringEvent,
    FileChangeType,
    GitCommit,
    GitHubCommitReference,
    GitHubPullRequestFile,
    GitHubPullRequestFileStatus,
)


def test_engineering_event_exposes_deterministic_structural_properties():
    event = EngineeringEvent(
        event_id="github.com__owner__repo__local_git_commit__abc123",
        repository=_repository(),
        anchor=_reference(ArtifactType.LOCAL_GIT_COMMIT, "abc123"),
        pull_request_files=[_pull_request_file("src/api.py")],
        github_commit_references=[_github_commit("abc123")],
        local_commits=[
            _local_commit(
                "abc123",
                changed_files=[_changed_file("tests/test_api.py")],
            )
        ],
    )

    assert event.issue_numbers == []
    assert event.pull_request_numbers == []
    assert event.commit_shas == ["abc123"]
    assert event.changed_paths == ["src/api.py", "tests/test_api.py"]


def test_engineering_event_rejects_secondary_anchor():
    with pytest.raises(ValidationError, match="anchors must be core artifacts"):
        EngineeringEvent(
            event_id="invalid",
            repository=_repository(),
            anchor=_reference(ArtifactType.ISSUE_COMMENT, "issue-comment:1"),
        )


def test_engineering_event_rejects_anchor_from_another_repository():
    with pytest.raises(ValidationError, match="anchor must belong"):
        EngineeringEvent(
            event_id="invalid",
            repository=_repository(),
            anchor=ArtifactReference(
                artifact_type=ArtifactType.ISSUE,
                repository=GitHubRepository(owner="other", name="repo"),
                identifier="10",
            ),
        )


def _reference(artifact_type: ArtifactType, identifier: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=artifact_type,
        repository=_repository(),
        identifier=identifier,
    )


def _repository() -> GitHubRepository:
    return GitHubRepository(owner="owner", name="repo")


def _github_commit(sha: str) -> GitHubCommitReference:
    return GitHubCommitReference(
        sha=sha,
        html_url=None,
        message="Commit",
        author_name=None,
        author_email=None,
        authored_at=None,
    )


def _local_commit(sha: str, *, changed_files: list[ChangedFile]) -> GitCommit:
    return GitCommit(
        sha=sha,
        message="Commit",
        author_name="Repo Tester",
        author_email=None,
        authored_at=_timestamp(),
        committed_at=_timestamp(),
        parent_shas=[],
        changed_files=changed_files,
    )


def _pull_request_file(path: str) -> GitHubPullRequestFile:
    return GitHubPullRequestFile(
        filename=path,
        status=GitHubPullRequestFileStatus.MODIFIED,
        additions=1,
        deletions=1,
        changes=2,
    )


def _changed_file(path: str) -> ChangedFile:
    return ChangedFile(
        path=path,
        change_type=FileChangeType.MODIFIED,
        additions=1,
        deletions=1,
        patch="@@ -1 +1 @@",
    )


def _timestamp() -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
