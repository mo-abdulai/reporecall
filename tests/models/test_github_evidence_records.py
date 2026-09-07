from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from reporecall.github import GitHubRepository
from reporecall.models import (
    GitHubCommitPullRequestAssociation,
    GitHubTimelineEvidenceType,
    GitHubTimelineRelationshipEvidence,
)


def test_cross_reference_evidence_requires_complete_source_identity():
    evidence = GitHubTimelineRelationshipEvidence(
        repository=_repository(),
        target_number=10,
        event_type=GitHubTimelineEvidenceType.CROSS_REFERENCED,
        created_at=_timestamp(),
        source_repository=_repository(),
        source_number=20,
        source_is_pull_request=True,
    )

    assert evidence.source_number == 20
    assert evidence.source_is_pull_request is True


def test_commit_timeline_evidence_requires_commit_identity():
    with pytest.raises(ValidationError):
        GitHubTimelineRelationshipEvidence(
            repository=_repository(),
            target_number=10,
            event_type=GitHubTimelineEvidenceType.REFERENCED,
            created_at=_timestamp(),
        )


def test_timeline_evidence_requires_timezone_aware_timestamp():
    with pytest.raises(ValidationError):
        GitHubTimelineRelationshipEvidence(
            repository=_repository(),
            target_number=10,
            event_type=GitHubTimelineEvidenceType.CROSS_REFERENCED,
            created_at=_timestamp().replace(tzinfo=None),
            source_repository=_repository(),
            source_number=20,
            source_is_pull_request=False,
        )


def test_commit_pull_request_association_normalizes_sha_whitespace():
    association = GitHubCommitPullRequestAssociation(
        repository=_repository(),
        commit_sha="  abc123  ",
        pull_request_number=20,
        pull_request_url="https://github.com/owner/repo/pull/20",
    )

    assert association.commit_sha == "abc123"


def _repository() -> GitHubRepository:
    return GitHubRepository(owner="owner", name="repo")


def _timestamp() -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
