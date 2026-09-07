from datetime import UTC, datetime

import reporecall.processing.relationship_enricher as relationship_enricher_module
from reporecall.github import GitHubRepository
from reporecall.models import (
    ArtifactType,
    GitCommit,
    GitHubBranchReference,
    GitHubCommitPullRequestAssociation,
    GitHubCommitReference,
    GitHubIssue,
    GitHubPullRequest,
    GitHubTimelineEvidenceType,
    GitHubTimelineRelationshipEvidence,
    IssueState,
    PullRequestState,
    RelationshipEvidenceType,
    RelationshipType,
)
from reporecall.processing import (
    RelationshipEnricher,
    RelationshipInput,
    RelationshipLinker,
)


def test_cross_reference_adds_pull_request_issue_relationship():
    records = RelationshipInput(
        repository=_repository(),
        issues=[_issue(10)],
        pull_requests=[_pull_request(20)],
    )

    relationships = RelationshipEnricher().enrich(
        [],
        records=records,
        timeline_evidence=[_cross_reference(source_number=20, target_number=10)],
    )

    assert _has_relationship(
        relationships,
        source_type=ArtifactType.PULL_REQUEST,
        source_id="20",
        relationship_type=RelationshipType.PULL_REQUEST_REFERENCES_ISSUE,
        target_type=ArtifactType.ISSUE,
        target_id="10",
        evidence_type=RelationshipEvidenceType.GITHUB_TIMELINE_CROSS_REFERENCE,
    )
    assert not any(
        relationship.relationship_type is RelationshipType.PULL_REQUEST_CLOSES_ISSUE
        for relationship in relationships
    )


def test_issue_source_and_pull_request_target_are_not_forced_into_unsupported_relationships():
    issue_source_records = RelationshipInput(
        repository=_repository(),
        issues=[_issue(10), _issue(20)],
    )
    pull_request_target_records = RelationshipInput(
        repository=_repository(),
        pull_requests=[_pull_request(10), _pull_request(20)],
    )
    issue_source = _cross_reference(
        source_number=20,
        target_number=10,
        source_is_pull_request=False,
    )
    pull_request_target = _cross_reference(source_number=20, target_number=10)

    enricher = RelationshipEnricher()

    assert enricher.enrich(
        [],
        records=issue_source_records,
        timeline_evidence=[issue_source],
    ) == []
    assert enricher.enrich(
        [],
        records=pull_request_target_records,
        timeline_evidence=[pull_request_target],
    ) == []


def test_timeline_commit_reference_adds_authoritative_relationship():
    records = RelationshipInput(
        repository=_repository(),
        issues=[_issue(10)],
        local_commits=[_local_commit("abc123")],
    )

    relationships = RelationshipEnricher().enrich(
        [],
        records=records,
        timeline_evidence=[_commit_evidence("referenced", "abc123", target_number=10)],
    )

    assert _has_relationship(
        relationships,
        source_type=ArtifactType.LOCAL_GIT_COMMIT,
        source_id="abc123",
        relationship_type=RelationshipType.COMMIT_REFERENCES_ISSUE,
        target_type=ArtifactType.ISSUE,
        target_id="10",
        evidence_type=RelationshipEvidenceType.GITHUB_TIMELINE_COMMIT_REFERENCE,
    )


def test_timeline_commit_reference_links_each_exactly_matching_commit_artifact():
    records = RelationshipInput(
        repository=_repository(),
        issues=[_issue(10)],
        pull_requests=[_pull_request(20)],
        pull_request_commits={20: [_commit_reference("abc123")]},
        local_commits=[_local_commit("abc123"), _local_commit("abc1234")],
    )

    relationships = RelationshipEnricher().enrich(
        [],
        records=records,
        timeline_evidence=[_commit_evidence("referenced", "abc123", target_number=10)],
    )

    sources = {
        (relationship.source.artifact_type, relationship.source.identifier)
        for relationship in relationships
    }
    assert sources == {
        (ArtifactType.GITHUB_COMMIT_REFERENCE, "abc123"),
        (ArtifactType.LOCAL_GIT_COMMIT, "abc123"),
    }


def test_closed_event_and_commit_association_add_only_direct_relationships():
    records = RelationshipInput(
        repository=_repository(),
        issues=[_issue(10)],
        pull_requests=[_pull_request(20)],
        local_commits=[_local_commit("abc123")],
    )

    relationships = RelationshipEnricher().enrich(
        [],
        records=records,
        timeline_evidence=[_commit_evidence("closed", "abc123", target_number=10)],
        commit_pr_associations=[_association("abc123", pull_request_number=20)],
    )

    assert _has_relationship(
        relationships,
        source_type=ArtifactType.LOCAL_GIT_COMMIT,
        source_id="abc123",
        relationship_type=RelationshipType.COMMIT_CLOSES_ISSUE,
        target_type=ArtifactType.ISSUE,
        target_id="10",
        evidence_type=RelationshipEvidenceType.GITHUB_TIMELINE_CLOSED_EVENT,
    )
    assert _has_relationship(
        relationships,
        source_type=ArtifactType.PULL_REQUEST,
        source_id="20",
        relationship_type=RelationshipType.PULL_REQUEST_HAS_COMMIT,
        target_type=ArtifactType.LOCAL_GIT_COMMIT,
        target_id="abc123",
        evidence_type=RelationshipEvidenceType.GITHUB_COMMIT_ASSOCIATED_PULL_REQUEST,
    )
    assert not any(
        relationship.relationship_type is RelationshipType.PULL_REQUEST_CLOSES_ISSUE
        for relationship in relationships
    )


def test_existing_relationships_and_multiple_provenance_types_are_preserved():
    records = RelationshipInput(
        repository=_repository(),
        issues=[_issue(10)],
        pull_requests=[_pull_request(20, body="Fixes #10")],
        local_commits=[_local_commit("abc123", message="See #10")],
    )
    existing = RelationshipLinker().link(records)

    relationships = RelationshipEnricher().enrich(
        existing,
        records=records,
        timeline_evidence=[
            _cross_reference(source_number=20, target_number=10),
            _commit_evidence("referenced", "abc123", target_number=10),
        ],
    )

    assert all(relationship in relationships for relationship in existing)
    commit_reference_evidence = {
        relationship.evidence_type
        for relationship in relationships
        if relationship.relationship_type is RelationshipType.COMMIT_REFERENCES_ISSUE
    }
    assert commit_reference_evidence == {
        RelationshipEvidenceType.TEXT_REFERENCE,
        RelationshipEvidenceType.GITHUB_TIMELINE_COMMIT_REFERENCE,
    }
    assert _has_relationship(
        relationships,
        source_type=ArtifactType.PULL_REQUEST,
        source_id="20",
        relationship_type=RelationshipType.PULL_REQUEST_CLOSES_ISSUE,
        target_type=ArtifactType.ISSUE,
        target_id="10",
        evidence_type=RelationshipEvidenceType.CLOSING_KEYWORD,
    )


def test_duplicate_evidence_is_deduplicated_and_output_is_stable():
    records = RelationshipInput(
        repository=_repository(),
        issues=[_issue(10)],
        pull_requests=[_pull_request(20)],
        local_commits=[_local_commit("abc123")],
    )
    timeline = _cross_reference(source_number=20, target_number=10)
    association = _association("abc123", pull_request_number=20)
    enricher = RelationshipEnricher()

    first = enricher.enrich(
        [],
        records=records,
        timeline_evidence=[timeline, timeline],
        commit_pr_associations=[association, association],
    )
    second = enricher.enrich(
        [],
        records=records,
        timeline_evidence=[timeline, timeline],
        commit_pr_associations=[association, association],
    )

    assert first == second
    assert len(first) == 2
    assert len({relationship.identity_key for relationship in first}) == 2


def test_missing_artifacts_and_non_exact_sha_do_not_create_relationships():
    records = RelationshipInput(
        repository=_repository(),
        issues=[_issue(10)],
        local_commits=[_local_commit("abc1234")],
    )

    relationships = RelationshipEnricher().enrich(
        [],
        records=records,
        timeline_evidence=[
            _cross_reference(source_number=999, target_number=10),
            _commit_evidence("referenced", "abc123", target_number=10),
            _commit_evidence("referenced", "abc1234", target_number=999),
        ],
        commit_pr_associations=[_association("abc1234", pull_request_number=999)],
    )

    assert relationships == []


def test_relationship_enricher_has_no_network_or_git_dependencies():
    module_attributes = vars(relationship_enricher_module)

    assert "GitHubClient" not in module_attributes
    assert "httpx" not in module_attributes
    assert "git" not in module_attributes


def _has_relationship(
    relationships,
    *,
    source_type: ArtifactType,
    source_id: str,
    relationship_type: RelationshipType,
    target_type: ArtifactType,
    target_id: str,
    evidence_type: RelationshipEvidenceType,
) -> bool:
    return any(
        relationship.source.artifact_type is source_type
        and relationship.source.identifier == source_id
        and relationship.relationship_type is relationship_type
        and relationship.target.artifact_type is target_type
        and relationship.target.identifier == target_id
        and relationship.evidence_type is evidence_type
        for relationship in relationships
    )


def _repository() -> GitHubRepository:
    return GitHubRepository(owner="owner", name="repo")


def _issue(number: int) -> GitHubIssue:
    return GitHubIssue(
        repository=_repository(),
        number=number,
        title=f"Issue {number}",
        body=None,
        state=IssueState.OPEN,
        author=None,
        labels=[],
        created_at=_timestamp(),
        updated_at=_timestamp(),
        closed_at=None,
        html_url=f"https://github.com/owner/repo/issues/{number}",
        comments_count=0,
        locked=False,
    )


def _pull_request(number: int, *, body: str | None = None) -> GitHubPullRequest:
    repository = _repository()
    return GitHubPullRequest(
        repository=repository,
        number=number,
        title=f"PR {number}",
        body=body,
        state=PullRequestState.OPEN,
        author=None,
        labels=[],
        draft=False,
        locked=False,
        created_at=_timestamp(),
        updated_at=_timestamp(),
        closed_at=None,
        merged_at=None,
        html_url=f"https://github.com/owner/repo/pull/{number}",
        head=GitHubBranchReference(
            repository=repository,
            ref="feature",
            sha="headsha",
        ),
        base=GitHubBranchReference(
            repository=repository,
            ref="main",
            sha="basesha",
        ),
        merge_commit_sha=None,
    )


def _commit_reference(sha: str) -> GitHubCommitReference:
    return GitHubCommitReference(
        sha=sha,
        html_url=f"https://github.com/owner/repo/commit/{sha}",
        message="Commit",
        author_name=None,
        author_email=None,
        authored_at=None,
    )


def _local_commit(sha: str, *, message: str = "Commit") -> GitCommit:
    return GitCommit(
        sha=sha,
        message=message,
        author_name="Repo Tester",
        author_email="tester@example.com",
        authored_at=_timestamp(),
        committed_at=_timestamp(),
        parent_shas=[],
        changed_files=[],
    )


def _cross_reference(
    *,
    source_number: int,
    target_number: int,
    source_is_pull_request: bool = True,
) -> GitHubTimelineRelationshipEvidence:
    return GitHubTimelineRelationshipEvidence(
        repository=_repository(),
        target_number=target_number,
        event_type=GitHubTimelineEvidenceType.CROSS_REFERENCED,
        created_at=_timestamp(),
        source_repository=_repository(),
        source_number=source_number,
        source_is_pull_request=source_is_pull_request,
    )


def _commit_evidence(
    event: str,
    sha: str,
    *,
    target_number: int,
) -> GitHubTimelineRelationshipEvidence:
    return GitHubTimelineRelationshipEvidence(
        repository=_repository(),
        target_number=target_number,
        event_type=GitHubTimelineEvidenceType(event),
        created_at=_timestamp(),
        commit_repository=_repository(),
        commit_sha=sha,
        commit_url=f"https://api.github.com/repos/owner/repo/commits/{sha}",
    )


def _association(
    sha: str,
    *,
    pull_request_number: int,
) -> GitHubCommitPullRequestAssociation:
    return GitHubCommitPullRequestAssociation(
        repository=_repository(),
        commit_sha=sha,
        pull_request_number=pull_request_number,
        pull_request_url=f"https://github.com/owner/repo/pull/{pull_request_number}",
    )


def _timestamp() -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
