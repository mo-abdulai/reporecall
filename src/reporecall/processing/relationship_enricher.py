from collections.abc import Sequence
from dataclasses import dataclass

from reporecall.models import (
    ArtifactReference,
    ArtifactType,
    EngineeringRelationship,
    GitHubCommitPullRequestAssociation,
    GitHubTimelineEvidenceType,
    GitHubTimelineRelationshipEvidence,
    RelationshipEvidenceType,
    RelationshipType,
)
from reporecall.processing.relationship_linker import RelationshipInput

type RelationshipIdentity = tuple[
    ArtifactReference,
    RelationshipType,
    ArtifactReference,
    RelationshipEvidenceType,
]


class RelationshipEnricher:
    """Add direct GitHub-authoritative evidence without performing I/O."""

    def enrich(
        self,
        relationships: Sequence[EngineeringRelationship],
        *,
        records: RelationshipInput,
        timeline_evidence: Sequence[GitHubTimelineRelationshipEvidence] = (),
        commit_pr_associations: Sequence[GitHubCommitPullRequestAssociation] = (),
    ) -> list[EngineeringRelationship]:
        """Return stable relationships enriched only for artifacts in records."""

        relationships_by_key = {
            relationship.identity_key: relationship for relationship in relationships
        }
        artifacts = _AvailableArtifacts.from_records(records)

        for evidence in timeline_evidence:
            self._add_timeline_relationship(relationships_by_key, artifacts, evidence)
        for association in commit_pr_associations:
            self._add_commit_pull_request_relationship(
                relationships_by_key,
                artifacts,
                association,
            )

        return sorted(relationships_by_key.values(), key=_relationship_sort_key)

    def _add_timeline_relationship(
        self,
        relationships_by_key: dict[RelationshipIdentity, EngineeringRelationship],
        artifacts: "_AvailableArtifacts",
        evidence: GitHubTimelineRelationshipEvidence,
    ) -> None:
        target_key = (evidence.repository.owner, evidence.repository.name, evidence.target_number)
        target = artifacts.issues.get(target_key)
        if target is None or target_key in artifacts.pull_requests:
            return

        if evidence.event_type is GitHubTimelineEvidenceType.CROSS_REFERENCED:
            self._add_cross_reference(relationships_by_key, artifacts, target, evidence)
            return

        if evidence.commit_repository is None or evidence.commit_sha is None:
            return
        commit_references = artifacts.commits.get(
            (
                evidence.commit_repository.owner,
                evidence.commit_repository.name,
                evidence.commit_sha,
            ),
            (),
        )
        relationship_type = (
            RelationshipType.COMMIT_CLOSES_ISSUE
            if evidence.event_type is GitHubTimelineEvidenceType.CLOSED
            else RelationshipType.COMMIT_REFERENCES_ISSUE
        )
        evidence_type = (
            RelationshipEvidenceType.GITHUB_TIMELINE_CLOSED_EVENT
            if evidence.event_type is GitHubTimelineEvidenceType.CLOSED
            else RelationshipEvidenceType.GITHUB_TIMELINE_COMMIT_REFERENCE
        )
        action = "closed" if evidence.event_type is GitHubTimelineEvidenceType.CLOSED else "referenced"
        for commit_reference in commit_references:
            _add_relationship(
                relationships_by_key,
                source=commit_reference,
                target=target,
                relationship_type=relationship_type,
                evidence_type=evidence_type,
                evidence=(
                    f"GitHub timeline recorded commit {evidence.commit_sha} "
                    f"{action} issue #{evidence.target_number} at "
                    f"{evidence.created_at.isoformat()}"
                ),
            )

    def _add_cross_reference(
        self,
        relationships_by_key: dict[RelationshipIdentity, EngineeringRelationship],
        artifacts: "_AvailableArtifacts",
        target: ArtifactReference,
        evidence: GitHubTimelineRelationshipEvidence,
    ) -> None:
        if not evidence.source_is_pull_request:
            return
        if evidence.source_repository is None or evidence.source_number is None:
            return

        source = artifacts.pull_requests.get(
            (
                evidence.source_repository.owner,
                evidence.source_repository.name,
                evidence.source_number,
            )
        )
        if source is None:
            return

        _add_relationship(
            relationships_by_key,
            source=source,
            target=target,
            relationship_type=RelationshipType.PULL_REQUEST_REFERENCES_ISSUE,
            evidence_type=RelationshipEvidenceType.GITHUB_TIMELINE_CROSS_REFERENCE,
            evidence=(
                f"GitHub timeline recorded pull request #{evidence.source_number} "
                f"cross-referencing issue #{evidence.target_number} at "
                f"{evidence.created_at.isoformat()}"
            ),
        )

    def _add_commit_pull_request_relationship(
        self,
        relationships_by_key: dict[RelationshipIdentity, EngineeringRelationship],
        artifacts: "_AvailableArtifacts",
        association: GitHubCommitPullRequestAssociation,
    ) -> None:
        repository_key = (association.repository.owner, association.repository.name)
        pull_request = artifacts.pull_requests.get(
            (*repository_key, association.pull_request_number)
        )
        if pull_request is None:
            return

        for commit_reference in artifacts.commits.get(
            (*repository_key, association.commit_sha),
            (),
        ):
            _add_relationship(
                relationships_by_key,
                source=pull_request,
                target=commit_reference,
                relationship_type=RelationshipType.PULL_REQUEST_HAS_COMMIT,
                evidence_type=RelationshipEvidenceType.GITHUB_COMMIT_ASSOCIATED_PULL_REQUEST,
                evidence=(
                    f"GitHub associated commit {association.commit_sha} with pull request "
                    f"#{association.pull_request_number}"
                ),
            )


@dataclass(frozen=True)
class _AvailableArtifacts:
    issues: dict[tuple[str, str, int], ArtifactReference]
    pull_requests: dict[tuple[str, str, int], ArtifactReference]
    commits: dict[tuple[str, str, str], tuple[ArtifactReference, ...]]

    @classmethod
    def from_records(cls, records: RelationshipInput) -> "_AvailableArtifacts":
        issues = {
            (issue.repository.owner, issue.repository.name, issue.number): ArtifactReference(
                artifact_type=ArtifactType.ISSUE,
                repository=issue.repository,
                identifier=str(issue.number),
            )
            for issue in records.issues
        }
        pull_requests = {
            (
                pull_request.repository.owner,
                pull_request.repository.name,
                pull_request.number,
            ): ArtifactReference(
                artifact_type=ArtifactType.PULL_REQUEST,
                repository=pull_request.repository,
                identifier=str(pull_request.number),
            )
            for pull_request in records.pull_requests
        }

        commit_references: dict[tuple[str, str, str], set[ArtifactReference]] = {}
        repository_key = (records.repository.owner, records.repository.name)
        for github_commits in records.pull_request_commits.values():
            for github_commit in github_commits:
                key = (*repository_key, github_commit.sha)
                commit_references.setdefault(key, set()).add(
                    ArtifactReference(
                        artifact_type=ArtifactType.GITHUB_COMMIT_REFERENCE,
                        repository=records.repository,
                        identifier=github_commit.sha,
                    )
                )
        for local_commit in records.local_commits:
            key = (*repository_key, local_commit.sha)
            commit_references.setdefault(key, set()).add(
                ArtifactReference(
                    artifact_type=ArtifactType.LOCAL_GIT_COMMIT,
                    repository=records.repository,
                    identifier=local_commit.sha,
                )
            )

        commits_by_key = {
            key: tuple(
                sorted(
                    references,
                    key=lambda reference: (
                        reference.artifact_type.value,
                        reference.identifier,
                    ),
                )
            )
            for key, references in commit_references.items()
        }
        return cls(issues=issues, pull_requests=pull_requests, commits=commits_by_key)


def _add_relationship(
    relationships_by_key: dict[RelationshipIdentity, EngineeringRelationship],
    *,
    source: ArtifactReference,
    target: ArtifactReference,
    relationship_type: RelationshipType,
    evidence_type: RelationshipEvidenceType,
    evidence: str,
) -> None:
    relationship = EngineeringRelationship(
        source=source,
        target=target,
        relationship_type=relationship_type,
        evidence_type=evidence_type,
        evidence=evidence,
    )
    relationships_by_key.setdefault(relationship.identity_key, relationship)


def _relationship_sort_key(relationship: EngineeringRelationship) -> tuple[str, ...]:
    return (
        relationship.source.repository.owner,
        relationship.source.repository.name,
        relationship.source.artifact_type.value,
        relationship.source.identifier,
        relationship.relationship_type.value,
        relationship.target.repository.owner,
        relationship.target.repository.name,
        relationship.target.artifact_type.value,
        relationship.target.identifier,
        relationship.evidence_type.value,
        relationship.evidence or "",
        relationship.source_field or "",
    )
