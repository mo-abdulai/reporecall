import pytest
from pydantic import ValidationError

from reporecall.github import GitHubRepository
from reporecall.models import (
    ArtifactReference,
    ArtifactType,
    EngineeringRelationship,
    RelationshipEvidenceType,
    RelationshipType,
)


def test_artifact_reference_preserves_repository_identity():
    repository = GitHubRepository(owner="owner", name="repo")
    reference = ArtifactReference(
        artifact_type=ArtifactType.ISSUE,
        repository=repository,
        identifier="123",
    )

    assert reference.repository == repository
    assert reference.identifier == "123"


def test_relationship_identity_uses_source_type_target_and_evidence_type():
    repository = GitHubRepository(owner="owner", name="repo")
    source = ArtifactReference(
        artifact_type=ArtifactType.PULL_REQUEST,
        repository=repository,
        identifier="10",
    )
    target = ArtifactReference(
        artifact_type=ArtifactType.ISSUE,
        repository=repository,
        identifier="20",
    )
    relationship = EngineeringRelationship(
        source=source,
        target=target,
        relationship_type=RelationshipType.PULL_REQUEST_CLOSES_ISSUE,
        evidence_type=RelationshipEvidenceType.CLOSING_KEYWORD,
        evidence="Fixes #20",
        source_field="body",
    )

    assert relationship.identity_key == (
        source,
        RelationshipType.PULL_REQUEST_CLOSES_ISSUE,
        target,
        RelationshipEvidenceType.CLOSING_KEYWORD,
    )


def test_relationship_models_reject_invalid_enum_values_and_empty_identifiers():
    repository = GitHubRepository(owner="owner", name="repo")

    with pytest.raises(ValidationError):
        ArtifactReference(
            artifact_type="engineering_event",
            repository=repository,
            identifier="123",
        )

    with pytest.raises(ValidationError):
        ArtifactReference(
            artifact_type=ArtifactType.ISSUE,
            repository=repository,
            identifier="",
        )

    source = ArtifactReference(
        artifact_type=ArtifactType.PULL_REQUEST,
        repository=repository,
        identifier="10",
    )
    target = ArtifactReference(
        artifact_type=ArtifactType.ISSUE,
        repository=repository,
        identifier="20",
    )

    with pytest.raises(ValidationError):
        EngineeringRelationship(
            source=source,
            target=target,
            relationship_type="related_to",
            evidence_type=RelationshipEvidenceType.TEXT_REFERENCE,
        )
