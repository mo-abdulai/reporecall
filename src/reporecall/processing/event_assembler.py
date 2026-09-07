from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import cast
from urllib.parse import quote

from pydantic import BaseModel, Field

from reporecall.models import (
    ArtifactReference,
    ArtifactType,
    ChangedFile,
    EngineeringEvent,
    EngineeringRelationship,
    GitCommit,
    GitHubCommitReference,
    GitHubIssue,
    GitHubIssueComment,
    GitHubPullRequest,
    GitHubPullRequestFile,
    GitHubPullRequestReview,
    GitHubPullRequestReviewComment,
    RelationshipEvidenceType,
    RelationshipType,
)
from reporecall.processing.relationship_linker import RelationshipInput

EVENT_FORMING_RELATIONSHIP_TYPES = frozenset(
    {
        RelationshipType.ISSUE_HAS_COMMENT,
        RelationshipType.PULL_REQUEST_HAS_COMMENT,
        RelationshipType.PULL_REQUEST_HAS_REVIEW,
        RelationshipType.REVIEW_HAS_COMMENT,
        RelationshipType.PULL_REQUEST_HAS_REVIEW_COMMENT,
        RelationshipType.PULL_REQUEST_HAS_COMMIT,
        RelationshipType.GITHUB_COMMIT_MATCHES_LOCAL_COMMIT,
        RelationshipType.PULL_REQUEST_CHANGES_FILE,
        RelationshipType.LOCAL_COMMIT_CHANGES_FILE,
        RelationshipType.PULL_REQUEST_CLOSES_ISSUE,
        RelationshipType.COMMIT_CLOSES_ISSUE,
    }
)

_ANCHOR_PRIORITY = {
    ArtifactType.PULL_REQUEST: 0,
    ArtifactType.ISSUE: 1,
    ArtifactType.LOCAL_GIT_COMMIT: 2,
    ArtifactType.GITHUB_COMMIT_REFERENCE: 3,
}

_ARTIFACT_TYPE_ORDER = {
    artifact_type: index for index, artifact_type in enumerate(ArtifactType)
}

type EventRecord = (
    GitHubIssue
    | GitHubIssueComment
    | GitHubPullRequest
    | GitHubPullRequestReview
    | GitHubPullRequestReviewComment
    | GitHubPullRequestFile
    | GitHubCommitReference
    | GitCommit
    | ChangedFile
)
type RelationshipIdentity = tuple[
    ArtifactReference,
    RelationshipType,
    ArtifactReference,
    RelationshipEvidenceType,
]


class EngineeringEventInput(RelationshipInput):
    """Normalized records and direct relationships supplied for event assembly."""

    relationships: list[EngineeringRelationship] = Field(default_factory=list)


class EngineeringEventAssembler:
    """Assemble repository-local events from normalized records and direct edges."""

    def assemble(self, data: EngineeringEventInput) -> list[EngineeringEvent]:
        """Return deterministically ordered structural engineering events."""

        artifact_index = _build_artifact_index(data)
        relationships = _deduplicate_relationships(data.relationships)
        graph = _build_membership_graph(artifact_index, relationships)
        core_artifacts = {
            reference
            for reference in artifact_index
            if reference.artifact_type in _ANCHOR_PRIORITY
        }

        visited: set[ArtifactReference] = set()
        events: list[EngineeringEvent] = []
        for core_artifact in sorted(core_artifacts, key=_anchor_sort_key):
            if core_artifact in visited:
                continue
            component = _connected_component(core_artifact, graph)
            visited.update(component)
            events.append(
                _build_event(
                    component,
                    core_artifacts=core_artifacts,
                    artifact_index=artifact_index,
                    relationships=relationships,
                )
            )

        return sorted(events, key=_event_sort_key)


@dataclass(frozen=True)
class _ResolvedArtifact:
    reference: ArtifactReference
    record: EventRecord


def _build_artifact_index(
    data: EngineeringEventInput,
) -> dict[ArtifactReference, _ResolvedArtifact]:
    index: dict[ArtifactReference, _ResolvedArtifact] = {}

    for issue in data.issues:
        _add_artifact(index, _issue_ref(issue), issue)
    for comment in data.issue_comments:
        _add_artifact(index, _issue_comment_ref(comment), comment)
    for pull_request in data.pull_requests:
        _add_artifact(index, _pull_request_ref(pull_request), pull_request)
    for comment in data.pull_request_comments:
        _add_artifact(index, _pull_request_comment_ref(comment), comment)
    for review in data.pull_request_reviews:
        _add_artifact(index, _review_ref(review), review)
    for review_comment in data.pull_request_review_comments:
        _add_artifact(index, _review_comment_ref(review_comment), review_comment)

    for pull_request_number, files in data.pull_request_files.items():
        for pull_request_file in files:
            _add_artifact(
                index,
                _pull_request_file_ref(data, pull_request_number, pull_request_file),
                pull_request_file,
            )
    for github_commits in data.pull_request_commits.values():
        for github_commit in github_commits:
            _add_artifact(index, _github_commit_ref(data, github_commit), github_commit)
    for local_commit in data.local_commits:
        _add_artifact(index, _local_commit_ref(data, local_commit), local_commit)
        for changed_file in local_commit.changed_files:
            _add_artifact(
                index,
                _local_changed_file_ref(data, local_commit, changed_file),
                changed_file,
            )

    return index


def _add_artifact(
    index: dict[ArtifactReference, _ResolvedArtifact],
    reference: ArtifactReference,
    record: EventRecord,
) -> None:
    candidate = _ResolvedArtifact(reference=reference, record=record)
    existing = index.get(reference)
    if existing is None or _record_sort_key(candidate.record) < _record_sort_key(existing.record):
        index[reference] = candidate


def _deduplicate_relationships(
    relationships: Iterable[EngineeringRelationship],
) -> list[EngineeringRelationship]:
    relationships_by_key: dict[RelationshipIdentity, EngineeringRelationship] = {}
    for relationship in sorted(relationships, key=_relationship_sort_key):
        relationships_by_key.setdefault(relationship.identity_key, relationship)
    return sorted(relationships_by_key.values(), key=_relationship_sort_key)


def _build_membership_graph(
    artifact_index: dict[ArtifactReference, _ResolvedArtifact],
    relationships: Iterable[EngineeringRelationship],
) -> dict[ArtifactReference, set[ArtifactReference]]:
    graph: dict[ArtifactReference, set[ArtifactReference]] = {
        reference: set() for reference in artifact_index
    }
    for relationship in relationships:
        if not _is_event_forming(relationship):
            continue
        if relationship.source not in artifact_index or relationship.target not in artifact_index:
            continue
        graph[relationship.source].add(relationship.target)
        graph[relationship.target].add(relationship.source)
    return graph


def _connected_component(
    start: ArtifactReference,
    graph: dict[ArtifactReference, set[ArtifactReference]],
) -> set[ArtifactReference]:
    component: set[ArtifactReference] = set()
    pending = [start]
    while pending:
        reference = pending.pop()
        if reference in component:
            continue
        component.add(reference)
        neighbors = sorted(graph[reference], key=_artifact_sort_key, reverse=True)
        pending.extend(neighbors)
    return component


def _build_event(
    component: set[ArtifactReference],
    *,
    core_artifacts: set[ArtifactReference],
    artifact_index: dict[ArtifactReference, _ResolvedArtifact],
    relationships: list[EngineeringRelationship],
) -> EngineeringEvent:
    anchor = min(component & core_artifacts, key=_anchor_sort_key)
    resolved = [artifact_index[reference] for reference in component]

    issues = sorted(
        (
            cast(GitHubIssue, item.record)
            for item in resolved
            if item.reference.artifact_type is ArtifactType.ISSUE
        ),
        key=lambda issue: issue.number,
    )
    issue_comments = sorted(
        (
            cast(GitHubIssueComment, item.record)
            for item in resolved
            if item.reference.artifact_type is ArtifactType.ISSUE_COMMENT
        ),
        key=lambda comment: (comment.created_at, comment.id),
    )
    pull_requests = sorted(
        (
            cast(GitHubPullRequest, item.record)
            for item in resolved
            if item.reference.artifact_type is ArtifactType.PULL_REQUEST
        ),
        key=lambda pull_request: pull_request.number,
    )
    pull_request_comments = sorted(
        (
            cast(GitHubIssueComment, item.record)
            for item in resolved
            if item.reference.artifact_type is ArtifactType.PULL_REQUEST_COMMENT
        ),
        key=lambda comment: (comment.created_at, comment.id),
    )
    reviews = sorted(
        (
            cast(GitHubPullRequestReview, item.record)
            for item in resolved
            if item.reference.artifact_type is ArtifactType.PULL_REQUEST_REVIEW
        ),
        key=_review_sort_key,
    )
    review_comments = sorted(
        (
            cast(GitHubPullRequestReviewComment, item.record)
            for item in resolved
            if item.reference.artifact_type is ArtifactType.PULL_REQUEST_REVIEW_COMMENT
        ),
        key=lambda comment: (comment.created_at, comment.id),
    )
    pull_request_files = [
        cast(GitHubPullRequestFile, item.record)
        for item in sorted(
            (
                item
                for item in resolved
                if item.reference.artifact_type is ArtifactType.PULL_REQUEST_FILE
            ),
            key=lambda item: (
                cast(GitHubPullRequestFile, item.record).filename,
                item.reference.identifier,
            ),
        )
    ]
    github_commits = sorted(
        (
            cast(GitHubCommitReference, item.record)
            for item in resolved
            if item.reference.artifact_type is ArtifactType.GITHUB_COMMIT_REFERENCE
        ),
        key=lambda commit: commit.sha,
    )
    local_commits = sorted(
        (
            cast(GitCommit, item.record)
            for item in resolved
            if item.reference.artifact_type is ArtifactType.LOCAL_GIT_COMMIT
        ),
        key=lambda commit: (commit.committed_at, commit.sha),
    )

    internal_relationships, contextual_relationships = _event_relationships(
        component,
        relationships,
    )
    return EngineeringEvent(
        event_id=_event_id(anchor),
        repository=anchor.repository,
        anchor=anchor,
        issues=issues,
        issue_comments=issue_comments,
        pull_requests=pull_requests,
        pull_request_comments=pull_request_comments,
        pull_request_reviews=reviews,
        pull_request_review_comments=review_comments,
        pull_request_files=pull_request_files,
        github_commit_references=github_commits,
        local_commits=local_commits,
        relationships=internal_relationships,
        contextual_relationships=contextual_relationships,
    )


def _event_relationships(
    component: set[ArtifactReference],
    relationships: Iterable[EngineeringRelationship],
) -> tuple[list[EngineeringRelationship], list[EngineeringRelationship]]:
    internal: list[EngineeringRelationship] = []
    contextual: list[EngineeringRelationship] = []
    for relationship in relationships:
        source_is_member = relationship.source in component
        target_is_member = relationship.target in component
        if _is_event_forming(relationship):
            if source_is_member and target_is_member:
                internal.append(relationship)
        elif source_is_member or target_is_member:
            contextual.append(relationship)
    return (
        sorted(internal, key=_relationship_sort_key),
        sorted(contextual, key=_relationship_sort_key),
    )


def _is_event_forming(relationship: EngineeringRelationship) -> bool:
    return (
        relationship.relationship_type in EVENT_FORMING_RELATIONSHIP_TYPES
        and relationship.source.repository == relationship.target.repository
    )


def _event_id(anchor: ArtifactReference) -> str:
    components = (
        "github.com",
        anchor.repository.owner,
        anchor.repository.name,
        anchor.artifact_type.value,
        anchor.identifier,
    )
    return "__".join(quote(component, safe="-.") for component in components)


def _event_sort_key(event: EngineeringEvent) -> tuple[str, str, int, int, str]:
    anchor_key = _anchor_sort_key(event.anchor)
    return (
        event.repository.owner,
        event.repository.name,
        anchor_key[0],
        anchor_key[1],
        anchor_key[2],
    )


def _anchor_sort_key(reference: ArtifactReference) -> tuple[int, int, str]:
    priority = _ANCHOR_PRIORITY[reference.artifact_type]
    numeric_identifier = (
        int(reference.identifier)
        if reference.artifact_type in {ArtifactType.PULL_REQUEST, ArtifactType.ISSUE}
        else 0
    )
    return (priority, numeric_identifier, reference.identifier)


def _artifact_sort_key(reference: ArtifactReference) -> tuple[str, str, int, str]:
    return (
        reference.repository.owner,
        reference.repository.name,
        _ARTIFACT_TYPE_ORDER[reference.artifact_type],
        reference.identifier,
    )


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


def _record_sort_key(record: EventRecord) -> str:
    return cast(BaseModel, record).model_dump_json()


def _review_sort_key(review: GitHubPullRequestReview) -> tuple[bool, datetime | None, int]:
    return (review.submitted_at is None, review.submitted_at, review.id)


def _issue_ref(issue: GitHubIssue) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.ISSUE,
        repository=issue.repository,
        identifier=str(issue.number),
    )


def _issue_comment_ref(comment: GitHubIssueComment) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.ISSUE_COMMENT,
        repository=comment.repository,
        identifier=f"issue-comment:{comment.id}",
    )


def _pull_request_ref(pull_request: GitHubPullRequest) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.PULL_REQUEST,
        repository=pull_request.repository,
        identifier=str(pull_request.number),
    )


def _pull_request_comment_ref(comment: GitHubIssueComment) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.PULL_REQUEST_COMMENT,
        repository=comment.repository,
        identifier=f"pr-comment:{comment.id}",
    )


def _review_ref(review: GitHubPullRequestReview) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.PULL_REQUEST_REVIEW,
        repository=review.repository,
        identifier=f"review:{review.id}",
    )


def _review_comment_ref(comment: GitHubPullRequestReviewComment) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.PULL_REQUEST_REVIEW_COMMENT,
        repository=comment.repository,
        identifier=f"review-comment:{comment.id}",
    )


def _pull_request_file_ref(
    data: EngineeringEventInput,
    pull_request_number: int,
    file: GitHubPullRequestFile,
) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.PULL_REQUEST_FILE,
        repository=data.repository,
        identifier=f"pr:{pull_request_number}:file:{file.filename}",
    )


def _github_commit_ref(
    data: EngineeringEventInput,
    commit: GitHubCommitReference,
) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.GITHUB_COMMIT_REFERENCE,
        repository=data.repository,
        identifier=commit.sha,
    )


def _local_commit_ref(
    data: EngineeringEventInput,
    commit: GitCommit,
) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.LOCAL_GIT_COMMIT,
        repository=data.repository,
        identifier=commit.sha,
    )


def _local_changed_file_ref(
    data: EngineeringEventInput,
    commit: GitCommit,
    file: ChangedFile,
) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.LOCAL_CHANGED_FILE,
        repository=data.repository,
        identifier=f"{commit.sha}:{file.path}",
    )
