import posixpath
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import quote

from reporecall.models import (
    ArtifactReference,
    ArtifactType,
    ChangedFile,
    EngineeringEvent,
    EngineeringRelationship,
    EventActor,
    EventActorType,
    EventMetadata,
    GitCommit,
    GitHubCommitReference,
    GitHubIssue,
    GitHubIssueComment,
    GitHubPullRequest,
    GitHubPullRequestFile,
    GitHubPullRequestReview,
    GitHubPullRequestReviewComment,
    GitHubUser,
    RelationshipType,
    RetrievalDocument,
    RetrievalDocumentSection,
    RetrievalSectionType,
    RetrievalSource,
)


class RetrievalDocumentBuilder:
    """Render one event and its metadata as a canonical retrieval document."""

    def build(
        self,
        event: EngineeringEvent,
        metadata: EventMetadata,
    ) -> RetrievalDocument:
        """Return a deterministic source-faithful document without performing I/O."""

        _validate_input_identity(event, metadata)
        document_id = f"{event.event_id}__retrieval"
        title = _document_title(event)
        pull_request_files = _pull_request_file_entries(event)
        local_files = _local_file_entries(event)

        sections = [
            _overview_section(event, document_id, title),
            *_metadata_sections(metadata),
            *_issue_sections(event),
            *_issue_comment_sections(event),
            *_pull_request_sections(event),
            *_pull_request_comment_sections(event),
            *_review_sections(event),
            *_review_comment_sections(event),
            *_commit_sections(event),
            *_changed_file_sections(pull_request_files, local_files),
            *_patch_sections(pull_request_files, local_files),
            *_relationship_sections(event.relationships, contextual=False),
            *_relationship_sections(event.contextual_relationships, contextual=True),
        ]
        section_tuple = tuple(sections)

        return RetrievalDocument(
            document_id=document_id,
            event_id=event.event_id,
            repository=event.repository,
            title=title,
            sections=section_tuple,
            text=_render_document_text(section_tuple),
            metadata=metadata,
            sources=_sources(event, pull_request_files, local_files),
        )


@dataclass(frozen=True)
class _PullRequestFileEntry:
    file: GitHubPullRequestFile
    artifact: ArtifactReference
    pull_request_number: str | None


@dataclass(frozen=True)
class _LocalFileEntry:
    file: ChangedFile
    artifact: ArtifactReference
    commit_sha: str


def _validate_input_identity(event: EngineeringEvent, metadata: EventMetadata) -> None:
    if event.event_id != metadata.event_id:
        raise ValueError("Event metadata does not belong to the supplied engineering event.")
    if event.repository != metadata.repository:
        raise ValueError("Event metadata repository does not match the engineering event.")


def _document_title(event: EngineeringEvent) -> str:
    pull_requests = _unique_pull_requests(event.pull_requests)
    if pull_requests:
        pull_request = _anchored_record(event.anchor, pull_requests)
        return _title_with_text(
            f"PR #{pull_request.number}",
            pull_request.title,
        )

    issues = _unique_issues(event.issues)
    if issues:
        issue = _anchored_record(event.anchor, issues)
        return _title_with_text(f"Issue #{issue.number}", issue.title)

    local_commits = _unique_local_commits(event.local_commits)
    if local_commits:
        commit = _anchored_commit(event.anchor, local_commits)
        return _title_with_text(
            f"Commit {commit.sha}",
            _first_line(commit.message),
        )

    github_commits = _unique_github_commits(event.github_commit_references)
    if github_commits:
        commit = _anchored_commit(event.anchor, github_commits)
        return _title_with_text(
            f"Commit {commit.sha}",
            _first_line(commit.message),
        )

    return event.event_id


def _title_with_text(prefix: str, source_text: str) -> str:
    normalized = _first_line(source_text)
    return f"{prefix} - {normalized}" if normalized else prefix


def _first_line(value: str) -> str:
    lines = _source_text(value).splitlines()
    return lines[0].strip() if lines else ""


def _anchored_record(
    anchor: ArtifactReference,
    records: tuple[GitHubIssue, ...] | tuple[GitHubPullRequest, ...],
) -> GitHubIssue | GitHubPullRequest:
    for record in records:
        if str(record.number) == anchor.identifier:
            expected_type = (
                ArtifactType.ISSUE
                if isinstance(record, GitHubIssue)
                else ArtifactType.PULL_REQUEST
            )
            if anchor.artifact_type is expected_type:
                return record
    return records[0]


def _anchored_commit(
    anchor: ArtifactReference,
    commits: tuple[GitCommit, ...] | tuple[GitHubCommitReference, ...],
) -> GitCommit | GitHubCommitReference:
    for commit in commits:
        if commit.sha == anchor.identifier:
            return commit
    return commits[0]


def _overview_section(
    event: EngineeringEvent,
    document_id: str,
    title: str,
) -> RetrievalDocumentSection:
    return RetrievalDocumentSection(
        section_id="overview",
        section_type=RetrievalSectionType.OVERVIEW,
        heading="OVERVIEW",
        content="\n".join(
            (
                f"Repository: {event.repository.owner}/{event.repository.name}",
                f"Event ID: {event.event_id}",
                f"Document ID: {document_id}",
                f"Title: {title}",
            )
        ),
        artifact=event.anchor,
    )


def _metadata_sections(metadata: EventMetadata) -> list[RetrievalDocumentSection]:
    lines = [f"Repository: {metadata.repository.owner}/{metadata.repository.name}"]
    _append_inline(lines, "Issue Numbers", (f"#{number}" for number in metadata.issue_numbers))
    _append_inline(
        lines,
        "Pull Request Numbers",
        (f"#{number}" for number in metadata.pull_request_numbers),
    )
    _append_list(lines, "Commit SHAs", metadata.commit_shas)
    _append_list(lines, "Authors", (_actor_text(actor) for actor in metadata.authors))
    _append_list(
        lines,
        "Participants",
        (_actor_text(actor) for actor in metadata.participants),
    )
    _append_inline(lines, "Labels", metadata.labels)
    _append_inline(lines, "Milestones", metadata.milestones)
    _append_inline(lines, "Languages", metadata.languages)
    _append_inline(lines, "Extensions", metadata.file_extensions)
    _append_list(lines, "Directories", metadata.directories)
    _append_list(lines, "Changed Paths", metadata.changed_paths)
    _append_list(lines, "Test Paths", metadata.test_paths)
    _append_list(lines, "Documentation Paths", metadata.documentation_paths)
    _append_list(lines, "Configuration Paths", metadata.configuration_paths)
    _append_list(lines, "Dependency Paths", metadata.dependency_paths)
    if metadata.changed_lines:
        _append_block(
            lines,
            "Statistics",
            (
                f"Added Lines: {metadata.added_lines}",
                f"Deleted Lines: {metadata.deleted_lines}",
                f"Changed Lines: {metadata.changed_lines}",
            ),
        )

    if not lines:
        return []
    return [
        RetrievalDocumentSection(
            section_id="metadata",
            section_type=RetrievalSectionType.METADATA,
            heading="METADATA",
            content="\n".join(lines),
        )
    ]


def _actor_text(actor: EventActor) -> str:
    if actor.actor_type is EventActorType.GITHUB_USER:
        return f"github_user: {actor.identifier}"
    if actor.name and actor.email:
        return f"git_author: {actor.name} <{actor.email}>"
    return f"git_author: {actor.name or actor.email or actor.identifier}"


def _issue_sections(event: EngineeringEvent) -> list[RetrievalDocumentSection]:
    sections: list[RetrievalDocumentSection] = []
    for issue in _unique_issues(event.issues):
        artifact = _issue_ref(issue)
        lines = [
            f"Title: {issue.title}",
            f"State: {issue.state.value}",
            f"Author: {_github_user_text(issue.author)}",
            f"Created: {issue.created_at.isoformat()}",
            f"Updated: {issue.updated_at.isoformat()}",
        ]
        if issue.closed_at is not None:
            lines.append(f"Closed: {issue.closed_at.isoformat()}")
        lines.append(f"URL: {issue.html_url}")
        if issue.labels:
            lines.append(
                "Labels: "
                + ", ".join(sorted({label.name for label in issue.labels}, key=_string_sort_key))
            )
        if issue.milestone is not None:
            lines.append(f"Milestone: {issue.milestone.title}")
        _append_source_body(lines, "Body", issue.body)
        sections.append(
            RetrievalDocumentSection(
                section_id=_section_id("issue", str(issue.number)),
                section_type=RetrievalSectionType.ISSUE,
                heading=f"ISSUE #{issue.number}",
                content="\n".join(lines),
                artifact=artifact,
            )
        )
    return sections


def _issue_comment_sections(event: EngineeringEvent) -> list[RetrievalDocumentSection]:
    sections: list[RetrievalDocumentSection] = []
    for comment in _unique_comments(event.issue_comments):
        lines = [
            f"Issue: #{comment.issue_number}",
            f"Author: {_github_user_text(comment.author)}",
            f"Created: {comment.created_at.isoformat()}",
            f"Updated: {comment.updated_at.isoformat()}",
            f"URL: {comment.html_url}",
        ]
        _append_source_body(lines, "Comment", comment.body)
        sections.append(
            RetrievalDocumentSection(
                section_id=_section_id(
                    "issue-comment",
                    str(comment.issue_number),
                    str(comment.id),
                ),
                section_type=RetrievalSectionType.ISSUE_COMMENT,
                heading=f"ISSUE #{comment.issue_number} COMMENT #{comment.id}",
                content="\n".join(lines),
                artifact=_issue_comment_ref(comment),
            )
        )
    return sections


def _pull_request_sections(event: EngineeringEvent) -> list[RetrievalDocumentSection]:
    sections: list[RetrievalDocumentSection] = []
    for pull_request in _unique_pull_requests(event.pull_requests):
        lines = [
            f"Title: {pull_request.title}",
            f"State: {pull_request.state.value}",
            f"Merged: {str(pull_request.is_merged).lower()}",
            f"Draft: {str(pull_request.draft).lower()}",
            f"Author: {_github_user_text(pull_request.author)}",
            f"Base: {pull_request.base.ref}",
            f"Head: {pull_request.head.ref}",
            f"Created: {pull_request.created_at.isoformat()}",
            f"Updated: {pull_request.updated_at.isoformat()}",
        ]
        if pull_request.closed_at is not None:
            lines.append(f"Closed: {pull_request.closed_at.isoformat()}")
        if pull_request.merged_at is not None:
            lines.append(f"Merged At: {pull_request.merged_at.isoformat()}")
        lines.append(f"URL: {pull_request.html_url}")
        if pull_request.labels:
            lines.append(
                "Labels: "
                + ", ".join(
                    sorted(
                        {label.name for label in pull_request.labels},
                        key=_string_sort_key,
                    )
                )
            )
        if pull_request.milestone is not None:
            lines.append(f"Milestone: {pull_request.milestone.title}")
        _append_source_body(lines, "Body", pull_request.body)
        sections.append(
            RetrievalDocumentSection(
                section_id=_section_id("pull-request", str(pull_request.number)),
                section_type=RetrievalSectionType.PULL_REQUEST,
                heading=f"PULL REQUEST #{pull_request.number}",
                content="\n".join(lines),
                artifact=_pull_request_ref(pull_request),
            )
        )
    return sections


def _pull_request_comment_sections(
    event: EngineeringEvent,
) -> list[RetrievalDocumentSection]:
    sections: list[RetrievalDocumentSection] = []
    for comment in _unique_comments(event.pull_request_comments):
        lines = [
            f"Pull Request: #{comment.issue_number}",
            f"Author: {_github_user_text(comment.author)}",
            f"Created: {comment.created_at.isoformat()}",
            f"Updated: {comment.updated_at.isoformat()}",
            f"URL: {comment.html_url}",
        ]
        _append_source_body(lines, "Comment", comment.body)
        sections.append(
            RetrievalDocumentSection(
                section_id=_section_id(
                    "pull-request-comment",
                    str(comment.issue_number),
                    str(comment.id),
                ),
                section_type=RetrievalSectionType.PULL_REQUEST_COMMENT,
                heading=(
                    f"PULL REQUEST #{comment.issue_number} CONVERSATION COMMENT #{comment.id}"
                ),
                content="\n".join(lines),
                artifact=_pull_request_comment_ref(comment),
            )
        )
    return sections


def _review_sections(event: EngineeringEvent) -> list[RetrievalDocumentSection]:
    sections: list[RetrievalDocumentSection] = []
    for review in _unique_reviews(event.pull_request_reviews):
        lines = [
            f"Pull Request: #{review.pull_request_number}",
            f"Reviewer: {_github_user_text(review.author)}",
            f"State: {review.state.value}",
        ]
        if review.submitted_at is not None:
            lines.append(f"Submitted: {review.submitted_at.isoformat()}")
        if review.commit_sha:
            lines.append(f"Commit: {review.commit_sha}")
        if review.html_url:
            lines.append(f"URL: {review.html_url}")
        _append_source_body(lines, "Body", review.body)
        sections.append(
            RetrievalDocumentSection(
                section_id=_section_id("review", str(review.id)),
                section_type=RetrievalSectionType.REVIEW,
                heading=(
                    f"PULL REQUEST #{review.pull_request_number} REVIEW #{review.id}"
                ),
                content="\n".join(lines),
                artifact=_review_ref(review),
            )
        )
    return sections


def _review_comment_sections(event: EngineeringEvent) -> list[RetrievalDocumentSection]:
    sections: list[RetrievalDocumentSection] = []
    for comment in _unique_review_comments(event.pull_request_review_comments):
        lines = [
            f"Pull Request: #{comment.pull_request_number}",
            f"Reviewer: {_github_user_text(comment.author)}",
            f"File: {comment.path}",
        ]
        if comment.review_id is not None:
            lines.append(f"Review: #{comment.review_id}")
        if comment.line is not None:
            lines.append(f"Line: {comment.line}")
        if comment.original_line is not None:
            lines.append(f"Original Line: {comment.original_line}")
        if comment.side is not None:
            lines.append(f"Side: {comment.side}")
        if comment.start_line is not None:
            lines.append(f"Start Line: {comment.start_line}")
        if comment.start_side is not None:
            lines.append(f"Start Side: {comment.start_side}")
        if comment.commit_sha:
            lines.append(f"Commit: {comment.commit_sha}")
        if comment.original_commit_sha:
            lines.append(f"Original Commit: {comment.original_commit_sha}")
        lines.extend(
            (
                f"Created: {comment.created_at.isoformat()}",
                f"Updated: {comment.updated_at.isoformat()}",
                f"URL: {comment.html_url}",
            )
        )
        _append_source_body(lines, "Comment", comment.body)
        sections.append(
            RetrievalDocumentSection(
                section_id=_section_id("review-comment", str(comment.id)),
                section_type=RetrievalSectionType.REVIEW_COMMENT,
                heading=(
                    f"PULL REQUEST #{comment.pull_request_number} REVIEW COMMENT #{comment.id}"
                ),
                content="\n".join(lines),
                artifact=_review_comment_ref(comment),
            )
        )
    return sections


def _commit_sections(event: EngineeringEvent) -> list[RetrievalDocumentSection]:
    local_by_sha = {commit.sha: commit for commit in _unique_local_commits(event.local_commits)}
    github_by_sha = {
        commit.sha: commit
        for commit in _unique_github_commits(event.github_commit_references)
    }
    shas = sorted(
        {*local_by_sha, *github_by_sha},
        key=lambda sha: _combined_commit_sort_key(
            sha,
            local_by_sha.get(sha),
            github_by_sha.get(sha),
        ),
    )
    sections: list[RetrievalDocumentSection] = []
    for sha in shas:
        local_commit = local_by_sha.get(sha)
        github_commit = github_by_sha.get(sha)
        if local_commit is not None:
            content = _local_commit_content(local_commit, github_commit)
            artifact = _local_commit_ref(event, local_commit)
        elif github_commit is not None:
            content = _github_commit_content(github_commit)
            artifact = _github_commit_ref(event, github_commit)
        else:
            continue
        sections.append(
            RetrievalDocumentSection(
                section_id=_section_id("commit", sha),
                section_type=RetrievalSectionType.COMMIT,
                heading=f"COMMIT {sha}",
                content=content,
                artifact=artifact,
            )
        )
    return sections


def _local_commit_content(
    commit: GitCommit,
    github_commit: GitHubCommitReference | None,
) -> str:
    lines = [
        f"SHA: {commit.sha}",
        f"Author: {commit.author_name or 'unknown'}",
    ]
    if commit.author_email:
        lines.append(f"Author Email: {commit.author_email}")
    lines.extend(
        (
            f"Authored: {commit.authored_at.isoformat()}",
            f"Committed: {commit.committed_at.isoformat()}",
        )
    )
    _append_source_body(lines, "Message", commit.message)
    if (
        github_commit is not None
        and _source_text(github_commit.message) != _source_text(commit.message)
    ):
        _append_source_body(lines, "GitHub Message", github_commit.message)
    return "\n".join(lines)


def _github_commit_content(commit: GitHubCommitReference) -> str:
    lines = [f"SHA: {commit.sha}"]
    if commit.author_name:
        lines.append(f"Author: {commit.author_name}")
    if commit.author_email:
        lines.append(f"Author Email: {commit.author_email}")
    if commit.authored_at is not None:
        lines.append(f"Authored: {commit.authored_at.isoformat()}")
    if commit.html_url:
        lines.append(f"URL: {commit.html_url}")
    _append_source_body(lines, "Message", commit.message)
    return "\n".join(lines)


def _changed_file_sections(
    pull_request_files: tuple[_PullRequestFileEntry, ...],
    local_files: tuple[_LocalFileEntry, ...],
) -> list[RetrievalDocumentSection]:
    sections: list[RetrievalDocumentSection] = []
    pull_request_paths = {_normalized_path(entry.file.filename) for entry in pull_request_files}
    for pull_request_entry in pull_request_files:
        lines = [f"Path: {pull_request_entry.file.filename}"]
        if pull_request_entry.pull_request_number is not None:
            lines.insert(0, f"Pull Request: #{pull_request_entry.pull_request_number}")
        lines.extend(
            (
                f"Status: {pull_request_entry.file.status.value}",
                f"Additions: {pull_request_entry.file.additions}",
                f"Deletions: {pull_request_entry.file.deletions}",
            )
        )
        if pull_request_entry.file.previous_filename:
            lines.append(f"Previous Path: {pull_request_entry.file.previous_filename}")
        sections.append(
            RetrievalDocumentSection(
                section_id=_section_id(
                    "pull-request-file",
                    pull_request_entry.artifact.identifier,
                ),
                section_type=RetrievalSectionType.CHANGED_FILE,
                heading=f"CHANGED FILE {pull_request_entry.file.filename}",
                content="\n".join(lines),
                artifact=pull_request_entry.artifact,
            )
        )

    for local_entry in local_files:
        if _normalized_path(local_entry.file.path) in pull_request_paths:
            continue
        lines = [
            f"Commit: {local_entry.commit_sha}",
            f"Path: {local_entry.file.path}",
            f"Status: {local_entry.file.change_type.value}",
            f"Additions: {local_entry.file.additions}",
            f"Deletions: {local_entry.file.deletions}",
        ]
        if local_entry.file.old_path:
            lines.append(f"Previous Path: {local_entry.file.old_path}")
        sections.append(
            RetrievalDocumentSection(
                section_id=_section_id(
                    "local-changed-file",
                    local_entry.commit_sha,
                    local_entry.file.path,
                ),
                section_type=RetrievalSectionType.CHANGED_FILE,
                heading=f"CHANGED FILE {local_entry.file.path}",
                content="\n".join(lines),
                artifact=local_entry.artifact,
            )
        )
    return sections


def _patch_sections(
    pull_request_files: tuple[_PullRequestFileEntry, ...],
    local_files: tuple[_LocalFileEntry, ...],
) -> list[RetrievalDocumentSection]:
    sections: list[RetrievalDocumentSection] = []
    local_patch_paths = {
        _normalized_path(entry.file.path)
        for entry in local_files
        if entry.file.patch is not None
    }
    for local_entry in local_files:
        if local_entry.file.patch is None:
            continue
        lines = [
            f"Commit: {local_entry.commit_sha}",
            f"File: {local_entry.file.path}",
        ]
        _append_source_body(lines, "Patch", local_entry.file.patch)
        sections.append(
            RetrievalDocumentSection(
                section_id=_section_id(
                    "patch",
                    local_entry.commit_sha,
                    local_entry.file.path,
                ),
                section_type=RetrievalSectionType.PATCH,
                heading=f"PATCH {local_entry.commit_sha} {local_entry.file.path}",
                content="\n".join(lines),
                artifact=local_entry.artifact,
            )
        )

    for pull_request_entry in pull_request_files:
        if pull_request_entry.file.patch is None:
            continue
        if _normalized_path(pull_request_entry.file.filename) in local_patch_paths:
            continue
        lines = [f"File: {pull_request_entry.file.filename}"]
        if pull_request_entry.pull_request_number is not None:
            lines.insert(0, f"Pull Request: #{pull_request_entry.pull_request_number}")
        _append_source_body(lines, "Patch", pull_request_entry.file.patch)
        sections.append(
            RetrievalDocumentSection(
                section_id=_section_id(
                    "pull-request-patch",
                    pull_request_entry.artifact.identifier,
                ),
                section_type=RetrievalSectionType.PATCH,
                heading=f"PATCH {pull_request_entry.file.filename}",
                content="\n".join(lines),
                artifact=pull_request_entry.artifact,
            )
        )
    return sections


def _relationship_sections(
    relationships: Iterable[EngineeringRelationship],
    *,
    contextual: bool,
) -> list[RetrievalDocumentSection]:
    relationship_list = sorted(relationships, key=_relationship_sort_key)
    if not relationship_list:
        return []

    grouped: dict[
        tuple[ArtifactReference, RelationshipType, ArtifactReference],
        list[EngineeringRelationship],
    ] = {}
    for relationship in relationship_list:
        key = (
            relationship.source,
            relationship.relationship_type,
            relationship.target,
        )
        grouped.setdefault(key, []).append(relationship)

    blocks: list[str] = []
    for key in sorted(grouped, key=_relationship_group_sort_key):
        source, relationship_type, target = key
        lines = [
            (
                f"{_artifact_label(source)} "
                f"[{relationship_type.value}] "
                f"{_artifact_label(target)}"
            ),
            "Evidence:",
        ]
        for relationship in grouped[key]:
            evidence = f"- {relationship.evidence_type.value}"
            if relationship.evidence:
                evidence += f": {relationship.evidence}"
            if relationship.source_field:
                evidence += f" (source field: {relationship.source_field})"
            lines.append(evidence)
        blocks.append("\n".join(lines))

    section_type = (
        RetrievalSectionType.CONTEXTUAL_RELATIONSHIP
        if contextual
        else RetrievalSectionType.RELATIONSHIP
    )
    heading = "CONTEXTUAL RELATIONSHIPS" if contextual else "RELATIONSHIPS"
    section_id = "contextual-relationships" if contextual else "relationships"
    return [
        RetrievalDocumentSection(
            section_id=section_id,
            section_type=section_type,
            heading=heading,
            content="\n\n".join(blocks),
        )
    ]


def _pull_request_file_entries(
    event: EngineeringEvent,
) -> tuple[_PullRequestFileEntry, ...]:
    artifacts_by_path: dict[str, list[tuple[ArtifactReference, str]]] = {}
    for relationship in sorted(event.relationships, key=_relationship_sort_key):
        if relationship.relationship_type is not RelationshipType.PULL_REQUEST_CHANGES_FILE:
            continue
        if relationship.target.artifact_type is not ArtifactType.PULL_REQUEST_FILE:
            continue
        prefix = f"pr:{relationship.source.identifier}:file:"
        if not relationship.target.identifier.startswith(prefix):
            continue
        path = _normalized_path(relationship.target.identifier.removeprefix(prefix))
        artifacts_by_path.setdefault(path, []).append(
            (relationship.target, relationship.source.identifier)
        )

    files = sorted(
        event.pull_request_files,
        key=lambda file: (_normalized_path(file.filename), file.model_dump_json()),
    )
    pull_requests = _unique_pull_requests(event.pull_requests)
    fallback_pull_request = str(pull_requests[0].number) if len(pull_requests) == 1 else None
    entries: list[_PullRequestFileEntry] = []
    occurrence_by_path: dict[str, int] = {}
    for file in files:
        path = _normalized_path(file.filename)
        candidates = artifacts_by_path.get(path, [])
        pull_request_number: str | None
        if candidates:
            artifact, pull_request_number = candidates.pop(0)
        else:
            occurrence = occurrence_by_path.get(path, 0)
            occurrence_by_path[path] = occurrence + 1
            pull_request_number = fallback_pull_request
            parent = pull_request_number or f"event:{event.event_id}:{occurrence}"
            artifact = ArtifactReference(
                artifact_type=ArtifactType.PULL_REQUEST_FILE,
                repository=event.repository,
                identifier=f"pr:{parent}:file:{file.filename}",
            )
        entries.append(
            _PullRequestFileEntry(
                file=file,
                artifact=artifact,
                pull_request_number=pull_request_number,
            )
        )
    return tuple(entries)


def _local_file_entries(event: EngineeringEvent) -> tuple[_LocalFileEntry, ...]:
    entries_by_identity: dict[tuple[str, str], _LocalFileEntry] = {}
    for commit in _unique_local_commits(event.local_commits):
        for file in sorted(
            commit.changed_files,
            key=lambda item: (_normalized_path(item.path), item.model_dump_json()),
        ):
            key = (commit.sha, _normalized_path(file.path))
            entries_by_identity.setdefault(
                key,
                _LocalFileEntry(
                    file=file,
                    artifact=ArtifactReference(
                        artifact_type=ArtifactType.LOCAL_CHANGED_FILE,
                        repository=event.repository,
                        identifier=f"{commit.sha}:{file.path}",
                    ),
                    commit_sha=commit.sha,
                ),
            )
    return tuple(
        entries_by_identity[key]
        for key in sorted(entries_by_identity, key=lambda item: (item[0], item[1]))
    )


def _sources(
    event: EngineeringEvent,
    pull_request_files: tuple[_PullRequestFileEntry, ...],
    local_files: tuple[_LocalFileEntry, ...],
) -> tuple[RetrievalSource, ...]:
    candidates = [
        RetrievalSource(
            artifact=event.anchor,
            label=_artifact_label(event.anchor),
        )
    ]
    candidates.extend(
        RetrievalSource(
            artifact=_issue_ref(issue),
            url=issue.html_url,
            label=f"Issue #{issue.number}",
        )
        for issue in _unique_issues(event.issues)
    )
    candidates.extend(
        RetrievalSource(
            artifact=_issue_comment_ref(comment),
            url=comment.html_url,
            label=f"Issue Comment #{comment.id}",
        )
        for comment in _unique_comments(event.issue_comments)
    )
    candidates.extend(
        RetrievalSource(
            artifact=_pull_request_ref(pull_request),
            url=pull_request.html_url,
            label=f"Pull Request #{pull_request.number}",
        )
        for pull_request in _unique_pull_requests(event.pull_requests)
    )
    candidates.extend(
        RetrievalSource(
            artifact=_pull_request_comment_ref(comment),
            url=comment.html_url,
            label=f"Pull Request Comment #{comment.id}",
        )
        for comment in _unique_comments(event.pull_request_comments)
    )
    candidates.extend(
        RetrievalSource(
            artifact=_review_ref(review),
            url=review.html_url,
            label=f"Review #{review.id}",
        )
        for review in _unique_reviews(event.pull_request_reviews)
    )
    candidates.extend(
        RetrievalSource(
            artifact=_review_comment_ref(comment),
            url=comment.html_url,
            label=f"Review Comment #{comment.id}",
        )
        for comment in _unique_review_comments(event.pull_request_review_comments)
    )
    candidates.extend(
        RetrievalSource(
            artifact=entry.artifact,
            url=entry.file.blob_url or entry.file.raw_url,
            label=f"Pull Request File {entry.file.filename}",
        )
        for entry in pull_request_files
    )
    candidates.extend(
        RetrievalSource(
            artifact=_github_commit_ref(event, commit),
            url=commit.html_url,
            label=f"GitHub Commit {commit.sha}",
        )
        for commit in _unique_github_commits(event.github_commit_references)
    )
    candidates.extend(
        RetrievalSource(
            artifact=_local_commit_ref(event, commit),
            label=f"Local Commit {commit.sha}",
        )
        for commit in _unique_local_commits(event.local_commits)
    )
    candidates.extend(
        RetrievalSource(
            artifact=entry.artifact,
            label=f"Local Changed File {entry.file.path} at {entry.commit_sha}",
        )
        for entry in local_files
    )
    for relationship in (*event.relationships, *event.contextual_relationships):
        candidates.extend(
            (
                RetrievalSource(
                    artifact=relationship.source,
                    label=_artifact_label(relationship.source),
                ),
                RetrievalSource(
                    artifact=relationship.target,
                    label=_artifact_label(relationship.target),
                ),
            )
        )

    by_artifact: dict[ArtifactReference, RetrievalSource] = {}
    for candidate in sorted(candidates, key=_source_preference_key):
        by_artifact.setdefault(candidate.artifact, candidate)

    seen_urls: set[str] = set()
    sources: list[RetrievalSource] = []
    for source in sorted(by_artifact.values(), key=_source_sort_key):
        if source.url is not None and source.url in seen_urls:
            continue
        if source.url is not None:
            seen_urls.add(source.url)
        sources.append(source)
    return tuple(sources)


def _render_document_text(sections: tuple[RetrievalDocumentSection, ...]) -> str:
    rendered_sections = [
        f"=== {section.heading} ===\n\n{section.content}"
        for section in sections
    ]
    return "REPORECALL ENGINEERING EVENT\n\n" + "\n\n".join(rendered_sections) + "\n"


def _append_inline(lines: list[str], heading: str, values: Iterable[object]) -> None:
    items = tuple(str(value) for value in values)
    if items:
        lines.append(f"{heading}: {', '.join(items)}")


def _append_list(lines: list[str], heading: str, values: Iterable[object]) -> None:
    items = tuple(str(value) for value in values)
    if items:
        _append_block(lines, heading, (f"- {item}" for item in items))


def _append_block(lines: list[str], heading: str, values: Iterable[str]) -> None:
    items = tuple(values)
    if not items:
        return
    if lines:
        lines.append("")
    lines.append(f"{heading}:")
    lines.extend(items)


def _append_source_body(lines: list[str], heading: str, value: str | None) -> None:
    if value is None:
        return
    source_text = _source_text(value)
    if not source_text:
        return
    lines.extend(("", f"{heading}:", source_text))


def _source_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip("\n")


def _github_user_text(user: GitHubUser | None) -> str:
    return user.login if user is not None else "unknown"


def _section_id(*components: str) -> str:
    return "-".join(quote(component, safe="._-") for component in components)


def _normalized_path(path: str) -> str:
    normalized = posixpath.normpath(path.strip().replace("\\", "/"))
    return "" if normalized == "." else normalized


def _artifact_label(reference: ArtifactReference) -> str:
    labels = {
        ArtifactType.ISSUE: "Issue",
        ArtifactType.ISSUE_COMMENT: "Issue Comment",
        ArtifactType.PULL_REQUEST: "Pull Request",
        ArtifactType.PULL_REQUEST_FILE: "Pull Request File",
        ArtifactType.PULL_REQUEST_COMMENT: "Pull Request Comment",
        ArtifactType.PULL_REQUEST_REVIEW: "Review",
        ArtifactType.PULL_REQUEST_REVIEW_COMMENT: "Review Comment",
        ArtifactType.GITHUB_COMMIT_REFERENCE: "GitHub Commit",
        ArtifactType.LOCAL_GIT_COMMIT: "Local Commit",
        ArtifactType.LOCAL_CHANGED_FILE: "Local Changed File",
    }
    prefix = labels[reference.artifact_type]
    identifier = reference.identifier.rsplit(":", 1)[-1]
    number_prefix = "#" if reference.artifact_type in {
        ArtifactType.ISSUE,
        ArtifactType.ISSUE_COMMENT,
        ArtifactType.PULL_REQUEST,
        ArtifactType.PULL_REQUEST_COMMENT,
        ArtifactType.PULL_REQUEST_REVIEW,
        ArtifactType.PULL_REQUEST_REVIEW_COMMENT,
    } else ""
    return f"{prefix} {number_prefix}{identifier}"


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


def _github_commit_ref(
    event: EngineeringEvent,
    commit: GitHubCommitReference,
) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.GITHUB_COMMIT_REFERENCE,
        repository=event.repository,
        identifier=commit.sha,
    )


def _local_commit_ref(event: EngineeringEvent, commit: GitCommit) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=ArtifactType.LOCAL_GIT_COMMIT,
        repository=event.repository,
        identifier=commit.sha,
    )


def _unique_issues(issues: Iterable[GitHubIssue]) -> tuple[GitHubIssue, ...]:
    by_number: dict[int, GitHubIssue] = {}
    for issue in sorted(issues, key=lambda item: (item.number, item.model_dump_json())):
        by_number.setdefault(issue.number, issue)
    return tuple(by_number[number] for number in sorted(by_number))


def _unique_pull_requests(
    pull_requests: Iterable[GitHubPullRequest],
) -> tuple[GitHubPullRequest, ...]:
    by_number: dict[int, GitHubPullRequest] = {}
    for pull_request in sorted(
        pull_requests,
        key=lambda item: (item.number, item.model_dump_json()),
    ):
        by_number.setdefault(pull_request.number, pull_request)
    return tuple(by_number[number] for number in sorted(by_number))


def _unique_comments(
    comments: Iterable[GitHubIssueComment],
) -> tuple[GitHubIssueComment, ...]:
    by_id: dict[int, GitHubIssueComment] = {}
    for comment in sorted(
        comments,
        key=lambda item: (item.created_at, item.id, item.model_dump_json()),
    ):
        by_id.setdefault(comment.id, comment)
    return tuple(sorted(by_id.values(), key=lambda item: (item.created_at, item.id)))


def _unique_reviews(
    reviews: Iterable[GitHubPullRequestReview],
) -> tuple[GitHubPullRequestReview, ...]:
    by_id: dict[int, GitHubPullRequestReview] = {}
    for review in sorted(
        reviews,
        key=lambda item: (
            item.submitted_at is None,
            item.submitted_at,
            item.id,
            item.model_dump_json(),
        ),
    ):
        by_id.setdefault(review.id, review)
    return tuple(
        sorted(
            by_id.values(),
            key=lambda item: (item.submitted_at is None, item.submitted_at, item.id),
        )
    )


def _unique_review_comments(
    comments: Iterable[GitHubPullRequestReviewComment],
) -> tuple[GitHubPullRequestReviewComment, ...]:
    by_id: dict[int, GitHubPullRequestReviewComment] = {}
    for comment in sorted(
        comments,
        key=lambda item: (item.created_at, item.id, item.model_dump_json()),
    ):
        by_id.setdefault(comment.id, comment)
    return tuple(sorted(by_id.values(), key=lambda item: (item.created_at, item.id)))


def _unique_local_commits(commits: Iterable[GitCommit]) -> tuple[GitCommit, ...]:
    by_sha: dict[str, GitCommit] = {}
    for commit in sorted(commits, key=lambda item: (item.sha, item.model_dump_json())):
        by_sha.setdefault(commit.sha, commit)
    return tuple(sorted(by_sha.values(), key=_local_commit_sort_key))


def _unique_github_commits(
    commits: Iterable[GitHubCommitReference],
) -> tuple[GitHubCommitReference, ...]:
    by_sha: dict[str, GitHubCommitReference] = {}
    for commit in sorted(commits, key=lambda item: (item.sha, item.model_dump_json())):
        by_sha.setdefault(commit.sha, commit)
    return tuple(sorted(by_sha.values(), key=_github_commit_sort_key))


def _local_commit_sort_key(commit: GitCommit) -> tuple[bool, str, str]:
    return False, commit.committed_at.isoformat(), commit.sha


def _github_commit_sort_key(
    commit: GitHubCommitReference,
) -> tuple[bool, str, str]:
    return (
        commit.authored_at is None,
        commit.authored_at.isoformat() if commit.authored_at is not None else "",
        commit.sha,
    )


def _combined_commit_sort_key(
    sha: str,
    local_commit: GitCommit | None,
    github_commit: GitHubCommitReference | None,
) -> tuple[bool, str, str]:
    timestamp: datetime | None = None
    if local_commit is not None:
        timestamp = local_commit.committed_at
    elif github_commit is not None:
        timestamp = github_commit.authored_at
    return timestamp is None, timestamp.isoformat() if timestamp is not None else "", sha


def _relationship_sort_key(relationship: EngineeringRelationship) -> tuple[str, ...]:
    return (
        *_artifact_sort_key(relationship.source),
        relationship.relationship_type.value,
        *_artifact_sort_key(relationship.target),
        relationship.evidence_type.value,
        relationship.evidence or "",
        relationship.source_field or "",
    )


def _relationship_group_sort_key(
    key: tuple[ArtifactReference, RelationshipType, ArtifactReference],
) -> tuple[str, ...]:
    source, relationship_type, target = key
    return (
        *_artifact_sort_key(source),
        relationship_type.value,
        *_artifact_sort_key(target),
    )


def _artifact_sort_key(reference: ArtifactReference) -> tuple[str, ...]:
    return (
        reference.repository.owner,
        reference.repository.name,
        reference.artifact_type.value,
        reference.identifier,
    )


def _source_preference_key(source: RetrievalSource) -> tuple[object, ...]:
    return (
        *_artifact_sort_key(source.artifact),
        source.url is None,
        source.url or "",
        source.label,
    )


def _source_sort_key(source: RetrievalSource) -> tuple[str, ...]:
    return (*_artifact_sort_key(source.artifact), source.url or "", source.label)


def _string_sort_key(value: str) -> tuple[str, str]:
    return value.casefold(), value
