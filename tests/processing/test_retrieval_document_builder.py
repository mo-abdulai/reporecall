from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

import reporecall.processing.retrieval_document_builder as builder_module
from reporecall.github import GitHubRepository
from reporecall.models import (
    ArtifactReference,
    ArtifactType,
    ChangedFile,
    EngineeringEvent,
    EngineeringRelationship,
    EventMetadata,
    FileChangeType,
    GitCommit,
    GitHubBranchReference,
    GitHubCommitReference,
    GitHubIssue,
    GitHubIssueComment,
    GitHubIssueLabel,
    GitHubPullRequest,
    GitHubPullRequestFile,
    GitHubPullRequestFileStatus,
    GitHubPullRequestReview,
    GitHubPullRequestReviewComment,
    GitHubUser,
    IssueState,
    PullRequestState,
    RelationshipEvidenceType,
    RelationshipType,
    RetrievalSectionType,
    ReviewState,
)
from reporecall.processing import EventMetadataExtractor, RetrievalDocumentBuilder


def test_complete_event_renders_structured_source_faithful_document():
    patch = "@@ -8,2 +8,3 @@\n connection = acquire()\n+release(connection)"
    commit_message = "Release connection after retry failure\n\nKeep cleanup deterministic."
    relationships = [
        _relationship(
            ArtifactType.PULL_REQUEST,
            "20",
            RelationshipType.PULL_REQUEST_CLOSES_ISSUE,
            ArtifactType.ISSUE,
            "10",
            RelationshipEvidenceType.CLOSING_KEYWORD,
            evidence="Fixes #10",
            source_field="body",
        ),
        _relationship(
            ArtifactType.PULL_REQUEST,
            "20",
            RelationshipType.PULL_REQUEST_CLOSES_ISSUE,
            ArtifactType.ISSUE,
            "10",
            RelationshipEvidenceType.GITHUB_TIMELINE_CLOSED_EVENT,
        ),
        _relationship(
            ArtifactType.PULL_REQUEST,
            "20",
            RelationshipType.PULL_REQUEST_CHANGES_FILE,
            ArtifactType.PULL_REQUEST_FILE,
            "pr:20:file:src/database.py",
            RelationshipEvidenceType.PR_FILE_LIST,
        ),
        _relationship(
            ArtifactType.PULL_REQUEST,
            "20",
            RelationshipType.PULL_REQUEST_HAS_COMMIT,
            ArtifactType.GITHUB_COMMIT_REFERENCE,
            "abc123",
            RelationshipEvidenceType.PR_COMMIT_LIST,
        ),
        _relationship(
            ArtifactType.GITHUB_COMMIT_REFERENCE,
            "abc123",
            RelationshipType.GITHUB_COMMIT_MATCHES_LOCAL_COMMIT,
            ArtifactType.LOCAL_GIT_COMMIT,
            "abc123",
            RelationshipEvidenceType.EXACT_SHA_MATCH,
        ),
        _relationship(
            ArtifactType.LOCAL_GIT_COMMIT,
            "abc123",
            RelationshipType.LOCAL_COMMIT_CHANGES_FILE,
            ArtifactType.LOCAL_CHANGED_FILE,
            "abc123:src/database.py",
            RelationshipEvidenceType.LOCAL_COMMIT_FILE_LIST,
        ),
    ]
    contextual_relationship = _relationship(
        ArtifactType.PULL_REQUEST,
        "20",
        RelationshipType.PULL_REQUEST_REFERENCES_ISSUE,
        ArtifactType.ISSUE,
        "99",
        RelationshipEvidenceType.TEXT_REFERENCE,
        evidence="See #99",
        source_field="body",
    )
    event = _event(
        issues=[
            _issue(
                10,
                title="Pool exhausted after worker retry",
                body=(
                    "Request failed at /workers/retry\n\n"
                    "Traceback (most recent call last):\n"
                    '  File "worker.py", line 42, in retry\n'
                    "ConnectionError: pool exhausted"
                ),
                author=GitHubUser(login="alice"),
                labels=[GitHubIssueLabel(name="bug")],
            )
        ],
        issue_comments=[
            _comment(101, 10, "The exact exception is PoolTimeoutError.", "bob")
        ],
        pull_requests=[
            _pull_request(
                20,
                title="Release connection after retry failure",
                body="Fixes #10\n\nThe cleanup stays in a finally block.",
                author=GitHubUser(login="alice"),
            )
        ],
        pull_request_comments=[
            _comment(102, 20, "Please retain the /workers/retry behavior.", "carol")
        ],
        pull_request_reviews=[
            _review(201, 20, "Cleanup ordering now looks correct.", "dave")
        ],
        pull_request_review_comments=[
            _review_comment(
                301,
                20,
                201,
                "This must run before the retry is scheduled.",
                "dave",
            )
        ],
        pull_request_files=[
            _pull_request_file(
                "src/database.py",
                additions=5,
                deletions=2,
                patch="@@ pull request patch",
            )
        ],
        github_commit_references=[
            _github_commit("abc123", commit_message, url=True)
        ],
        local_commits=[
            _local_commit(
                "abc123",
                commit_message,
                changed_files=[_changed_file("src/database.py", patch=patch)],
            )
        ],
        relationships=relationships,
        contextual_relationships=[contextual_relationship],
    )
    original_event = event.model_dump()
    metadata = EventMetadataExtractor().extract(event)
    original_metadata = metadata.model_dump()

    document = RetrievalDocumentBuilder().build(event, metadata)

    assert document.document_id == f"{event.event_id}__retrieval"
    assert document.event_id == event.event_id
    assert document.repository == _repository()
    assert document.title == "PR #20 - Release connection after retry failure"
    assert [section.section_type for section in document.sections] == [
        RetrievalSectionType.OVERVIEW,
        RetrievalSectionType.METADATA,
        RetrievalSectionType.ISSUE,
        RetrievalSectionType.ISSUE_COMMENT,
        RetrievalSectionType.PULL_REQUEST,
        RetrievalSectionType.PULL_REQUEST_COMMENT,
        RetrievalSectionType.REVIEW,
        RetrievalSectionType.REVIEW_COMMENT,
        RetrievalSectionType.COMMIT,
        RetrievalSectionType.CHANGED_FILE,
        RetrievalSectionType.PATCH,
        RetrievalSectionType.RELATIONSHIP,
        RetrievalSectionType.CONTEXTUAL_RELATIONSHIP,
    ]
    assert "ConnectionError: pool exhausted" in document.text
    assert "PoolTimeoutError" in document.text
    assert "/workers/retry" in document.text
    assert "Labels: bug" in document.text
    assert "Languages: Python" in document.text
    assert "Added Lines: 5" in document.text
    assert "Deleted Lines: 2" in document.text
    assert patch in document.text
    assert document.text.count(commit_message) == 1
    assert len(_sections(document, RetrievalSectionType.COMMIT)) == 1
    assert len(_sections(document, RetrievalSectionType.CHANGED_FILE)) == 1
    assert len(_sections(document, RetrievalSectionType.PATCH)) == 1
    relationship_text = _sections(document, RetrievalSectionType.RELATIONSHIP)[0].content
    assert "closing_keyword: Fixes #10 (source field: body)" in relationship_text
    assert "github_timeline_closed_event" in relationship_text
    contextual_text = _sections(
        document,
        RetrievalSectionType.CONTEXTUAL_RELATIONSHIP,
    )[0].content
    assert "pull_request_references_issue" in contextual_text
    assert "text_reference: See #99" in contextual_text
    assert {
        source.artifact.artifact_type
        for source in document.sources
        if source.artifact.identifier == "abc123"
    } == {ArtifactType.GITHUB_COMMIT_REFERENCE, ArtifactType.LOCAL_GIT_COMMIT}
    assert any(
        source.url == "https://github.com/owner/repo/commit/abc123"
        for source in document.sources
    )
    assert event.model_dump() == original_event
    assert metadata.model_dump() == original_metadata


def test_issue_only_event_uses_issue_title_and_omits_empty_domains():
    event = _event(
        event_id="github.com__owner__repo__issue__10",
        anchor=_ref(ArtifactType.ISSUE, "10"),
        issues=[_issue(10, title="Deleted user report", body=None, author=None)],
    )

    document = _build(event)

    assert document.title == "Issue #10 - Deleted user report"
    assert "Author: unknown" in document.text
    assert "Body:" not in document.text
    assert "None" not in document.text
    section_types = {section.section_type for section in document.sections}
    assert RetrievalSectionType.PULL_REQUEST not in section_types
    assert RetrievalSectionType.REVIEW not in section_types
    assert RetrievalSectionType.COMMIT not in section_types
    assert RetrievalSectionType.PATCH not in section_types


def test_pr_only_and_commit_only_events_use_exact_source_titles():
    pull_request_event = _event(
        event_id="github.com__owner__repo__pull_request__30",
        anchor=_ref(ArtifactType.PULL_REQUEST, "30"),
        pull_requests=[_pull_request(30, title="Do Not Rewrite APIConfig")],
    )
    commit_event = _event(
        event_id="github.com__owner__repo__local_git_commit__def456",
        anchor=_ref(ArtifactType.LOCAL_GIT_COMMIT, "def456"),
        local_commits=[
            _local_commit("def456", "Preserve APIConfig casing\n\nDetailed rationale")
        ],
    )

    pull_request_document = _build(pull_request_event)
    commit_document = _build(commit_event)

    assert pull_request_document.title == "PR #30 - Do Not Rewrite APIConfig"
    assert commit_document.title == "Commit def456 - Preserve APIConfig casing"
    assert "Detailed rationale" in commit_document.text


def test_github_commit_title_fallback_and_missing_patch_are_rendered_cleanly():
    event = _event(
        event_id="github.com__owner__repo__github_commit_reference__fed987",
        anchor=_ref(ArtifactType.GITHUB_COMMIT_REFERENCE, "fed987"),
        pull_request_files=[_pull_request_file("assets/logo.png", patch=None)],
        github_commit_references=[
            _github_commit("fed987", "Record binary logo update\n\nNo textual patch")
        ],
    )

    document = _build(event)

    assert document.title == "Commit fed987 - Record binary logo update"
    assert "No textual patch" in document.text
    assert _sections(document, RetrievalSectionType.PATCH) == []
    assert "Patch: None" not in document.text


def test_empty_metadata_collections_do_not_render_empty_labels():
    event = _event()
    document = RetrievalDocumentBuilder().build(
        event,
        EventMetadata(event_id=event.event_id, repository=event.repository),
    )
    metadata_section = _sections(document, RetrievalSectionType.METADATA)[0]

    assert metadata_section.content == "Repository: owner/repo"
    assert "Labels:" not in document.text
    assert "Milestones:" not in document.text
    assert "Tests Changed:" not in document.text


def test_local_commit_and_patch_take_rendering_precedence_without_losing_sources():
    local_patch = "@@ -1 +1 @@\n-old_call()\n+new_call()\n"
    event = _event(
        pull_requests=[_pull_request(20)],
        pull_request_files=[
            _pull_request_file(
                "src/app.py",
                additions=1,
                deletions=1,
                patch="@@ abbreviated GitHub patch",
            )
        ],
        github_commit_references=[
            _github_commit("abc123", "Use new call", url=True)
        ],
        local_commits=[
            _local_commit(
                "abc123",
                "Use new call",
                changed_files=[
                    _changed_file(
                        "src/app.py",
                        additions=1,
                        deletions=1,
                        patch=local_patch,
                    )
                ],
            )
        ],
    )

    document = _build(event)

    assert document.text.count("Use new call") == 1
    assert document.text.count("Path: src/app.py") == 1
    assert local_patch.rstrip("\n") in document.text
    assert "@@ abbreviated GitHub patch" not in document.text
    assert len(_sections(document, RetrievalSectionType.COMMIT)) == 1
    assert len(_sections(document, RetrievalSectionType.CHANGED_FILE)) == 1
    assert len(_sections(document, RetrievalSectionType.PATCH)) == 1
    assert {
        source.artifact.artifact_type
        for source in document.sources
        if source.artifact.identifier == "abc123"
    } == {ArtifactType.GITHUB_COMMIT_REFERENCE, ArtifactType.LOCAL_GIT_COMMIT}


def test_distinct_github_commit_information_is_retained_in_local_commit_section():
    event = _event(
        github_commit_references=[_github_commit("abc123", "GitHub message")],
        local_commits=[_local_commit("abc123", "Local message")],
    )

    commit_section = _sections(_build(event), RetrievalSectionType.COMMIT)[0]

    assert "Message:\nLocal message" in commit_section.content
    assert "GitHub Message:\nGitHub message" in commit_section.content


def test_section_ids_do_not_collide_for_shared_numbers_paths_or_commit_paths():
    event = _event(
        issues=[_issue(20)],
        pull_requests=[_pull_request(20)],
        local_commits=[
            _local_commit(
                "aaa111",
                "First",
                changed_files=[_changed_file("src/shared.py", patch="@@ first")],
            ),
            _local_commit(
                "bbb222",
                "Second",
                changed_files=[_changed_file("src/shared.py", patch="@@ second")],
            ),
        ],
    )

    document = _build(event)
    section_ids = [section.section_id for section in document.sections]

    assert len(section_ids) == len(set(section_ids))
    assert "issue-20" in section_ids
    assert "pull-request-20" in section_ids
    assert "patch-aaa111-src%2Fshared.py" in section_ids
    assert "patch-bbb222-src%2Fshared.py" in section_ids


def test_input_order_does_not_change_sections_text_or_sources():
    issues = [_issue(30), _issue(10), _issue(20)]
    pull_requests = [_pull_request(30), _pull_request(10), _pull_request(20)]
    comments = [
        _comment(103, 10, "Third", "alice", offset_minutes=2),
        _comment(101, 10, "First", "alice"),
        _comment(102, 10, "Second", "alice", offset_minutes=1),
    ]
    first_event = _event(
        issues=issues,
        issue_comments=comments,
        pull_requests=pull_requests,
    )
    second_event = _event(
        issues=list(reversed(issues)),
        issue_comments=list(reversed(comments)),
        pull_requests=list(reversed(pull_requests)),
    )

    first = _build(first_event)
    second = _build(second_event)

    assert first == second
    assert first.title == "PR #20 - PR 20"
    assert [
        section.artifact.identifier
        for section in _sections(first, RetrievalSectionType.ISSUE)
    ] == ["10", "20", "30"]
    assert [
        section.artifact.identifier
        for section in _sections(first, RetrievalSectionType.PULL_REQUEST)
    ] == ["10", "20", "30"]
    assert [
        section.artifact.identifier
        for section in _sections(first, RetrievalSectionType.ISSUE_COMMENT)
    ] == ["issue-comment:101", "issue-comment:102", "issue-comment:103"]


def test_builder_rejects_mismatched_metadata_identity():
    event = _event()
    builder = RetrievalDocumentBuilder()

    with pytest.raises(ValueError, match="does not belong"):
        builder.build(
            event,
            EventMetadata(event_id="other", repository=event.repository),
        )

    with pytest.raises(ValueError, match="repository"):
        builder.build(
            event,
            EventMetadata(
                event_id=event.event_id,
                repository=GitHubRepository(owner="other", name="repo"),
            ),
        )


def test_repository_isolation_rejects_foreign_event_records_before_rendering():
    foreign_issue = _issue(10).model_copy(
        update={"repository": GitHubRepository(owner="other", name="repo")}
    )

    with pytest.raises(ValidationError, match="one repository"):
        _event(issues=[foreign_issue])


def test_canonical_text_is_derived_from_sections_with_stable_whitespace():
    document = _build(
        _event(
            issues=[
                _issue(
                    10,
                    body="Line one\r\n\r\n  indented code\r\nLine four\r\n",
                )
            ]
        )
    )
    expected = "REPORECALL ENGINEERING EVENT\n\n" + "\n\n".join(
        f"=== {section.heading} ===\n\n{section.content}"
        for section in document.sections
    ) + "\n"

    assert document.text == expected
    assert "Line one\n\n  indented code\nLine four" in document.text
    assert "\r" not in document.text
    assert all(not line.endswith(" ") for line in document.text.splitlines())


def test_builder_has_no_io_retrieval_or_chunking_dependencies():
    module_attributes = vars(builder_module)

    assert "GitHubClient" not in module_attributes
    assert "httpx" not in module_attributes
    assert "git" not in module_attributes
    assert "RetrievalChunk" not in module_attributes
    assert "OpenAI" not in module_attributes
    assert "Embedding" not in module_attributes


def _build(event: EngineeringEvent):
    metadata = EventMetadataExtractor().extract(event)
    return RetrievalDocumentBuilder().build(event, metadata)


def _sections(document, section_type: RetrievalSectionType):
    return [
        section for section in document.sections if section.section_type is section_type
    ]


def _event(
    *,
    event_id: str = "github.com__owner__repo__pull_request__20",
    anchor: ArtifactReference | None = None,
    issues: list[GitHubIssue] | None = None,
    issue_comments: list[GitHubIssueComment] | None = None,
    pull_requests: list[GitHubPullRequest] | None = None,
    pull_request_comments: list[GitHubIssueComment] | None = None,
    pull_request_reviews: list[GitHubPullRequestReview] | None = None,
    pull_request_review_comments: list[GitHubPullRequestReviewComment] | None = None,
    pull_request_files: list[GitHubPullRequestFile] | None = None,
    github_commit_references: list[GitHubCommitReference] | None = None,
    local_commits: list[GitCommit] | None = None,
    relationships: list[EngineeringRelationship] | None = None,
    contextual_relationships: list[EngineeringRelationship] | None = None,
) -> EngineeringEvent:
    return EngineeringEvent(
        event_id=event_id,
        repository=_repository(),
        anchor=anchor or _ref(ArtifactType.PULL_REQUEST, "20"),
        issues=issues or [],
        issue_comments=issue_comments or [],
        pull_requests=pull_requests or [],
        pull_request_comments=pull_request_comments or [],
        pull_request_reviews=pull_request_reviews or [],
        pull_request_review_comments=pull_request_review_comments or [],
        pull_request_files=pull_request_files or [],
        github_commit_references=github_commit_references or [],
        local_commits=local_commits or [],
        relationships=relationships or [],
        contextual_relationships=contextual_relationships or [],
    )


def _issue(
    number: int,
    *,
    title: str | None = None,
    body: str | None = "Issue body",
    author: GitHubUser | None = None,
    labels: list[GitHubIssueLabel] | None = None,
) -> GitHubIssue:
    return GitHubIssue(
        repository=_repository(),
        number=number,
        title=title or f"Issue {number}",
        body=body,
        state=IssueState.OPEN,
        author=author,
        labels=labels or [],
        created_at=_timestamp(),
        updated_at=_timestamp(),
        closed_at=None,
        html_url=f"https://github.com/owner/repo/issues/{number}",
        comments_count=0,
        locked=False,
    )


def _pull_request(
    number: int,
    *,
    title: str | None = None,
    body: str | None = "Pull request body",
    author: GitHubUser | None = None,
) -> GitHubPullRequest:
    repository = _repository()
    return GitHubPullRequest(
        repository=repository,
        number=number,
        title=title or f"PR {number}",
        body=body,
        state=PullRequestState.CLOSED,
        author=author,
        labels=[],
        draft=False,
        locked=False,
        created_at=_timestamp(),
        updated_at=_timestamp(),
        closed_at=_timestamp(),
        merged_at=_timestamp(),
        html_url=f"https://github.com/owner/repo/pull/{number}",
        head=GitHubBranchReference(repository=repository, ref="feature", sha="head"),
        base=GitHubBranchReference(repository=repository, ref="main", sha="base"),
        merge_commit_sha="abc123",
    )


def _comment(
    comment_id: int,
    parent_number: int,
    body: str | None,
    author: str | None,
    *,
    offset_minutes: int = 0,
) -> GitHubIssueComment:
    timestamp = _timestamp() + timedelta(minutes=offset_minutes)
    return GitHubIssueComment(
        repository=_repository(),
        id=comment_id,
        issue_number=parent_number,
        author=GitHubUser(login=author) if author is not None else None,
        body=body,
        created_at=timestamp,
        updated_at=timestamp,
        html_url=(
            f"https://github.com/owner/repo/issues/{parent_number}"
            f"#issuecomment-{comment_id}"
        ),
    )


def _review(
    review_id: int,
    pull_request_number: int,
    body: str | None,
    author: str | None,
) -> GitHubPullRequestReview:
    return GitHubPullRequestReview(
        repository=_repository(),
        id=review_id,
        pull_request_number=pull_request_number,
        author=GitHubUser(login=author) if author is not None else None,
        body=body,
        state=ReviewState.APPROVED,
        submitted_at=_timestamp(),
        commit_sha="abc123",
        html_url=f"https://github.com/owner/repo/pull/{pull_request_number}#review-{review_id}",
    )


def _review_comment(
    comment_id: int,
    pull_request_number: int,
    review_id: int,
    body: str | None,
    author: str | None,
) -> GitHubPullRequestReviewComment:
    return GitHubPullRequestReviewComment(
        repository=_repository(),
        id=comment_id,
        pull_request_number=pull_request_number,
        review_id=review_id,
        author=GitHubUser(login=author) if author is not None else None,
        body=body,
        created_at=_timestamp(),
        updated_at=_timestamp(),
        html_url=(
            f"https://github.com/owner/repo/pull/{pull_request_number}"
            f"#discussion-{comment_id}"
        ),
        commit_sha="abc123",
        original_commit_sha="abc123",
        path="src/database.py",
        line=42,
        side="RIGHT",
    )


def _pull_request_file(
    path: str,
    *,
    additions: int = 3,
    deletions: int = 1,
    patch: str | None = None,
) -> GitHubPullRequestFile:
    return GitHubPullRequestFile(
        filename=path,
        status=GitHubPullRequestFileStatus.MODIFIED,
        additions=additions,
        deletions=deletions,
        changes=additions + deletions,
        patch=patch,
        raw_url=f"https://raw.githubusercontent.com/owner/repo/abc123/{path}",
        blob_url=f"https://github.com/owner/repo/blob/abc123/{path}",
    )


def _github_commit(
    sha: str,
    message: str,
    *,
    url: bool = False,
) -> GitHubCommitReference:
    return GitHubCommitReference(
        sha=sha,
        html_url=f"https://github.com/owner/repo/commit/{sha}" if url else None,
        message=message,
        author_name="Repo Tester",
        author_email="tester@example.com",
        authored_at=_timestamp(),
    )


def _local_commit(
    sha: str,
    message: str,
    *,
    changed_files: list[ChangedFile] | None = None,
) -> GitCommit:
    return GitCommit(
        sha=sha,
        message=message,
        author_name="Repo Tester",
        author_email="tester@example.com",
        authored_at=_timestamp(),
        committed_at=_timestamp(),
        parent_shas=[],
        changed_files=changed_files or [],
    )


def _changed_file(
    path: str,
    *,
    additions: int = 3,
    deletions: int = 1,
    patch: str | None,
) -> ChangedFile:
    return ChangedFile(
        path=path,
        change_type=FileChangeType.MODIFIED,
        additions=additions,
        deletions=deletions,
        patch=patch,
    )


def _relationship(
    source_type: ArtifactType,
    source_identifier: str,
    relationship_type: RelationshipType,
    target_type: ArtifactType,
    target_identifier: str,
    evidence_type: RelationshipEvidenceType,
    *,
    evidence: str | None = None,
    source_field: str | None = None,
) -> EngineeringRelationship:
    return EngineeringRelationship(
        source=_ref(source_type, source_identifier),
        target=_ref(target_type, target_identifier),
        relationship_type=relationship_type,
        evidence_type=evidence_type,
        evidence=evidence,
        source_field=source_field,
    )


def _ref(artifact_type: ArtifactType, identifier: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=artifact_type,
        repository=_repository(),
        identifier=identifier,
    )


def _repository() -> GitHubRepository:
    return GitHubRepository(owner="owner", name="repo")


def _timestamp() -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
