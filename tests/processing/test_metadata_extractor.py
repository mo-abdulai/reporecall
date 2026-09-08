from datetime import UTC, datetime

import reporecall.processing.metadata_extractor as metadata_extractor_module
from reporecall.github import GitHubRepository
from reporecall.models import (
    ArtifactReference,
    ArtifactType,
    ChangedFile,
    EngineeringEvent,
    EventActor,
    EventActorType,
    FileChangeType,
    GitCommit,
    GitHubBranchReference,
    GitHubCommitReference,
    GitHubIssue,
    GitHubIssueComment,
    GitHubIssueLabel,
    GitHubMilestone,
    GitHubPullRequest,
    GitHubPullRequestFile,
    GitHubPullRequestFileStatus,
    GitHubPullRequestReview,
    GitHubPullRequestReviewComment,
    GitHubUser,
    IssueState,
    PullRequestState,
    ReviewState,
)
from reporecall.processing import EventMetadataExtractor


def test_minimal_event_extracts_identity_without_mutating_source():
    event = _event(issues=[_issue(10)])
    original = event.model_dump()
    extractor = EventMetadataExtractor()

    first = extractor.extract(event)
    second = extractor.extract(event)

    assert first == second
    assert first.event_id == event.event_id
    assert first.repository == _repository()
    assert first.issue_numbers == (10,)
    assert first.pull_request_numbers == ()
    assert first.changed_paths == ()
    assert first.authors == ()
    assert first.participants == ()
    assert first.changed_lines == 0
    assert event.model_dump() == original


def test_people_labels_and_milestones_preserve_explicit_identities():
    alice = GitHubUser(login="alice")
    bob = GitHubUser(login="bob")
    carol = GitHubUser(login="carol")
    dave = GitHubUser(login="dave")
    event = _event(
        issues=[
            _issue(
                10,
                author=alice,
                labels=[_label("bug"), _label(" bug ")],
                milestone=_milestone(1, "Release 1"),
            )
        ],
        issue_comments=[_comment(100, 10, carol)],
        pull_requests=[
            _pull_request(
                20,
                author=bob,
                labels=[_label("documentation"), _label("bug")],
                milestone=_milestone(2, "Release 1"),
            )
        ],
        pull_request_comments=[_comment(101, 20, carol)],
        pull_request_reviews=[_review(200, 20, dave)],
        pull_request_review_comments=[_review_comment(300, 20, 200, carol)],
        github_commit_references=[_github_commit("abc123", "Git User", "git@example.com")],
        local_commits=[_local_commit("abc123", "Git User", "git@example.com")],
    )

    metadata = EventMetadataExtractor().extract(event)

    assert metadata.issue_numbers == (10,)
    assert metadata.pull_request_numbers == (20,)
    assert metadata.commit_shas == ("abc123",)
    assert metadata.labels == ("bug", "documentation")
    assert metadata.milestones == ("Release 1",)
    assert _actor_identities(metadata.authors) == (
        (EventActorType.GIT_AUTHOR, "git@example.com"),
        (EventActorType.GITHUB_USER, "alice"),
        (EventActorType.GITHUB_USER, "bob"),
    )
    assert _actor_identities(metadata.participants) == (
        (EventActorType.GIT_AUTHOR, "git@example.com"),
        (EventActorType.GITHUB_USER, "alice"),
        (EventActorType.GITHUB_USER, "bob"),
        (EventActorType.GITHUB_USER, "carol"),
        (EventActorType.GITHUB_USER, "dave"),
    )


def test_path_metadata_languages_and_classifications_are_deterministic():
    paths = [
        "web/widget.unknown",
        "tests/test_auth.py",
        "src/component.spec.ts",
        "src/api/client.tsx",
        "docs/guide.md",
        ".github/workflows/ci.yml",
        "pyproject.toml",
        "Dockerfile",
        "k8s/deploy.yaml",
        "./src\\api\\client.tsx",
    ]
    event = _event(
        pull_request_files=[_pull_request_file(path) for path in paths],
    )

    metadata = EventMetadataExtractor().extract(event)

    assert metadata.changed_paths == tuple(
        sorted(
            {
                ".github/workflows/ci.yml",
                "Dockerfile",
                "docs/guide.md",
                "k8s/deploy.yaml",
                "pyproject.toml",
                "src/api/client.tsx",
                "src/component.spec.ts",
                "tests/test_auth.py",
                "web/widget.unknown",
            },
            key=lambda value: (value.casefold(), value),
        )
    )
    assert metadata.file_extensions == (
        ".md",
        ".py",
        ".toml",
        ".ts",
        ".tsx",
        ".unknown",
        ".yaml",
        ".yml",
    )
    assert metadata.languages == ("Python", "TypeScript")
    assert metadata.directories == (
        ".github",
        ".github/workflows",
        "docs",
        "k8s",
        "src",
        "src/api",
        "tests",
        "web",
    )
    assert metadata.has_tests is True
    assert metadata.test_paths == ("src/component.spec.ts", "tests/test_auth.py")
    assert metadata.has_documentation_changes is True
    assert metadata.documentation_paths == ("docs/guide.md",)
    assert metadata.has_configuration_changes is True
    assert metadata.configuration_paths == (
        ".github/workflows/ci.yml",
        "Dockerfile",
        "k8s/deploy.yaml",
        "pyproject.toml",
    )
    assert metadata.has_dependency_changes is True
    assert metadata.dependency_paths == ("pyproject.toml",)


def test_detection_rules_cover_common_filename_conventions():
    paths = [
        "CHANGELOG.rst",
        "CONTRIBUTING",
        "README.md",
        "app/__tests__/widget.js",
        "cmd/server_test.go",
        "docker-compose.yml",
        "infra/main.tf",
        "package-lock.json",
        "requirements-dev.txt",
    ]
    metadata = EventMetadataExtractor().extract(
        _event(pull_request_files=[_pull_request_file(path) for path in paths])
    )

    assert metadata.test_paths == ("app/__tests__/widget.js", "cmd/server_test.go")
    assert metadata.documentation_paths == (
        "CHANGELOG.rst",
        "CONTRIBUTING",
        "README.md",
    )
    assert metadata.configuration_paths == (
        "docker-compose.yml",
        "infra/main.tf",
    )
    assert metadata.dependency_paths == (
        "package-lock.json",
        "requirements-dev.txt",
    )


def test_diff_statistics_prefer_pr_paths_and_keep_local_only_paths():
    shared_file = _changed_file("src/app.py", additions=6, deletions=2)
    local_only_file = _changed_file("src/local.py", additions=4, deletions=1)
    local_commit = _local_commit(
        "abc123",
        "Git User",
        "git@example.com",
        changed_files=[shared_file, local_only_file],
    )
    event = _event(
        pull_request_files=[
            _pull_request_file("src/app.py", additions=10, deletions=3),
        ],
        github_commit_references=[_github_commit("abc123", "Git User", "git@example.com")],
        local_commits=[local_commit, local_commit],
    )

    metadata = EventMetadataExtractor().extract(event)

    assert metadata.added_lines == 14
    assert metadata.deleted_lines == 4
    assert metadata.changed_lines == 18


def test_local_file_statistics_take_precedence_over_pr_aggregate_statistics():
    event = _event(
        pull_requests=[_pull_request(20, additions=40, deletions=20)],
        local_commits=[
            _local_commit(
                "abc123",
                "Git User",
                None,
                changed_files=[_changed_file("src/app.py", additions=4, deletions=1)],
            )
        ],
    )

    metadata = EventMetadataExtractor().extract(event)

    assert (metadata.added_lines, metadata.deleted_lines, metadata.changed_lines) == (4, 1, 5)


def test_pr_aggregate_statistics_are_used_as_a_final_fallback():
    event = _event(
        pull_requests=[
            _pull_request(20, additions=8, deletions=2),
            _pull_request(30, additions=None, deletions=None),
        ]
    )

    metadata = EventMetadataExtractor().extract(event)

    assert (metadata.added_lines, metadata.deleted_lines, metadata.changed_lines) == (8, 2, 10)


def test_metadata_extractor_has_no_network_or_git_dependencies():
    module_attributes = vars(metadata_extractor_module)

    assert "GitHubClient" not in module_attributes
    assert "httpx" not in module_attributes
    assert "git" not in module_attributes


def _event(
    *,
    issues: list[GitHubIssue] | None = None,
    issue_comments: list[GitHubIssueComment] | None = None,
    pull_requests: list[GitHubPullRequest] | None = None,
    pull_request_comments: list[GitHubIssueComment] | None = None,
    pull_request_reviews: list[GitHubPullRequestReview] | None = None,
    pull_request_review_comments: list[GitHubPullRequestReviewComment] | None = None,
    pull_request_files: list[GitHubPullRequestFile] | None = None,
    github_commit_references: list[GitHubCommitReference] | None = None,
    local_commits: list[GitCommit] | None = None,
) -> EngineeringEvent:
    return EngineeringEvent(
        event_id="github.com__owner__repo__pull_request__20",
        repository=_repository(),
        anchor=_reference(ArtifactType.PULL_REQUEST, "20"),
        issues=issues or [],
        issue_comments=issue_comments or [],
        pull_requests=pull_requests or [],
        pull_request_comments=pull_request_comments or [],
        pull_request_reviews=pull_request_reviews or [],
        pull_request_review_comments=pull_request_review_comments or [],
        pull_request_files=pull_request_files or [],
        github_commit_references=github_commit_references or [],
        local_commits=local_commits or [],
    )


def _issue(
    number: int,
    *,
    author: GitHubUser | None = None,
    labels: list[GitHubIssueLabel] | None = None,
    milestone: GitHubMilestone | None = None,
) -> GitHubIssue:
    return GitHubIssue(
        repository=_repository(),
        number=number,
        title=f"Issue {number}",
        body=None,
        state=IssueState.OPEN,
        author=author,
        labels=labels or [],
        milestone=milestone,
        created_at=_timestamp(),
        updated_at=_timestamp(),
        closed_at=None,
        html_url=f"https://github.com/owner/repo/issues/{number}",
        comments_count=0,
        locked=False,
    )


def _comment(comment_id: int, parent_number: int, author: GitHubUser) -> GitHubIssueComment:
    return GitHubIssueComment(
        repository=_repository(),
        id=comment_id,
        issue_number=parent_number,
        author=author,
        body="Comment",
        created_at=_timestamp(),
        updated_at=_timestamp(),
        html_url=f"https://github.com/owner/repo/issues/{parent_number}#issuecomment-{comment_id}",
    )


def _pull_request(
    number: int,
    *,
    author: GitHubUser | None = None,
    labels: list[GitHubIssueLabel] | None = None,
    milestone: GitHubMilestone | None = None,
    additions: int | None = None,
    deletions: int | None = None,
) -> GitHubPullRequest:
    repository = _repository()
    return GitHubPullRequest(
        repository=repository,
        number=number,
        title=f"PR {number}",
        body=None,
        state=PullRequestState.OPEN,
        author=author,
        labels=labels or [],
        milestone=milestone,
        draft=False,
        locked=False,
        created_at=_timestamp(),
        updated_at=_timestamp(),
        closed_at=None,
        merged_at=None,
        html_url=f"https://github.com/owner/repo/pull/{number}",
        head=GitHubBranchReference(repository=repository, ref="feature", sha="head"),
        base=GitHubBranchReference(repository=repository, ref="main", sha="base"),
        merge_commit_sha=None,
        additions=additions,
        deletions=deletions,
    )


def _review(review_id: int, pull_request_number: int, author: GitHubUser) -> GitHubPullRequestReview:
    return GitHubPullRequestReview(
        repository=_repository(),
        id=review_id,
        pull_request_number=pull_request_number,
        author=author,
        body="Review",
        state=ReviewState.APPROVED,
        submitted_at=_timestamp(),
        commit_sha="abc123",
        html_url=None,
    )


def _review_comment(
    comment_id: int,
    pull_request_number: int,
    review_id: int,
    author: GitHubUser,
) -> GitHubPullRequestReviewComment:
    return GitHubPullRequestReviewComment(
        repository=_repository(),
        id=comment_id,
        pull_request_number=pull_request_number,
        review_id=review_id,
        author=author,
        body="Review comment",
        created_at=_timestamp(),
        updated_at=_timestamp(),
        html_url=f"https://github.com/owner/repo/pull/{pull_request_number}#discussion-{comment_id}",
        commit_sha="abc123",
        original_commit_sha="abc123",
        path="src/app.py",
    )


def _pull_request_file(
    path: str,
    *,
    additions: int = 1,
    deletions: int = 1,
) -> GitHubPullRequestFile:
    return GitHubPullRequestFile(
        filename=path,
        status=GitHubPullRequestFileStatus.MODIFIED,
        additions=additions,
        deletions=deletions,
        changes=additions + deletions,
    )


def _github_commit(
    sha: str,
    author_name: str | None,
    author_email: str | None,
) -> GitHubCommitReference:
    return GitHubCommitReference(
        sha=sha,
        html_url=None,
        message="Commit",
        author_name=author_name,
        author_email=author_email,
        authored_at=_timestamp(),
    )


def _local_commit(
    sha: str,
    author_name: str,
    author_email: str | None,
    *,
    changed_files: list[ChangedFile] | None = None,
) -> GitCommit:
    return GitCommit(
        sha=sha,
        message="Commit",
        author_name=author_name,
        author_email=author_email,
        authored_at=_timestamp(),
        committed_at=_timestamp(),
        parent_shas=[],
        changed_files=changed_files or [],
    )


def _changed_file(path: str, *, additions: int, deletions: int) -> ChangedFile:
    return ChangedFile(
        path=path,
        change_type=FileChangeType.MODIFIED,
        additions=additions,
        deletions=deletions,
        patch="@@ patch",
    )


def _label(name: str) -> GitHubIssueLabel:
    return GitHubIssueLabel(name=name)


def _milestone(number: int, title: str) -> GitHubMilestone:
    return GitHubMilestone(number=number, title=title)


def _reference(artifact_type: ArtifactType, identifier: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=artifact_type,
        repository=_repository(),
        identifier=identifier,
    )


def _actor_identities(
    actors: tuple[EventActor, ...],
) -> tuple[tuple[EventActorType, str], ...]:
    return tuple((actor.actor_type, actor.identifier) for actor in actors)


def _repository() -> GitHubRepository:
    return GitHubRepository(owner="owner", name="repo")


def _timestamp() -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
