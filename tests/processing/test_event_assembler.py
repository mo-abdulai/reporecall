from datetime import UTC, datetime

import pytest

import reporecall.processing.event_assembler as event_assembler_module
from reporecall.github import GitHubRepository
from reporecall.models import (
    ArtifactReference,
    ArtifactType,
    ChangedFile,
    EngineeringRelationship,
    FileChangeType,
    GitCommit,
    GitHubBranchReference,
    GitHubCommitReference,
    GitHubIssue,
    GitHubIssueComment,
    GitHubPullRequest,
    GitHubPullRequestFile,
    GitHubPullRequestFileStatus,
    GitHubPullRequestReview,
    GitHubPullRequestReviewComment,
    IssueState,
    PullRequestState,
    RelationshipEvidenceType,
    RelationshipType,
    ReviewState,
)
from reporecall.processing import EngineeringEventAssembler, EngineeringEventInput


def test_full_event_assembly_contains_all_connected_records():
    changed_file = _changed_file("src/app.py")
    data = EngineeringEventInput(
        repository=_repository(),
        issues=[_issue(10)],
        issue_comments=[_comment(100, parent_number=10)],
        pull_requests=[_pull_request(20)],
        pull_request_comments=[_comment(101, parent_number=20)],
        pull_request_reviews=[_review(200, pull_request_number=20)],
        pull_request_review_comments=[
            _review_comment(300, pull_request_number=20, review_id=200)
        ],
        pull_request_files={20: [_pull_request_file("src/app.py")]},
        pull_request_commits={20: [_github_commit("abc123")]},
        local_commits=[_local_commit("abc123", changed_files=[changed_file])],
        relationships=[
            _relationship(
                _ref(ArtifactType.ISSUE, "10"),
                _ref(ArtifactType.ISSUE_COMMENT, "issue-comment:100"),
                RelationshipType.ISSUE_HAS_COMMENT,
            ),
            _relationship(
                _ref(ArtifactType.PULL_REQUEST, "20"),
                _ref(ArtifactType.ISSUE, "10"),
                RelationshipType.PULL_REQUEST_CLOSES_ISSUE,
            ),
            _relationship(
                _ref(ArtifactType.PULL_REQUEST, "20"),
                _ref(ArtifactType.PULL_REQUEST_COMMENT, "pr-comment:101"),
                RelationshipType.PULL_REQUEST_HAS_COMMENT,
            ),
            _relationship(
                _ref(ArtifactType.PULL_REQUEST, "20"),
                _ref(ArtifactType.PULL_REQUEST_REVIEW, "review:200"),
                RelationshipType.PULL_REQUEST_HAS_REVIEW,
            ),
            _relationship(
                _ref(ArtifactType.PULL_REQUEST_REVIEW, "review:200"),
                _ref(ArtifactType.PULL_REQUEST_REVIEW_COMMENT, "review-comment:300"),
                RelationshipType.REVIEW_HAS_COMMENT,
            ),
            _relationship(
                _ref(ArtifactType.PULL_REQUEST, "20"),
                _ref(ArtifactType.PULL_REQUEST_REVIEW_COMMENT, "review-comment:300"),
                RelationshipType.PULL_REQUEST_HAS_REVIEW_COMMENT,
            ),
            _relationship(
                _ref(ArtifactType.PULL_REQUEST, "20"),
                _ref(ArtifactType.PULL_REQUEST_FILE, "pr:20:file:src/app.py"),
                RelationshipType.PULL_REQUEST_CHANGES_FILE,
            ),
            _relationship(
                _ref(ArtifactType.PULL_REQUEST, "20"),
                _ref(ArtifactType.GITHUB_COMMIT_REFERENCE, "abc123"),
                RelationshipType.PULL_REQUEST_HAS_COMMIT,
            ),
            _relationship(
                _ref(ArtifactType.GITHUB_COMMIT_REFERENCE, "abc123"),
                _ref(ArtifactType.LOCAL_GIT_COMMIT, "abc123"),
                RelationshipType.GITHUB_COMMIT_MATCHES_LOCAL_COMMIT,
            ),
            _relationship(
                _ref(ArtifactType.LOCAL_GIT_COMMIT, "abc123"),
                _ref(ArtifactType.LOCAL_CHANGED_FILE, "abc123:src/app.py"),
                RelationshipType.LOCAL_COMMIT_CHANGES_FILE,
            ),
        ],
    )

    events = EngineeringEventAssembler().assemble(data)

    assert len(events) == 1
    event = events[0]
    assert event.event_id == "github.com__owner__repo__pull_request__20"
    assert event.anchor == _ref(ArtifactType.PULL_REQUEST, "20")
    assert event.issue_numbers == [10]
    assert event.pull_request_numbers == [20]
    assert event.commit_shas == ["abc123"]
    assert event.changed_paths == ["src/app.py"]
    assert [comment.id for comment in event.issue_comments] == [100]
    assert [comment.id for comment in event.pull_request_comments] == [101]
    assert [review.id for review in event.pull_request_reviews] == [200]
    assert [comment.id for comment in event.pull_request_review_comments] == [300]
    assert [file.filename for file in event.pull_request_files] == ["src/app.py"]
    assert [commit.sha for commit in event.github_commit_references] == ["abc123"]
    assert [commit.sha for commit in event.local_commits] == ["abc123"]
    assert len(event.relationships) == 10
    assert event.contextual_relationships == []


def test_weak_reference_does_not_merge_events_and_is_preserved_as_context():
    reference = _relationship(
        _ref(ArtifactType.PULL_REQUEST, "20"),
        _ref(ArtifactType.ISSUE, "10"),
        RelationshipType.PULL_REQUEST_REFERENCES_ISSUE,
    )
    data = EngineeringEventInput(
        repository=_repository(),
        issues=[_issue(10)],
        pull_requests=[_pull_request(20)],
        relationships=[reference],
    )

    events = EngineeringEventAssembler().assemble(data)

    assert len(events) == 2
    assert events[0].pull_request_numbers == [20]
    assert events[1].issue_numbers == [10]
    assert events[0].contextual_relationships == [reference]
    assert events[1].contextual_relationships == [reference]
    assert all(event.relationships == [] for event in events)


def test_closing_relationship_merges_pull_request_and_issue():
    events = EngineeringEventAssembler().assemble(
        EngineeringEventInput(
            repository=_repository(),
            issues=[_issue(10)],
            pull_requests=[_pull_request(20)],
            relationships=[
                _relationship(
                    _ref(ArtifactType.PULL_REQUEST, "20"),
                    _ref(ArtifactType.ISSUE, "10"),
                    RelationshipType.PULL_REQUEST_CLOSES_ISSUE,
                )
            ],
        )
    )

    assert len(events) == 1
    assert events[0].pull_request_numbers == [20]
    assert events[0].issue_numbers == [10]


def test_commit_closure_merges_local_commit_and_issue():
    events = EngineeringEventAssembler().assemble(
        EngineeringEventInput(
            repository=_repository(),
            issues=[_issue(10)],
            local_commits=[_local_commit("abc123")],
            relationships=[
                _relationship(
                    _ref(ArtifactType.LOCAL_GIT_COMMIT, "abc123"),
                    _ref(ArtifactType.ISSUE, "10"),
                    RelationshipType.COMMIT_CLOSES_ISSUE,
                )
            ],
        )
    )

    assert len(events) == 1
    assert events[0].issue_numbers == [10]
    assert events[0].commit_shas == ["abc123"]
    assert events[0].anchor.artifact_type is ArtifactType.ISSUE


def test_commit_identity_keeps_github_and_local_records_in_one_event():
    events = EngineeringEventAssembler().assemble(
        EngineeringEventInput(
            repository=_repository(),
            pull_request_commits={99: [_github_commit("abc123")]},
            local_commits=[_local_commit("abc123")],
            relationships=[
                _relationship(
                    _ref(ArtifactType.GITHUB_COMMIT_REFERENCE, "abc123"),
                    _ref(ArtifactType.LOCAL_GIT_COMMIT, "abc123"),
                    RelationshipType.GITHUB_COMMIT_MATCHES_LOCAL_COMMIT,
                )
            ],
        )
    )

    assert len(events) == 1
    assert len(events[0].github_commit_references) == 1
    assert len(events[0].local_commits) == 1
    assert events[0].anchor.artifact_type is ArtifactType.LOCAL_GIT_COMMIT


def test_pull_request_commit_identity_chain_forms_one_event_without_transitive_edges():
    relationships = [
        _relationship(
            _ref(ArtifactType.PULL_REQUEST, "20"),
            _ref(ArtifactType.GITHUB_COMMIT_REFERENCE, "abc123"),
            RelationshipType.PULL_REQUEST_HAS_COMMIT,
        ),
        _relationship(
            _ref(ArtifactType.GITHUB_COMMIT_REFERENCE, "abc123"),
            _ref(ArtifactType.LOCAL_GIT_COMMIT, "abc123"),
            RelationshipType.GITHUB_COMMIT_MATCHES_LOCAL_COMMIT,
        ),
    ]
    events = EngineeringEventAssembler().assemble(
        EngineeringEventInput(
            repository=_repository(),
            pull_requests=[_pull_request(20)],
            pull_request_commits={20: [_github_commit("abc123")]},
            local_commits=[_local_commit("abc123")],
            relationships=relationships,
        )
    )

    assert len(events) == 1
    assert events[0].pull_request_numbers == [20]
    assert events[0].commit_shas == ["abc123"]
    assert len(events[0].relationships) == 2
    assert {
        relationship.identity_key for relationship in events[0].relationships
    } == {relationship.identity_key for relationship in relationships}


def test_connected_component_supports_multiple_issues_and_pull_requests():
    relationships = [
        _relationship(
            _ref(ArtifactType.PULL_REQUEST, "20"),
            _ref(ArtifactType.ISSUE, "10"),
            RelationshipType.PULL_REQUEST_CLOSES_ISSUE,
        ),
        _relationship(
            _ref(ArtifactType.PULL_REQUEST, "20"),
            _ref(ArtifactType.ISSUE, "11"),
            RelationshipType.PULL_REQUEST_CLOSES_ISSUE,
        ),
        _relationship(
            _ref(ArtifactType.PULL_REQUEST, "30"),
            _ref(ArtifactType.ISSUE, "10"),
            RelationshipType.PULL_REQUEST_CLOSES_ISSUE,
        ),
    ]

    events = EngineeringEventAssembler().assemble(
        EngineeringEventInput(
            repository=_repository(),
            issues=[_issue(11), _issue(10)],
            pull_requests=[_pull_request(30), _pull_request(20)],
            relationships=relationships,
        )
    )

    assert len(events) == 1
    assert events[0].issue_numbers == [10, 11]
    assert events[0].pull_request_numbers == [20, 30]
    assert events[0].anchor.identifier == "20"


@pytest.mark.parametrize(
    ("record_type", "anchor_type", "identifier"),
    [
        ("issue", ArtifactType.ISSUE, "10"),
        ("pull_request", ArtifactType.PULL_REQUEST, "20"),
        ("local_commit", ArtifactType.LOCAL_GIT_COMMIT, "abc123"),
        ("github_commit", ArtifactType.GITHUB_COMMIT_REFERENCE, "abc123"),
    ],
)
def test_isolated_core_artifact_forms_standalone_event(
    record_type: str,
    anchor_type: ArtifactType,
    identifier: str,
):
    inputs = {
        "issue": EngineeringEventInput(repository=_repository(), issues=[_issue(10)]),
        "pull_request": EngineeringEventInput(
            repository=_repository(),
            pull_requests=[_pull_request(20)],
        ),
        "local_commit": EngineeringEventInput(
            repository=_repository(),
            local_commits=[_local_commit("abc123")],
        ),
        "github_commit": EngineeringEventInput(
            repository=_repository(),
            pull_request_commits={99: [_github_commit("abc123")]},
        ),
    }

    data = inputs[record_type]
    events = EngineeringEventAssembler().assemble(data)

    assert len(events) == 1
    assert events[0].anchor.artifact_type is anchor_type
    assert events[0].anchor.identifier == identifier


def test_orphan_secondary_records_do_not_form_events():
    data = EngineeringEventInput(
        repository=_repository(),
        issue_comments=[_comment(100, parent_number=10)],
        pull_request_comments=[_comment(101, parent_number=20)],
        pull_request_reviews=[_review(200, pull_request_number=20)],
        pull_request_review_comments=[
            _review_comment(300, pull_request_number=20, review_id=200)
        ],
        pull_request_files={20: [_pull_request_file("src/app.py")]},
    )

    assert EngineeringEventAssembler().assemble(data) == []


def test_same_filename_across_commits_does_not_merge_events():
    commits = [
        _local_commit("aaa111", changed_files=[_changed_file("src/app.py")]),
        _local_commit("bbb222", changed_files=[_changed_file("src/app.py")]),
    ]
    relationships = [
        _relationship(
            _ref(ArtifactType.LOCAL_GIT_COMMIT, commit.sha),
            _ref(ArtifactType.LOCAL_CHANGED_FILE, f"{commit.sha}:src/app.py"),
            RelationshipType.LOCAL_COMMIT_CHANGES_FILE,
        )
        for commit in commits
    ]

    events = EngineeringEventAssembler().assemble(
        EngineeringEventInput(
            repository=_repository(),
            local_commits=commits,
            relationships=relationships,
        )
    )

    assert len(events) == 2
    assert [event.commit_shas for event in events] == [["aaa111"], ["bbb222"]]


def test_same_issue_number_across_repositories_remains_separate():
    repo_a = GitHubRepository(owner="owner", name="repo-a")
    repo_b = GitHubRepository(owner="owner", name="repo-b")

    events = EngineeringEventAssembler().assemble(
        EngineeringEventInput(
            repository=repo_a,
            issues=[_issue(10, repository=repo_b), _issue(10, repository=repo_a)],
        )
    )

    assert len(events) == 2
    assert [event.repository for event in events] == [repo_a, repo_b]
    assert events[0].event_id != events[1].event_id


def test_same_sha_in_separate_repository_inputs_has_distinct_identity():
    repo_a = GitHubRepository(owner="owner", name="repo-a")
    repo_b = GitHubRepository(owner="owner", name="repo-b")
    assembler = EngineeringEventAssembler()

    event_a = assembler.assemble(
        EngineeringEventInput(repository=repo_a, local_commits=[_local_commit("abc123")])
    )[0]
    event_b = assembler.assemble(
        EngineeringEventInput(repository=repo_b, local_commits=[_local_commit("abc123")])
    )[0]

    assert event_a.anchor.identifier == event_b.anchor.identifier
    assert event_a.repository != event_b.repository
    assert event_a.event_id != event_b.event_id


def test_input_relationship_and_duplicate_order_do_not_change_output():
    issue = _issue(10)
    pull_request = _pull_request(20)
    closing = _relationship(
        _ref(ArtifactType.PULL_REQUEST, "20"),
        _ref(ArtifactType.ISSUE, "10"),
        RelationshipType.PULL_REQUEST_CLOSES_ISSUE,
    )
    contextual = _relationship(
        _ref(ArtifactType.PULL_REQUEST, "20"),
        _ref(ArtifactType.ISSUE, "10"),
        RelationshipType.PULL_REQUEST_REFERENCES_ISSUE,
    )
    assembler = EngineeringEventAssembler()

    first = assembler.assemble(
        EngineeringEventInput(
            repository=_repository(),
            issues=[issue, issue],
            pull_requests=[pull_request, pull_request],
            relationships=[closing, contextual, closing],
        )
    )
    second = assembler.assemble(
        EngineeringEventInput(
            repository=_repository(),
            issues=[issue, issue],
            pull_requests=[pull_request, pull_request],
            relationships=[closing, closing, contextual],
        )
    )

    assert first == second
    assert len(first[0].issues) == 1
    assert len(first[0].pull_requests) == 1
    assert first[0].relationships == [closing]
    assert first[0].contextual_relationships == [contextual]


def test_anchor_priority_and_event_id_are_deterministic():
    issue_commit_relationship = _relationship(
        _ref(ArtifactType.LOCAL_GIT_COMMIT, "abc123"),
        _ref(ArtifactType.ISSUE, "10"),
        RelationshipType.COMMIT_CLOSES_ISSUE,
    )
    pr_issue_relationship = _relationship(
        _ref(ArtifactType.PULL_REQUEST, "20"),
        _ref(ArtifactType.ISSUE, "10"),
        RelationshipType.PULL_REQUEST_CLOSES_ISSUE,
    )
    data = EngineeringEventInput(
        repository=_repository(),
        issues=[_issue(10)],
        pull_requests=[_pull_request(20)],
        local_commits=[_local_commit("abc123")],
        relationships=[issue_commit_relationship, pr_issue_relationship],
    )
    assembler = EngineeringEventAssembler()

    first = assembler.assemble(data)
    second = assembler.assemble(data)

    assert first == second
    assert first[0].anchor == _ref(ArtifactType.PULL_REQUEST, "20")
    assert first[0].event_id == "github.com__owner__repo__pull_request__20"


def test_missing_relationship_endpoint_is_ignored_for_membership():
    events = EngineeringEventAssembler().assemble(
        EngineeringEventInput(
            repository=_repository(),
            issues=[_issue(10)],
            relationships=[
                _relationship(
                    _ref(ArtifactType.PULL_REQUEST, "999"),
                    _ref(ArtifactType.ISSUE, "10"),
                    RelationshipType.PULL_REQUEST_CLOSES_ISSUE,
                )
            ],
        )
    )

    assert len(events) == 1
    assert events[0].issue_numbers == [10]
    assert events[0].relationships == []


def test_cross_repository_strong_relationship_remains_contextual():
    repo_a = _repository()
    repo_b = GitHubRepository(owner="other", name="repo")
    relationship = _relationship(
        _ref(ArtifactType.PULL_REQUEST, "20", repository=repo_a),
        _ref(ArtifactType.ISSUE, "10", repository=repo_b),
        RelationshipType.PULL_REQUEST_CLOSES_ISSUE,
    )

    events = EngineeringEventAssembler().assemble(
        EngineeringEventInput(
            repository=repo_a,
            issues=[_issue(10, repository=repo_b)],
            pull_requests=[_pull_request(20, repository=repo_a)],
            relationships=[relationship],
        )
    )

    assert len(events) == 2
    assert all(event.relationships == [] for event in events)
    assert all(event.contextual_relationships == [relationship] for event in events)


def test_event_assembler_has_no_network_or_git_dependencies():
    module_attributes = vars(event_assembler_module)

    assert "GitHubClient" not in module_attributes
    assert "httpx" not in module_attributes
    assert "git" not in module_attributes


def _relationship(
    source: ArtifactReference,
    target: ArtifactReference,
    relationship_type: RelationshipType,
) -> EngineeringRelationship:
    return EngineeringRelationship(
        source=source,
        target=target,
        relationship_type=relationship_type,
        evidence_type=RelationshipEvidenceType.PARENT_IDENTIFIER,
        evidence=f"{source.identifier} -> {target.identifier}",
    )


def _ref(
    artifact_type: ArtifactType,
    identifier: str,
    *,
    repository: GitHubRepository | None = None,
) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=artifact_type,
        repository=repository or _repository(),
        identifier=identifier,
    )


def _repository() -> GitHubRepository:
    return GitHubRepository(owner="owner", name="repo")


def _issue(
    number: int,
    *,
    repository: GitHubRepository | None = None,
) -> GitHubIssue:
    repository = repository or _repository()
    return GitHubIssue(
        repository=repository,
        number=number,
        title=f"Issue {number}",
        body=None,
        state=IssueState.OPEN,
        author=None,
        labels=[],
        created_at=_timestamp(number),
        updated_at=_timestamp(number),
        closed_at=None,
        html_url=f"https://github.com/{repository.owner}/{repository.name}/issues/{number}",
        comments_count=0,
        locked=False,
    )


def _comment(comment_id: int, *, parent_number: int) -> GitHubIssueComment:
    return GitHubIssueComment(
        repository=_repository(),
        id=comment_id,
        issue_number=parent_number,
        author=None,
        body="Comment",
        created_at=_timestamp(1),
        updated_at=_timestamp(1),
        html_url="https://github.com/owner/repo/issues/10#issuecomment",
    )


def _pull_request(
    number: int,
    *,
    repository: GitHubRepository | None = None,
) -> GitHubPullRequest:
    repository = repository or _repository()
    return GitHubPullRequest(
        repository=repository,
        number=number,
        title=f"PR {number}",
        body=None,
        state=PullRequestState.OPEN,
        author=None,
        labels=[],
        draft=False,
        locked=False,
        created_at=_timestamp(number),
        updated_at=_timestamp(number),
        closed_at=None,
        merged_at=None,
        html_url=f"https://github.com/{repository.owner}/{repository.name}/pull/{number}",
        head=GitHubBranchReference(repository=repository, ref="feature", sha="headsha"),
        base=GitHubBranchReference(repository=repository, ref="main", sha="basesha"),
        merge_commit_sha=None,
    )


def _review(review_id: int, *, pull_request_number: int) -> GitHubPullRequestReview:
    return GitHubPullRequestReview(
        repository=_repository(),
        id=review_id,
        pull_request_number=pull_request_number,
        author=None,
        body="Review",
        state=ReviewState.APPROVED,
        submitted_at=_timestamp(2),
        commit_sha="abc123",
        html_url=None,
    )


def _review_comment(
    comment_id: int,
    *,
    pull_request_number: int,
    review_id: int,
) -> GitHubPullRequestReviewComment:
    return GitHubPullRequestReviewComment(
        repository=_repository(),
        id=comment_id,
        pull_request_number=pull_request_number,
        review_id=review_id,
        author=None,
        body="Review comment",
        created_at=_timestamp(3),
        updated_at=_timestamp(3),
        html_url="https://github.com/owner/repo/pull/20#discussion",
        commit_sha="abc123",
        original_commit_sha="abc123",
        path="src/app.py",
    )


def _pull_request_file(path: str) -> GitHubPullRequestFile:
    return GitHubPullRequestFile(
        filename=path,
        status=GitHubPullRequestFileStatus.MODIFIED,
        additions=1,
        deletions=1,
        changes=2,
    )


def _github_commit(sha: str) -> GitHubCommitReference:
    return GitHubCommitReference(
        sha=sha,
        html_url=None,
        message="Commit",
        author_name=None,
        author_email=None,
        authored_at=_timestamp(4),
    )


def _local_commit(
    sha: str,
    *,
    changed_files: list[ChangedFile] | None = None,
) -> GitCommit:
    return GitCommit(
        sha=sha,
        message="Commit",
        author_name="Repo Tester",
        author_email=None,
        authored_at=_timestamp(4),
        committed_at=_timestamp(4),
        parent_shas=[],
        changed_files=changed_files or [],
    )


def _changed_file(path: str) -> ChangedFile:
    return ChangedFile(
        path=path,
        change_type=FileChangeType.MODIFIED,
        additions=1,
        deletions=1,
        patch="@@ -1 +1 @@",
    )


def _timestamp(day: int) -> datetime:
    return datetime(2026, 1, min(day, 28), 12, 0, tzinfo=UTC)
