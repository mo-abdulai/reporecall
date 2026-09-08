import pytest
from pydantic import ValidationError

from reporecall.github import GitHubRepository
from reporecall.models import EventActor, EventActorType, EventMetadata


def test_event_metadata_is_immutable_and_uses_tuple_collections():
    actor = EventActor(
        actor_type=EventActorType.GITHUB_USER,
        identifier="octocat",
        name="octocat",
    )
    metadata = EventMetadata(
        event_id="github.com__owner__repo__issue__10",
        repository=_repository(),
        issue_numbers=[10],
        authors=[actor],
        participants=[actor],
    )

    assert metadata.issue_numbers == (10,)
    assert metadata.authors == (actor,)
    with pytest.raises(ValidationError, match="frozen"):
        metadata.event_id = "changed"


def test_event_metadata_validates_derived_statistics_and_flags():
    with pytest.raises(ValidationError, match="changed_lines"):
        EventMetadata(
            event_id="event",
            repository=_repository(),
            added_lines=2,
            deleted_lines=1,
            changed_lines=2,
        )

    with pytest.raises(ValidationError, match="has_tests"):
        EventMetadata(
            event_id="event",
            repository=_repository(),
            has_tests=True,
        )


def test_event_actor_keeps_github_and_git_namespaces_distinct():
    github_actor = EventActor(
        actor_type=EventActorType.GITHUB_USER,
        identifier="alex",
    )
    git_actor = EventActor(
        actor_type=EventActorType.GIT_AUTHOR,
        identifier="alex",
        name="alex",
    )

    assert github_actor != git_actor


def _repository() -> GitHubRepository:
    return GitHubRepository(owner="owner", name="repo")
