from pathlib import Path

import pytest
from git import Actor, Repo

from reporecall.ingestion import (
    CommitNotFoundError,
    GitLoader,
    InvalidCommitLimitError,
    InvalidRepositoryError,
)
from reporecall.models import FileChangeType, GitCommit
from tests.conftest import commit_all, write_file


def test_valid_repository_loads(temp_repo: Repo):
    loader = GitLoader(Path(temp_repo.working_tree_dir or ""))

    assert loader.repo_path == Path(temp_repo.working_tree_dir or "")


def test_missing_repository_path_raises(tmp_path: Path):
    with pytest.raises(InvalidRepositoryError):
        GitLoader(tmp_path / "missing")


def test_normal_directory_that_is_not_git_repository_raises(tmp_path: Path):
    directory = tmp_path / "not-git"
    directory.mkdir()

    with pytest.raises(InvalidRepositoryError):
        GitLoader(directory)


def test_get_commits_returns_newest_first_and_supports_limit(
    temp_repo: Repo,
    git_actor: Actor,
):
    first_sha = _commit_file(temp_repo, "first.txt", "first\n", "first commit", git_actor)
    second_sha = _commit_file(temp_repo, "second.txt", "second\n", "second commit", git_actor)

    commits = GitLoader(Path(temp_repo.working_tree_dir or "")).get_commits()

    assert [commit.sha for commit in commits] == [second_sha, first_sha]
    assert [commit.sha for commit in GitLoader(Path(temp_repo.working_tree_dir or "")).get_commits(limit=1)] == [
        second_sha
    ]


@pytest.mark.parametrize("limit", [0, -1])
def test_get_commits_rejects_invalid_limit(temp_repo: Repo, limit: int):
    with pytest.raises(InvalidCommitLimitError):
        GitLoader(Path(temp_repo.working_tree_dir or "")).get_commits(limit=limit)


def test_commit_metadata(temp_repo: Repo, git_actor: Actor):
    root_sha = _commit_file(temp_repo, "root.txt", "root\n", "root commit", git_actor)
    child_sha = _commit_file(temp_repo, "child.txt", "child\n", "child commit", git_actor)

    commit = GitLoader(Path(temp_repo.working_tree_dir or "")).get_commit(child_sha)

    assert isinstance(commit, GitCommit)
    assert commit.sha == child_sha
    assert commit.message == "child commit"
    assert commit.author_name == "RepoRecall Tester"
    assert commit.author_email == "tester@example.com"
    assert commit.authored_at.tzinfo is not None
    assert commit.committed_at.tzinfo is not None
    assert commit.parent_shas == [root_sha]


def test_root_commit_includes_added_files(temp_repo: Repo, git_actor: Actor):
    root_sha = _commit_file(temp_repo, "root.txt", "root\n", "root commit", git_actor)

    commit = GitLoader(Path(temp_repo.working_tree_dir or "")).get_commit(root_sha)

    assert len(commit.changed_files) == 1
    changed_file = commit.changed_files[0]
    assert changed_file.path == "root.txt"
    assert changed_file.change_type == FileChangeType.ADDED
    assert changed_file.additions == 1
    assert changed_file.deletions == 0


def test_added_modified_deleted_file_changes(temp_repo: Repo, git_actor: Actor):
    _commit_file(temp_repo, "tracked.txt", "one\n", "add file", git_actor)
    write_file(temp_repo, "tracked.txt", "one\ntwo\n")
    modified_sha = commit_all(temp_repo, "modify file", git_actor)
    (Path(temp_repo.working_tree_dir or "") / "tracked.txt").unlink()
    deleted_sha = commit_all(temp_repo, "delete file", git_actor)

    loader = GitLoader(Path(temp_repo.working_tree_dir or ""))
    modified = loader.get_commit(modified_sha).changed_files[0]
    deleted = loader.get_commit(deleted_sha).changed_files[0]

    assert modified.path == "tracked.txt"
    assert modified.change_type == FileChangeType.MODIFIED
    assert modified.additions == 1
    assert modified.deletions == 0
    assert modified.patch is not None
    assert "+two" in modified.patch

    assert deleted.path == "tracked.txt"
    assert deleted.change_type == FileChangeType.DELETED
    assert deleted.additions == 0
    assert deleted.deletions == 2


def test_rename_file_change_preserves_old_and_new_path(temp_repo: Repo, git_actor: Actor):
    _commit_file(temp_repo, "old.txt", "same\ncontent\n", "add file", git_actor)
    temp_repo.git.mv("old.txt", "new.txt")
    rename_sha = commit_all(temp_repo, "rename file", git_actor)

    changed_file = GitLoader(Path(temp_repo.working_tree_dir or "")).get_commit(rename_sha).changed_files[0]

    assert changed_file.change_type == FileChangeType.RENAMED
    assert changed_file.path == "new.txt"
    assert changed_file.old_path == "old.txt"


def test_get_commit_accepts_abbreviated_sha(temp_repo: Repo, git_actor: Actor):
    sha = _commit_file(temp_repo, "tracked.txt", "content\n", "add file", git_actor)

    commit = GitLoader(Path(temp_repo.working_tree_dir or "")).get_commit(sha[:8])

    assert commit.sha == sha


def test_get_commit_invalid_revision_raises(temp_repo: Repo):
    with pytest.raises(CommitNotFoundError):
        GitLoader(Path(temp_repo.working_tree_dir or "")).get_commit("not-a-commit")


def _commit_file(
    repo: Repo,
    path: str,
    content: str,
    message: str,
    actor: Actor,
) -> str:
    write_file(repo, path, content)
    return commit_all(repo, message, actor)
