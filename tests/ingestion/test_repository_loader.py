from pathlib import Path

import pytest
from git import Actor, Repo

from reporecall.ingestion import (
    InvalidRepositoryError,
    InvalidRepositoryURLError,
    RepositoryCloneError,
    RepositoryLoader,
)
from tests.conftest import commit_all, write_file


def test_safe_directory_name_derivation():
    assert (
        RepositoryLoader._safe_directory_name("https://github.com/fastapi/fastapi.git")
        == "github-com__fastapi__fastapi"
    )
    assert (
        RepositoryLoader._safe_directory_name("https://github.com/fastapi/fastapi")
        == "github-com__fastapi__fastapi"
    )


def test_successful_local_clone(tmp_path: Path, git_actor: Actor):
    source = _create_source_repo(tmp_path, git_actor)
    loader = RepositoryLoader(base_dir=tmp_path / "clones")

    cloned_path = loader.clone_or_get(str(source.working_tree_dir))

    assert cloned_path.exists()
    assert (cloned_path / ".git").exists()
    assert Repo(cloned_path).head.commit.message == "initial commit"


def test_reuses_already_cloned_repository(tmp_path: Path, git_actor: Actor):
    source = _create_source_repo(tmp_path, git_actor)
    loader = RepositoryLoader(base_dir=tmp_path / "clones")
    first_path = loader.clone_or_get(str(source.working_tree_dir))

    second_path = loader.clone_or_get(str(source.working_tree_dir))

    assert second_path == first_path
    assert len(list((tmp_path / "clones").iterdir())) == 1


def test_update_fetches_remote_refs(tmp_path: Path, git_actor: Actor):
    source = _create_source_repo(tmp_path, git_actor)
    loader = RepositoryLoader(base_dir=tmp_path / "clones")
    cloned_path = loader.clone_or_get(str(source.working_tree_dir))
    write_file(source, "second.txt", "second\n")
    second_sha = commit_all(source, "second commit", git_actor)

    reused_path = loader.clone_or_get(str(source.working_tree_dir), update=True)

    assert reused_path == cloned_path
    assert Repo(cloned_path).commit("FETCH_HEAD").hexsha == second_sha


def test_invalid_repository_url_input_raises(tmp_path: Path):
    loader = RepositoryLoader(base_dir=tmp_path / "clones")

    with pytest.raises(InvalidRepositoryURLError):
        loader.clone_or_get("")


def test_clone_failure_raises_repository_clone_error(tmp_path: Path):
    loader = RepositoryLoader(base_dir=tmp_path / "clones")

    with pytest.raises(RepositoryCloneError):
        loader.clone_or_get(str(tmp_path / "does-not-exist"))


def test_existing_non_repository_destination_raises(tmp_path: Path):
    loader = RepositoryLoader(base_dir=tmp_path / "clones")
    destination = tmp_path / "clones" / RepositoryLoader._safe_directory_name("https://github.com/a/b.git")
    destination.mkdir(parents=True)

    with pytest.raises(InvalidRepositoryError):
        loader.clone_or_get("https://github.com/a/b.git")


def _create_source_repo(tmp_path: Path, actor: Actor) -> Repo:
    source_path = tmp_path / "source"
    repo = Repo.init(source_path)
    with repo.config_writer() as config:
        config.set_value("user", "name", actor.name)
        config.set_value("user", "email", actor.email)
    write_file(repo, "README.md", "# Source\n")
    commit_all(repo, "initial commit", actor)
    return repo
