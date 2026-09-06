from collections.abc import Iterator
from pathlib import Path

import pytest
from git import Actor, Repo


@pytest.fixture
def git_actor() -> Actor:
    return Actor("RepoRecall Tester", "tester@example.com")


@pytest.fixture
def temp_repo(tmp_path: Path, git_actor: Actor) -> Iterator[Repo]:
    repo_path = tmp_path / "repo"
    repo = Repo.init(repo_path)
    with repo.config_writer() as config:
        config.set_value("user", "name", git_actor.name)
        config.set_value("user", "email", git_actor.email)
    yield repo


def write_file(repo: Repo, relative_path: str, content: str) -> Path:
    file_path = Path(repo.working_tree_dir or "") / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return file_path


def commit_all(repo: Repo, message: str, actor: Actor) -> str:
    repo.git.add(A=True)
    commit = repo.index.commit(message, author=actor, committer=actor)
    return commit.hexsha
