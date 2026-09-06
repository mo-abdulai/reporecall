import pytest
from pydantic import ValidationError

from reporecall.github import GitHubRepository


@pytest.mark.parametrize(
    ("value", "owner", "name"),
    [
        ("fastapi/fastapi", "fastapi", "fastapi"),
        ("https://github.com/fastapi/fastapi", "fastapi", "fastapi"),
        ("https://github.com/fastapi/fastapi.git", "fastapi", "fastapi"),
        ("git@github.com:fastapi/fastapi.git", "fastapi", "fastapi"),
    ],
)
def test_parse_repository_identifier(value: str, owner: str, name: str):
    repository = GitHubRepository.parse(value)

    assert repository.owner == owner
    assert repository.name == name


@pytest.mark.parametrize(
    "value",
    [
        "",
        "fastapi",
        "https://github.com/",
        "/",
        "https://gitlab.com/fastapi/fastapi",
        "owner/repo/extra",
    ],
)
def test_rejects_malformed_repository_identifier(value: str):
    with pytest.raises(ValueError):
        GitHubRepository.parse(value)


def test_repository_model_rejects_path_segments():
    with pytest.raises(ValidationError):
        GitHubRepository(owner="fastapi/core", name="fastapi")
