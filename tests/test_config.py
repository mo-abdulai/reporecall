from pathlib import Path

from reporecall.config import Settings


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_API_URL", raising=False)
    monkeypatch.delenv("REPO_DATA_DIR", raising=False)
    monkeypatch.delenv("CACHE_DIR", raising=False)

    settings = Settings(_env_file=None)

    assert settings.github_token is None
    assert settings.github_api_url == "https://api.github.com"
    assert settings.repo_data_dir == Path("data/repositories")
    assert settings.cache_dir == Path("data/cache")


def test_settings_environment_overrides(monkeypatch, tmp_path):
    repo_dir = tmp_path / "repositories"
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_API_URL", "https://github.example/api")
    monkeypatch.setenv("REPO_DATA_DIR", str(repo_dir))
    monkeypatch.setenv("CACHE_DIR", str(cache_dir))

    settings = Settings(_env_file=None)

    assert settings.github_token == "test-token"
    assert settings.github_api_url == "https://github.example/api"
    assert settings.repo_data_dir == repo_dir
    assert settings.cache_dir == cache_dir
