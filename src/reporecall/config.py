from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    github_token: str | None = None
    github_api_url: str = "https://api.github.com"
    openai_api_key: str | None = None
    repo_data_dir: Path = Path("data/repositories")
    cache_dir: Path = Path("data/cache")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()


settings = get_settings()
