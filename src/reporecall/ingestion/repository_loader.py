from pathlib import Path
from urllib.parse import urlparse

from git import GitCommandError, Repo
from git import InvalidGitRepositoryError as GitPythonInvalidRepositoryError

from reporecall.config import settings
from reporecall.ingestion.git_loader import InvalidRepositoryError


class InvalidRepositoryURLError(ValueError):
    """Raised when a repository URL or path cannot be used for cloning."""


class RepositoryCloneError(RuntimeError):
    """Raised when cloning a repository fails."""


class RepositoryUpdateError(RuntimeError):
    """Raised when fetching repository updates fails."""


class RepositoryLoader:
    """Acquire and reuse local Git repository clones."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else settings.repo_data_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def clone_or_get(self, repository_url: str, *, update: bool = False) -> Path:
        """Clone a repository if missing, otherwise return the existing local clone."""

        self._validate_repository_url(repository_url)
        destination = self.base_dir / self._safe_directory_name(repository_url)

        if destination.exists():
            self._validate_existing_repository(destination)
            if update:
                self._fetch_updates(destination)
            return destination

        try:
            Repo.clone_from(repository_url, destination)
        except GitCommandError as exc:
            raise RepositoryCloneError(
                f"Failed to clone repository into {destination}."
            ) from exc

        return destination

    @staticmethod
    def _safe_directory_name(repository_url: str) -> str:
        parsed = urlparse(repository_url)
        if parsed.scheme and parsed.netloc:
            parts = [part for part in parsed.path.strip("/").split("/") if part]
            host = parsed.netloc
        else:
            path = Path(repository_url).expanduser()
            parts = [path.name]
            host = path.parent.name if path.parent.name else "local"

        if not parts:
            raise InvalidRepositoryURLError("Repository URL does not contain a repository name.")

        parts[-1] = parts[-1].removesuffix(".git")

        relevant_parts = [host, *parts[-2:]]
        safe_parts = [_sanitize_path_part(part) for part in relevant_parts if part]
        return "__".join(safe_parts)

    @staticmethod
    def _validate_repository_url(repository_url: str) -> None:
        if not repository_url or not repository_url.strip():
            raise InvalidRepositoryURLError("Repository URL must not be empty.")

        parsed = urlparse(repository_url)
        if parsed.scheme and parsed.scheme not in {"file", "git", "http", "https", "ssh"}:
            raise InvalidRepositoryURLError(
                f"Unsupported repository URL scheme: {parsed.scheme}"
            )

    @staticmethod
    def _validate_existing_repository(destination: Path) -> None:
        try:
            Repo(destination)
        except (GitCommandError, GitPythonInvalidRepositoryError) as exc:
            raise InvalidRepositoryError(
                f"Existing destination is not a valid Git repository: {destination}"
            ) from exc

    @staticmethod
    def _fetch_updates(destination: Path) -> None:
        repo = Repo(destination)
        try:
            repo.remotes.origin.fetch()
        except (AttributeError, GitCommandError) as exc:
            raise RepositoryUpdateError(
                f"Failed to fetch updates for repository at {destination}."
            ) from exc


def _sanitize_path_part(value: str) -> str:
    safe_chars = [character if character.isalnum() else "-" for character in value]
    cleaned = "".join(safe_chars).strip("-").lower()
    return cleaned or "repository"
