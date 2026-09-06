from datetime import datetime


class GitHubAPIError(RuntimeError):
    """Base exception for GitHub API failures."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        endpoint: str | None = None,
        reset_at: datetime | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.endpoint = endpoint
        self.reset_at = reset_at


class GitHubAuthenticationError(GitHubAPIError):
    """Raised when GitHub rejects authentication."""


class GitHubNotFoundError(GitHubAPIError):
    """Raised when GitHub cannot find the requested resource."""


class GitHubRateLimitError(GitHubAPIError):
    """Raised when GitHub reports that the caller is rate limited."""


class GitHubResponseError(GitHubAPIError):
    """Raised when GitHub returns an invalid or unexpected response."""


class GitHubNetworkError(GitHubAPIError):
    """Raised when an HTTP request cannot reach GitHub."""
