from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Self
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel

from reporecall.config import settings
from reporecall.github.exceptions import (
    GitHubAPIError,
    GitHubAuthenticationError,
    GitHubNetworkError,
    GitHubNotFoundError,
    GitHubRateLimitError,
    GitHubResponseError,
)
from reporecall.github.models import GitHubRepository

type JsonValue = dict[str, object] | list[object] | str | int | float | bool | None
type QueryParamValue = str | int | float | bool | None


class GitHubRateLimit(BaseModel):
    """Rate-limit metadata observed from a GitHub response."""

    limit: int | None = None
    remaining: int | None = None
    used: int | None = None
    reset_at: datetime | None = None
    resource: str | None = None


class GitHubClient:
    """Small HTTPX-backed client for GitHub REST API requests."""

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.token = token if token is not None else settings.github_token
        self.base_url = (base_url or settings.github_api_url).rstrip("/") + "/"
        self.rate_limit: GitHubRateLimit | None = None
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self._client.close()

    def get(
        self,
        path: str,
        *,
        params: Mapping[str, QueryParamValue] | None = None,
    ) -> JsonValue:
        """Perform a GitHub GET request and return decoded JSON data."""

        response = self._send_get(path, params=params)
        return self._decode_json(response, endpoint=path)

    def paginate(
        self,
        path: str,
        *,
        params: Mapping[str, QueryParamValue] | None = None,
        per_page: int = 100,
        max_pages: int | None = None,
    ) -> list[dict[str, object]]:
        """Return all item objects from a paginated GitHub list endpoint."""

        if per_page <= 0:
            raise ValueError("per_page must be a positive integer.")
        if max_pages is not None and max_pages <= 0:
            raise ValueError("max_pages must be a positive integer.")

        page_params = dict(params or {})
        page_params["per_page"] = per_page
        next_url: str | None = path
        items: list[dict[str, object]] = []
        pages_seen = 0
        seen_urls: set[str] = set()

        while next_url is not None:
            if max_pages is not None and pages_seen >= max_pages:
                break

            request_url = self._request_url(next_url)
            if request_url in seen_urls:
                raise GitHubResponseError(
                    "GitHub pagination produced a repeated next URL.",
                    endpoint=path,
                )
            seen_urls.add(request_url)

            response = self._send_get(next_url, params=page_params if pages_seen == 0 else None)
            data = self._decode_json(response, endpoint=next_url)
            if not isinstance(data, list):
                raise GitHubResponseError(
                    "GitHub paginated response was not a JSON array.",
                    status_code=response.status_code,
                    endpoint=next_url,
                )

            for item in data:
                if not isinstance(item, dict):
                    raise GitHubResponseError(
                        "GitHub paginated response item was not a JSON object.",
                        status_code=response.status_code,
                        endpoint=next_url,
                    )
                items.append(dict(item))

            pages_seen += 1
            next_url = response.links.get("next", {}).get("url")

        return items

    def get_repository(self, repository: GitHubRepository) -> dict[str, object]:
        """Return raw GitHub repository JSON for an owner/name identifier."""

        data = self.get(f"/repos/{repository.owner}/{repository.name}")
        if not isinstance(data, dict):
            raise GitHubResponseError("GitHub repository response was not a JSON object.")
        return data

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "RepoRecall",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _send_get(
        self,
        path: str,
        *,
        params: Mapping[str, QueryParamValue] | None = None,
    ) -> httpx.Response:
        try:
            response = self._client.get(self._request_url(path), params=params)
        except httpx.RequestError as exc:
            raise GitHubNetworkError(f"GitHub request failed for endpoint: {path}") from exc

        self.rate_limit = self._rate_limit_from_headers(response.headers)
        self._raise_for_status(response, endpoint=path)
        return response

    def _request_url(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path
        return urljoin(self.base_url, path.lstrip("/"))

    def _raise_for_status(self, response: httpx.Response, *, endpoint: str) -> None:
        if response.status_code < 400:
            return

        message = self._safe_error_message(response)
        reset_at = self.rate_limit.reset_at if self.rate_limit else None
        if response.status_code == 401:
            raise GitHubAuthenticationError(
                message,
                status_code=response.status_code,
                endpoint=endpoint,
                reset_at=reset_at,
            )
        if response.status_code == 404:
            raise GitHubNotFoundError(
                message,
                status_code=response.status_code,
                endpoint=endpoint,
                reset_at=reset_at,
            )
        if response.status_code == 403 and self._is_rate_limited(response, message):
            raise GitHubRateLimitError(
                message,
                status_code=response.status_code,
                endpoint=endpoint,
                reset_at=reset_at,
            )
        raise GitHubAPIError(
            message,
            status_code=response.status_code,
            endpoint=endpoint,
            reset_at=reset_at,
        )

    @staticmethod
    def _decode_json(response: httpx.Response, *, endpoint: str) -> JsonValue:
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise GitHubResponseError(
                "GitHub response body was not valid JSON.",
                status_code=response.status_code,
                endpoint=endpoint,
            ) from exc

    @staticmethod
    def _safe_error_message(response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return f"GitHub API request failed with status {response.status_code}."
        if isinstance(data, dict) and isinstance(data.get("message"), str):
            return data["message"]
        return f"GitHub API request failed with status {response.status_code}."

    @staticmethod
    def _is_rate_limited(response: httpx.Response, message: str) -> bool:
        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining == "0":
            return True
        return "rate limit" in message.lower()

    @staticmethod
    def _rate_limit_from_headers(headers: httpx.Headers) -> GitHubRateLimit | None:
        values = {
            "limit": _optional_int(headers.get("X-RateLimit-Limit")),
            "remaining": _optional_int(headers.get("X-RateLimit-Remaining")),
            "used": _optional_int(headers.get("X-RateLimit-Used")),
            "reset_at": _reset_at(headers.get("X-RateLimit-Reset")),
            "resource": headers.get("X-RateLimit-Resource"),
        }
        if all(value is None for value in values.values()):
            return None
        return GitHubRateLimit(**values)


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _reset_at(value: str | None) -> datetime | None:
    timestamp = _optional_int(value)
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=UTC)
