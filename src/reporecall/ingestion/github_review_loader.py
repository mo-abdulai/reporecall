from collections.abc import Mapping

from pydantic import ValidationError

from reporecall.github import GitHubClient, GitHubRepository, GitHubResponseError
from reporecall.ingestion.github_pull_request_loader import (
    InvalidPullRequestNumberError,
)
from reporecall.models import (
    GitHubPullRequestReview,
    GitHubPullRequestReviewComment,
    GitHubUser,
)


class GitHubReviewLoader:
    """Load GitHub pull request reviews and line-level review comments."""

    def __init__(self, client: GitHubClient, repository: GitHubRepository) -> None:
        self.client = client
        self.repository = repository

    def get_pull_request_reviews(self, pull_request_number: int) -> list[GitHubPullRequestReview]:
        """Return reviews submitted on a pull request."""

        self._validate_pull_request_number(pull_request_number)
        endpoint = f"{self._pulls_path()}/{pull_request_number}/reviews"
        return [
            self._parse_review(item, pull_request_number=pull_request_number, endpoint=endpoint)
            for item in self.client.paginate(endpoint)
        ]

    def get_pull_request_review_comments(
        self,
        pull_request_number: int,
    ) -> list[GitHubPullRequestReviewComment]:
        """Return line-level review comments submitted on a pull request."""

        self._validate_pull_request_number(pull_request_number)
        endpoint = f"{self._pulls_path()}/{pull_request_number}/comments"
        return [
            self._parse_review_comment(item, pull_request_number=pull_request_number, endpoint=endpoint)
            for item in self.client.paginate(endpoint)
        ]

    def _pulls_path(self) -> str:
        return f"/repos/{self.repository.owner}/{self.repository.name}/pulls"

    def _validate_pull_request_number(self, pull_request_number: int) -> None:
        if pull_request_number <= 0:
            raise InvalidPullRequestNumberError("Pull request number must be a positive integer.")

    def _parse_review(
        self,
        data: Mapping[str, object],
        *,
        pull_request_number: int,
        endpoint: str,
    ) -> GitHubPullRequestReview:
        try:
            review_data = {
                "repository": self.repository,
                "id": data["id"],
                "pull_request_number": pull_request_number,
                "author": self._parse_user(data.get("user"), context="review", endpoint=endpoint),
                "body": data.get("body"),
                "state": data["state"],
                "submitted_at": data.get("submitted_at"),
                "commit_sha": data.get("commit_id"),
                "html_url": data.get("html_url"),
            }
            return GitHubPullRequestReview.model_validate(review_data)
        except KeyError as exc:
            raise GitHubResponseError(
                f"GitHub pull request review response was missing required field: {exc.args[0]}.",
                endpoint=endpoint,
            ) from exc
        except ValidationError as exc:
            raise GitHubResponseError(
                "GitHub pull request review response could not be parsed into a RepoRecall model.",
                endpoint=endpoint,
            ) from exc

    def _parse_review_comment(
        self,
        data: Mapping[str, object],
        *,
        pull_request_number: int,
        endpoint: str,
    ) -> GitHubPullRequestReviewComment:
        try:
            comment_data = {
                "repository": self.repository,
                "id": data["id"],
                "pull_request_number": pull_request_number,
                "review_id": data["pull_request_review_id"],
                "author": self._parse_user(data.get("user"), context="review comment", endpoint=endpoint),
                "body": data.get("body"),
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
                "html_url": data["html_url"],
                "commit_sha": data.get("commit_id"),
                "original_commit_sha": data.get("original_commit_id"),
                "path": data["path"],
                "line": data.get("line"),
                "original_line": data.get("original_line"),
                "side": data.get("side"),
                "start_line": data.get("start_line"),
                "start_side": data.get("start_side"),
                "author_association": data.get("author_association"),
            }
            return GitHubPullRequestReviewComment.model_validate(comment_data)
        except KeyError as exc:
            raise GitHubResponseError(
                f"GitHub pull request review comment response was missing required field: {exc.args[0]}.",
                endpoint=endpoint,
            ) from exc
        except ValidationError as exc:
            raise GitHubResponseError(
                "GitHub pull request review comment response could not be parsed into a RepoRecall model.",
                endpoint=endpoint,
            ) from exc

    def _parse_user(self, value: object, *, context: str, endpoint: str) -> GitHubUser | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise GitHubResponseError(
                f"GitHub pull request {context} author was not a JSON object.",
                endpoint=endpoint,
            )
        try:
            return GitHubUser(
                login=value["login"],
                id=value.get("id"),
                html_url=value.get("html_url"),
            )
        except KeyError as exc:
            raise GitHubResponseError(
                f"GitHub pull request {context} author was missing required field: {exc.args[0]}.",
                endpoint=endpoint,
            ) from exc
        except ValidationError as exc:
            raise GitHubResponseError(
                f"GitHub pull request {context} author could not be parsed.",
                endpoint=endpoint,
            ) from exc
