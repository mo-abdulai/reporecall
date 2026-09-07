from collections.abc import Mapping

from pydantic import ValidationError

from reporecall.github import GitHubClient, GitHubRepository, GitHubResponseError
from reporecall.ingestion.github_issue_loader import InvalidIssueNumberError
from reporecall.ingestion.github_pull_request_loader import (
    InvalidPullRequestNumberError,
)
from reporecall.models import GitHubIssueComment, GitHubUser


class GitHubCommentLoader:
    """Load issue-style GitHub comments for issues and pull request conversations."""

    def __init__(self, client: GitHubClient, repository: GitHubRepository) -> None:
        self.client = client
        self.repository = repository

    def get_issue_comments(self, issue_number: int) -> list[GitHubIssueComment]:
        """Return normal conversation comments attached to an issue."""

        if issue_number <= 0:
            raise InvalidIssueNumberError("Issue number must be a positive integer.")

        endpoint = f"{self._issues_path()}/{issue_number}/comments"
        return [
            self._parse_issue_comment(item, parent_number=issue_number, endpoint=endpoint)
            for item in self.client.paginate(endpoint)
        ]

    def get_pull_request_comments(self, pull_request_number: int) -> list[GitHubIssueComment]:
        """Return normal conversation comments attached to a pull request."""

        if pull_request_number <= 0:
            raise InvalidPullRequestNumberError("Pull request number must be a positive integer.")

        endpoint = f"{self._issues_path()}/{pull_request_number}/comments"
        return [
            self._parse_issue_comment(item, parent_number=pull_request_number, endpoint=endpoint)
            for item in self.client.paginate(endpoint)
        ]

    def _issues_path(self) -> str:
        return f"/repos/{self.repository.owner}/{self.repository.name}/issues"

    def _parse_issue_comment(
        self,
        data: Mapping[str, object],
        *,
        parent_number: int,
        endpoint: str,
    ) -> GitHubIssueComment:
        try:
            comment_data = {
                "repository": self.repository,
                "id": data["id"],
                "issue_number": parent_number,
                "author": self._parse_user(data.get("user"), endpoint=endpoint),
                "body": data.get("body"),
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
                "html_url": data["html_url"],
                "author_association": data.get("author_association"),
            }
            return GitHubIssueComment.model_validate(comment_data)
        except KeyError as exc:
            raise GitHubResponseError(
                f"GitHub issue comment response was missing required field: {exc.args[0]}.",
                endpoint=endpoint,
            ) from exc
        except ValidationError as exc:
            raise GitHubResponseError(
                "GitHub issue comment response could not be parsed into a RepoRecall model.",
                endpoint=endpoint,
            ) from exc

    def _parse_user(self, value: object, *, endpoint: str) -> GitHubUser | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise GitHubResponseError(
                "GitHub issue comment author was not a JSON object.",
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
                f"GitHub issue comment author was missing required field: {exc.args[0]}.",
                endpoint=endpoint,
            ) from exc
        except ValidationError as exc:
            raise GitHubResponseError(
                "GitHub issue comment author could not be parsed.",
                endpoint=endpoint,
            ) from exc
