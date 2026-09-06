from collections.abc import Mapping

from pydantic import ValidationError

from reporecall.github import GitHubClient, GitHubRepository, GitHubResponseError
from reporecall.models import GitHubIssue, GitHubIssueLabel, GitHubUser, IssueState


class InvalidIssueLimitError(ValueError):
    """Raised when an issue listing limit is not positive."""


class InvalidIssueNumberError(ValueError):
    """Raised when an issue number is not positive."""


class NotAnIssueError(ValueError):
    """Raised when GitHub returns a pull request from an issue endpoint."""


class GitHubIssueLoader:
    """Load GitHub issues and convert GitHub JSON into RepoRecall models."""

    def __init__(self, client: GitHubClient, repository: GitHubRepository) -> None:
        self.client = client
        self.repository = repository

    def get_issues(
        self,
        *,
        state: IssueState | None = None,
        limit: int | None = None,
    ) -> list[GitHubIssue]:
        """Return repository issues, excluding pull requests from GitHub's issues API."""

        if limit is not None and limit <= 0:
            raise InvalidIssueLimitError("Issue limit must be a positive integer.")

        params = {"state": state.value if state is not None else "all"}
        raw_items = self.client.paginate(self._issues_path(), params=params)

        issues: list[GitHubIssue] = []
        for item in raw_items:
            if self._is_pull_request(item):
                continue
            issues.append(self._parse_issue(item, endpoint=self._issues_path()))
            if limit is not None and len(issues) >= limit:
                break

        return issues

    def get_issue(self, number: int) -> GitHubIssue:
        """Return a single issue by number."""

        if number <= 0:
            raise InvalidIssueNumberError("Issue number must be a positive integer.")

        endpoint = f"{self._issues_path()}/{number}"
        data = self.client.get(endpoint)
        if not isinstance(data, dict):
            raise GitHubResponseError(
                "GitHub issue response was not a JSON object.",
                endpoint=endpoint,
            )
        if self._is_pull_request(data):
            raise NotAnIssueError(f"GitHub issue #{number} is a pull request, not an issue.")
        return self._parse_issue(data, endpoint=endpoint)

    def _issues_path(self) -> str:
        return f"/repos/{self.repository.owner}/{self.repository.name}/issues"

    def _parse_issue(self, data: Mapping[str, object], *, endpoint: str) -> GitHubIssue:
        try:
            issue_data = {
                "repository": self.repository,
                "number": data["number"],
                "title": data["title"],
                "body": data.get("body"),
                "state": data["state"],
                "author": self._parse_user(data.get("user"), endpoint=endpoint),
                "labels": self._parse_labels(data["labels"], endpoint=endpoint),
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
                "closed_at": data.get("closed_at"),
                "html_url": data["html_url"],
                "comments_count": data["comments"],
                "locked": data["locked"],
            }
            return GitHubIssue.model_validate(issue_data)
        except KeyError as exc:
            raise GitHubResponseError(
                f"GitHub issue response was missing required field: {exc.args[0]}.",
                endpoint=endpoint,
            ) from exc
        except ValidationError as exc:
            raise GitHubResponseError(
                "GitHub issue response could not be parsed into a RepoRecall issue model.",
                endpoint=endpoint,
            ) from exc

    def _parse_user(self, value: object, *, endpoint: str) -> GitHubUser | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise GitHubResponseError(
                "GitHub issue author was not a JSON object.",
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
                f"GitHub issue author was missing required field: {exc.args[0]}.",
                endpoint=endpoint,
            ) from exc
        except ValidationError as exc:
            raise GitHubResponseError(
                "GitHub issue author could not be parsed.",
                endpoint=endpoint,
            ) from exc

    def _parse_labels(self, value: object, *, endpoint: str) -> list[GitHubIssueLabel]:
        if not isinstance(value, list):
            raise GitHubResponseError(
                "GitHub issue labels were not a JSON array.",
                endpoint=endpoint,
            )

        labels: list[GitHubIssueLabel] = []
        for item in value:
            if not isinstance(item, Mapping):
                raise GitHubResponseError(
                    "GitHub issue label was not a JSON object.",
                    endpoint=endpoint,
                )
            try:
                labels.append(
                    GitHubIssueLabel(
                        name=item["name"],
                        color=item.get("color"),
                        description=item.get("description"),
                    )
                )
            except KeyError as exc:
                raise GitHubResponseError(
                    f"GitHub issue label was missing required field: {exc.args[0]}.",
                    endpoint=endpoint,
                ) from exc
            except ValidationError as exc:
                raise GitHubResponseError(
                    "GitHub issue label could not be parsed.",
                    endpoint=endpoint,
                ) from exc

        return labels

    @staticmethod
    def _is_pull_request(data: Mapping[str, object]) -> bool:
        return "pull_request" in data
