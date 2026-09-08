from collections.abc import Mapping

from pydantic import ValidationError

from reporecall.github import GitHubClient, GitHubRepository, GitHubResponseError
from reporecall.models import (
    GitHubBranchReference,
    GitHubCommitReference,
    GitHubIssueLabel,
    GitHubMilestone,
    GitHubPullRequest,
    GitHubPullRequestFile,
    GitHubPullRequestFileStatus,
    GitHubUser,
    PullRequestState,
)


class InvalidPullRequestLimitError(ValueError):
    """Raised when a pull request listing limit is not positive."""


class InvalidPullRequestNumberError(ValueError):
    """Raised when a pull request number is not positive."""


class GitHubPullRequestLoader:
    """Load GitHub pull request metadata, changed files, and commit references."""

    def __init__(self, client: GitHubClient, repository: GitHubRepository) -> None:
        self.client = client
        self.repository = repository

    def get_pull_requests(
        self,
        *,
        state: PullRequestState | None = None,
        limit: int | None = None,
    ) -> list[GitHubPullRequest]:
        """Return repository pull requests without fetching files or commits."""

        if limit is not None and limit <= 0:
            raise InvalidPullRequestLimitError("Pull request limit must be a positive integer.")

        params = {"state": state.value if state is not None else "all"}
        endpoint = self._pulls_path()
        raw_items = self.client.paginate(endpoint, params=params)

        pull_requests = [
            self._parse_pull_request(item, endpoint=endpoint)
            for item in raw_items[:limit]
        ]
        return pull_requests

    def get_pull_request(self, number: int) -> GitHubPullRequest:
        """Return a single pull request by number."""

        self._validate_number(number)
        endpoint = f"{self._pulls_path()}/{number}"
        data = self.client.get(endpoint)
        if not isinstance(data, dict):
            raise GitHubResponseError(
                "GitHub pull request response was not a JSON object.",
                endpoint=endpoint,
            )
        return self._parse_pull_request(data, endpoint=endpoint)

    def get_pull_request_files(self, number: int) -> list[GitHubPullRequestFile]:
        """Return files changed by a pull request."""

        self._validate_number(number)
        endpoint = f"{self._pulls_path()}/{number}/files"
        return [
            self._parse_pull_request_file(item, endpoint=endpoint)
            for item in self.client.paginate(endpoint)
        ]

    def get_pull_request_commits(self, number: int) -> list[GitHubCommitReference]:
        """Return lightweight commit references associated with a pull request."""

        self._validate_number(number)
        endpoint = f"{self._pulls_path()}/{number}/commits"
        return [
            self._parse_commit_reference(item, endpoint=endpoint)
            for item in self.client.paginate(endpoint)
        ]

    def _pulls_path(self) -> str:
        return f"/repos/{self.repository.owner}/{self.repository.name}/pulls"

    def _validate_number(self, number: int) -> None:
        if number <= 0:
            raise InvalidPullRequestNumberError("Pull request number must be a positive integer.")

    def _parse_pull_request(
        self,
        data: Mapping[str, object],
        *,
        endpoint: str,
    ) -> GitHubPullRequest:
        try:
            pull_request_data = {
                "repository": self.repository,
                "number": data["number"],
                "title": data["title"],
                "body": data.get("body"),
                "state": data["state"],
                "author": self._parse_user(data.get("user"), endpoint=endpoint),
                "labels": self._parse_labels(data.get("labels", []), endpoint=endpoint),
                "milestone": self._parse_milestone(
                    data.get("milestone"),
                    endpoint=endpoint,
                ),
                "draft": data.get("draft", False),
                "locked": data["locked"],
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
                "closed_at": data.get("closed_at"),
                "merged_at": data.get("merged_at"),
                "html_url": data["html_url"],
                "head": self._parse_branch_reference(data["head"], endpoint=endpoint),
                "base": self._parse_branch_reference(data["base"], endpoint=endpoint),
                "merge_commit_sha": data.get("merge_commit_sha"),
                "commits_count": data.get("commits"),
                "changed_files_count": data.get("changed_files"),
                "additions": data.get("additions"),
                "deletions": data.get("deletions"),
                "comments_count": data.get("comments"),
                "review_comments_count": data.get("review_comments"),
                "maintainer_can_modify": data.get("maintainer_can_modify"),
            }
            return GitHubPullRequest.model_validate(pull_request_data)
        except KeyError as exc:
            raise GitHubResponseError(
                f"GitHub pull request response was missing required field: {exc.args[0]}.",
                endpoint=endpoint,
            ) from exc
        except ValidationError as exc:
            raise GitHubResponseError(
                "GitHub pull request response could not be parsed into a RepoRecall model.",
                endpoint=endpoint,
            ) from exc

    def _parse_branch_reference(
        self,
        value: object,
        *,
        endpoint: str,
    ) -> GitHubBranchReference:
        if not isinstance(value, Mapping):
            raise GitHubResponseError(
                "GitHub pull request branch reference was not a JSON object.",
                endpoint=endpoint,
            )
        try:
            branch_data = {
                "repository": self._parse_repository(value.get("repo"), endpoint=endpoint),
                "ref": value["ref"],
                "sha": value["sha"],
                "label": value.get("label"),
            }
            return GitHubBranchReference.model_validate(branch_data)
        except KeyError as exc:
            raise GitHubResponseError(
                f"GitHub pull request branch reference was missing required field: {exc.args[0]}.",
                endpoint=endpoint,
            ) from exc
        except ValidationError as exc:
            raise GitHubResponseError(
                "GitHub pull request branch reference could not be parsed.",
                endpoint=endpoint,
            ) from exc

    def _parse_pull_request_file(
        self,
        data: Mapping[str, object],
        *,
        endpoint: str,
    ) -> GitHubPullRequestFile:
        try:
            file_data = {
                "filename": data["filename"],
                "status": self._normalize_file_status(data["status"]),
                "additions": data["additions"],
                "deletions": data["deletions"],
                "changes": data["changes"],
                "patch": data.get("patch"),
                "previous_filename": data.get("previous_filename"),
                "raw_url": data.get("raw_url"),
                "blob_url": data.get("blob_url"),
            }
            return GitHubPullRequestFile.model_validate(file_data)
        except KeyError as exc:
            raise GitHubResponseError(
                f"GitHub pull request file response was missing required field: {exc.args[0]}.",
                endpoint=endpoint,
            ) from exc
        except (ValueError, ValidationError) as exc:
            raise GitHubResponseError(
                "GitHub pull request file response could not be parsed.",
                endpoint=endpoint,
            ) from exc

    def _parse_commit_reference(
        self,
        data: Mapping[str, object],
        *,
        endpoint: str,
    ) -> GitHubCommitReference:
        try:
            commit = data["commit"]
            if not isinstance(commit, Mapping):
                raise GitHubResponseError(
                    "GitHub pull request commit object was not a JSON object.",
                    endpoint=endpoint,
                )
            author = commit.get("author")
            if author is not None and not isinstance(author, Mapping):
                raise GitHubResponseError(
                    "GitHub pull request commit author was not a JSON object.",
                    endpoint=endpoint,
                )
            commit_data = {
                "sha": data["sha"],
                "html_url": data.get("html_url"),
                "message": commit["message"],
                "author_name": author.get("name") if author is not None else None,
                "author_email": author.get("email") if author is not None else None,
                "authored_at": author.get("date") if author is not None else None,
            }
            return GitHubCommitReference.model_validate(commit_data)
        except KeyError as exc:
            raise GitHubResponseError(
                f"GitHub pull request commit response was missing required field: {exc.args[0]}.",
                endpoint=endpoint,
            ) from exc
        except ValidationError as exc:
            raise GitHubResponseError(
                "GitHub pull request commit response could not be parsed.",
                endpoint=endpoint,
            ) from exc

    def _parse_repository(self, value: object, *, endpoint: str) -> GitHubRepository | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise GitHubResponseError(
                "GitHub branch repository was not a JSON object.",
                endpoint=endpoint,
            )
        owner = value.get("owner")
        if owner is not None and not isinstance(owner, Mapping):
            raise GitHubResponseError(
                "GitHub branch repository owner was not a JSON object.",
                endpoint=endpoint,
            )
        try:
            owner_login = owner["login"] if owner is not None else value["owner"]
            repository_data = {"owner": owner_login, "name": value["name"]}
            return GitHubRepository.model_validate(repository_data)
        except KeyError as exc:
            raise GitHubResponseError(
                f"GitHub branch repository was missing required field: {exc.args[0]}.",
                endpoint=endpoint,
            ) from exc
        except ValidationError as exc:
            raise GitHubResponseError(
                "GitHub branch repository could not be parsed.",
                endpoint=endpoint,
            ) from exc

    def _parse_user(self, value: object, *, endpoint: str) -> GitHubUser | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise GitHubResponseError(
                "GitHub pull request author was not a JSON object.",
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
                f"GitHub pull request author was missing required field: {exc.args[0]}.",
                endpoint=endpoint,
            ) from exc
        except ValidationError as exc:
            raise GitHubResponseError(
                "GitHub pull request author could not be parsed.",
                endpoint=endpoint,
            ) from exc

    def _parse_labels(self, value: object, *, endpoint: str) -> list[GitHubIssueLabel]:
        if not isinstance(value, list):
            raise GitHubResponseError(
                "GitHub pull request labels were not a JSON array.",
                endpoint=endpoint,
            )

        labels: list[GitHubIssueLabel] = []
        for item in value:
            if not isinstance(item, Mapping):
                raise GitHubResponseError(
                    "GitHub pull request label was not a JSON object.",
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
                    f"GitHub pull request label was missing required field: {exc.args[0]}.",
                    endpoint=endpoint,
                ) from exc
            except ValidationError as exc:
                raise GitHubResponseError(
                    "GitHub pull request label could not be parsed.",
                    endpoint=endpoint,
                ) from exc

        return labels

    @staticmethod
    def _parse_milestone(value: object, *, endpoint: str) -> GitHubMilestone | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise GitHubResponseError(
                "GitHub pull request milestone was not a JSON object.",
                endpoint=endpoint,
            )
        try:
            return GitHubMilestone(
                number=value["number"],
                title=value["title"],
                html_url=value.get("html_url"),
            )
        except KeyError as exc:
            raise GitHubResponseError(
                f"GitHub pull request milestone was missing required field: {exc.args[0]}.",
                endpoint=endpoint,
            ) from exc
        except ValidationError as exc:
            raise GitHubResponseError(
                "GitHub pull request milestone could not be parsed.",
                endpoint=endpoint,
            ) from exc

    @staticmethod
    def _normalize_file_status(value: object) -> GitHubPullRequestFileStatus:
        if value == "removed":
            return GitHubPullRequestFileStatus.DELETED
        return GitHubPullRequestFileStatus(value)
