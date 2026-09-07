from collections.abc import Mapping
from urllib.parse import urlparse

from pydantic import ValidationError

from reporecall.github import GitHubClient, GitHubRepository, GitHubResponseError
from reporecall.ingestion.github_issue_loader import InvalidIssueNumberError
from reporecall.models import (
    GitHubCommitPullRequestAssociation,
    GitHubTimelineEvidenceType,
    GitHubTimelineRelationshipEvidence,
    GitHubUser,
)

_SUPPORTED_TIMELINE_EVENTS = frozenset(member.value for member in GitHubTimelineEvidenceType)


class GitHubRelationshipEvidenceLoader:
    """Explicitly load GitHub-provided evidence for direct relationships."""

    def __init__(self, client: GitHubClient, repository: GitHubRepository) -> None:
        self.client = client
        self.repository = repository

    def get_timeline_relationship_evidence(
        self,
        number: int,
    ) -> list[GitHubTimelineRelationshipEvidence]:
        """Return supported relationship evidence from an issue-shaped timeline."""

        if number <= 0:
            raise InvalidIssueNumberError("Issue or pull request number must be a positive integer.")

        endpoint = f"{self._issues_path()}/{number}/timeline"
        evidence: list[GitHubTimelineRelationshipEvidence] = []
        for item in self.client.paginate(endpoint):
            event = item.get("event")
            if not isinstance(event, str) or event not in _SUPPORTED_TIMELINE_EVENTS:
                continue
            if event == GitHubTimelineEvidenceType.CLOSED.value and item.get("commit_id") is None:
                continue
            evidence.append(self._parse_timeline_evidence(item, number=number, endpoint=endpoint))
        return evidence

    def get_pull_requests_for_commit(
        self,
        commit_sha: str,
    ) -> list[GitHubCommitPullRequestAssociation]:
        """Return pull requests GitHub directly associates with one commit."""

        normalized_sha = commit_sha.strip()
        if not normalized_sha:
            raise ValueError("Commit SHA must not be empty.")

        endpoint = (
            f"/repos/{self.repository.owner}/{self.repository.name}"
            f"/commits/{normalized_sha}/pulls"
        )
        return [
            self._parse_commit_pull_request_association(
                item,
                commit_sha=normalized_sha,
                endpoint=endpoint,
            )
            for item in self.client.paginate(endpoint)
        ]

    def _parse_timeline_evidence(
        self,
        data: Mapping[str, object],
        *,
        number: int,
        endpoint: str,
    ) -> GitHubTimelineRelationshipEvidence:
        try:
            event_type = GitHubTimelineEvidenceType(data["event"])
            evidence_data: dict[str, object] = {
                "repository": self.repository,
                "target_number": number,
                "event_type": event_type,
                "created_at": data["created_at"],
                "actor": self._parse_user(data.get("actor"), endpoint=endpoint),
            }
            if event_type is GitHubTimelineEvidenceType.CROSS_REFERENCED:
                evidence_data.update(self._parse_cross_reference(data, endpoint=endpoint))
            else:
                evidence_data.update(self._parse_commit_event(data, endpoint=endpoint))
            return GitHubTimelineRelationshipEvidence.model_validate(evidence_data)
        except KeyError as exc:
            raise GitHubResponseError(
                f"GitHub timeline relationship event was missing required field: {exc.args[0]}.",
                endpoint=endpoint,
            ) from exc
        except (TypeError, ValueError, ValidationError) as exc:
            raise GitHubResponseError(
                "GitHub timeline relationship event could not be parsed.",
                endpoint=endpoint,
            ) from exc

    def _parse_cross_reference(
        self,
        data: Mapping[str, object],
        *,
        endpoint: str,
    ) -> dict[str, object]:
        source = data["source"]
        if not isinstance(source, Mapping) or source.get("type") != "issue":
            raise GitHubResponseError(
                "GitHub timeline cross-reference source was not an issue-shaped object.",
                endpoint=endpoint,
            )
        issue = source.get("issue")
        if not isinstance(issue, Mapping):
            raise GitHubResponseError(
                "GitHub timeline cross-reference source issue was not a JSON object.",
                endpoint=endpoint,
            )

        repository_url = issue.get("repository_url", issue.get("url"))
        return {
            "source_repository": self._parse_repository_url(
                repository_url,
                endpoint=endpoint,
                field_name="cross-reference source repository URL",
            ),
            "source_number": issue["number"],
            "source_is_pull_request": "pull_request" in issue,
        }

    def _parse_commit_event(
        self,
        data: Mapping[str, object],
        *,
        endpoint: str,
    ) -> dict[str, object]:
        commit_url = data["commit_url"]
        commit_repository = data.get("commit_repository")
        if commit_repository is not None:
            repository = self._parse_repository_object(commit_repository, endpoint=endpoint)
        else:
            repository = self._parse_repository_url(
                commit_url,
                endpoint=endpoint,
                field_name="commit URL",
            )
        return {
            "commit_repository": repository,
            "commit_sha": data["commit_id"],
            "commit_url": commit_url,
        }

    def _parse_commit_pull_request_association(
        self,
        data: Mapping[str, object],
        *,
        commit_sha: str,
        endpoint: str,
    ) -> GitHubCommitPullRequestAssociation:
        try:
            association_data = {
                "repository": self.repository,
                "commit_sha": commit_sha,
                "pull_request_number": data["number"],
                "pull_request_url": data["html_url"],
            }
            return GitHubCommitPullRequestAssociation.model_validate(association_data)
        except KeyError as exc:
            raise GitHubResponseError(
                f"GitHub commit pull request association was missing required field: {exc.args[0]}.",
                endpoint=endpoint,
            ) from exc
        except ValidationError as exc:
            raise GitHubResponseError(
                "GitHub commit pull request association could not be parsed.",
                endpoint=endpoint,
            ) from exc

    def _parse_repository_object(
        self,
        value: object,
        *,
        endpoint: str,
    ) -> GitHubRepository:
        if not isinstance(value, Mapping):
            raise GitHubResponseError(
                "GitHub commit repository was not a JSON object.",
                endpoint=endpoint,
            )

        try:
            full_name = value.get("full_name")
            if isinstance(full_name, str):
                return GitHubRepository.parse(full_name)

            owner = value["owner"]
            if not isinstance(owner, Mapping):
                raise GitHubResponseError(
                    "GitHub commit repository owner was not a JSON object.",
                    endpoint=endpoint,
                )
            return GitHubRepository.model_validate(
                {"owner": owner["login"], "name": value["name"]}
            )
        except KeyError as exc:
            raise GitHubResponseError(
                f"GitHub commit repository was missing required field: {exc.args[0]}.",
                endpoint=endpoint,
            ) from exc
        except (ValueError, ValidationError) as exc:
            raise GitHubResponseError(
                "GitHub commit repository could not be parsed.",
                endpoint=endpoint,
            ) from exc

    def _parse_repository_url(
        self,
        value: object,
        *,
        endpoint: str,
        field_name: str,
    ) -> GitHubRepository:
        if not isinstance(value, str):
            raise GitHubResponseError(
                f"GitHub timeline {field_name} was not a string.",
                endpoint=endpoint,
            )

        parts = [part for part in urlparse(value).path.split("/") if part]
        try:
            repos_index = parts.index("repos")
            return GitHubRepository(
                owner=parts[repos_index + 1],
                name=parts[repos_index + 2],
            )
        except (IndexError, ValueError, ValidationError) as exc:
            raise GitHubResponseError(
                f"GitHub timeline {field_name} could not be parsed.",
                endpoint=endpoint,
            ) from exc

    def _parse_user(self, value: object, *, endpoint: str) -> GitHubUser | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise GitHubResponseError(
                "GitHub timeline actor was not a JSON object.",
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
                f"GitHub timeline actor was missing required field: {exc.args[0]}.",
                endpoint=endpoint,
            ) from exc
        except ValidationError as exc:
            raise GitHubResponseError(
                "GitHub timeline actor could not be parsed.",
                endpoint=endpoint,
            ) from exc

    def _issues_path(self) -> str:
        return f"/repos/{self.repository.owner}/{self.repository.name}/issues"
