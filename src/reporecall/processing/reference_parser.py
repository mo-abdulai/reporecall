import re
from enum import Enum

from pydantic import BaseModel

from reporecall.github import GitHubRepository


class IssueReferenceKind(str, Enum):
    """Supported deterministic issue-reference meanings."""

    CLOSES = "closes"
    REFERENCES = "references"


class ParsedIssueReference(BaseModel):
    """A syntactic issue reference extracted from human-authored text."""

    repository: GitHubRepository
    issue_number: int
    kind: IssueReferenceKind
    evidence: str


class ReferenceParser:
    """Parse conservative GitHub issue references from text."""

    _REFERENCE_PATTERN = r"(?:(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+))?#(?P<number>[1-9]\d*)"
    _CLOSING_PATTERN = re.compile(
        rf"\b(?P<keyword>close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved)\s+"
        rf"(?P<reference>{_REFERENCE_PATTERN})",
        re.IGNORECASE,
    )
    _PLAIN_PATTERN = re.compile(
        rf"\b(?P<keyword>see|related to|follow-up to)\s+(?P<reference>{_REFERENCE_PATTERN})",
        re.IGNORECASE,
    )

    def parse_issue_references(
        self,
        text: str | None,
        *,
        default_repository: GitHubRepository,
    ) -> list[ParsedIssueReference]:
        """Return supported issue references in deterministic text order."""

        if not text:
            return []

        matches: list[tuple[int, ParsedIssueReference]] = []
        for match in self._CLOSING_PATTERN.finditer(text):
            matches.append(
                (
                    match.start(),
                    self._parsed_reference(
                        match,
                        default_repository=default_repository,
                        kind=IssueReferenceKind.CLOSES,
                    ),
                )
            )
        for match in self._PLAIN_PATTERN.finditer(text):
            matches.append(
                (
                    match.start(),
                    self._parsed_reference(
                        match,
                        default_repository=default_repository,
                        kind=IssueReferenceKind.REFERENCES,
                    ),
                )
            )

        return [parsed for _, parsed in sorted(matches, key=lambda item: item[0])]

    def _parsed_reference(
        self,
        match: re.Match[str],
        *,
        default_repository: GitHubRepository,
        kind: IssueReferenceKind,
    ) -> ParsedIssueReference:
        repository = default_repository
        owner = match.group("owner")
        repo = match.group("repo")
        if owner is not None and repo is not None:
            repository = GitHubRepository(owner=owner, name=repo)

        return ParsedIssueReference(
            repository=repository,
            issue_number=int(match.group("number")),
            kind=kind,
            evidence=match.group(0),
        )
