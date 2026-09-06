import re
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GitHubRepository(BaseModel):
    """A GitHub repository owner/name identifier."""

    owner: str = Field(min_length=1)
    name: str = Field(min_length=1)

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_identifier_parts(self) -> "GitHubRepository":
        for value in (self.owner, self.name):
            if "/" in value or value in {".", ".."}:
                raise ValueError("Repository owner and name must be path segments.")
        return self

    @classmethod
    def parse(cls, value: str) -> "GitHubRepository":
        """Parse common GitHub repository identifier formats."""

        candidate = value.strip()
        if not candidate:
            raise ValueError("Repository identifier must not be empty.")

        ssh_match = re.fullmatch(r"git@github\.com:([^/]+)/(.+?)(?:\.git)?", candidate)
        if ssh_match:
            return cls(owner=ssh_match.group(1), name=ssh_match.group(2))

        parsed = urlparse(candidate)
        if parsed.scheme:
            if parsed.netloc.lower() != "github.com":
                raise ValueError("Only github.com repository URLs are supported.")
            parts = [part for part in parsed.path.strip("/").split("/") if part]
        else:
            parts = [part for part in candidate.split("/") if part]

        if len(parts) != 2:
            raise ValueError("Repository identifier must include owner and repository name.")

        owner, name = parts
        return cls(owner=owner, name=name.removesuffix(".git"))
