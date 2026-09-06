from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class FileChangeType(str, Enum):
    """Raw Git file-level change categories supported by RepoRecall."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


class ChangedFile(BaseModel):
    """A file changed by a Git commit."""

    path: str
    change_type: FileChangeType
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    patch: str | None
    old_path: str | None = None


class GitCommit(BaseModel):
    """Normalized raw Git commit data."""

    sha: str
    message: str
    author_name: str
    author_email: str | None
    authored_at: datetime
    committed_at: datetime
    parent_shas: list[str]
    changed_files: list[ChangedFile]
