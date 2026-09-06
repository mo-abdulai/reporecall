from pathlib import Path

from git import NULL_TREE, BadName, GitCommandError, NoSuchPathError, Repo
from git import InvalidGitRepositoryError as GitPythonInvalidRepositoryError
from git.diff import Diff
from git.objects.commit import Commit

from reporecall.models import ChangedFile, FileChangeType, GitCommit


class InvalidRepositoryError(ValueError):
    """Raised when a path is not a usable local Git repository."""


class CommitNotFoundError(ValueError):
    """Raised when a commit revision cannot be resolved."""


class InvalidCommitLimitError(ValueError):
    """Raised when a commit limit is not positive."""


class GitLoader:
    """Load commit history from an existing local Git repository."""

    def __init__(self, repo_path: str | Path) -> None:
        self.repo_path = Path(repo_path)
        self.repo = self._open_repository(self.repo_path)

    def get_commits(self, limit: int | None = None) -> list[GitCommit]:
        """Return reachable commits newest-first."""

        if limit is not None and limit <= 0:
            raise InvalidCommitLimitError("Commit limit must be a positive integer.")

        commits = self.repo.iter_commits(max_count=limit)
        return [self._to_git_commit(commit) for commit in commits]

    def get_commit(self, sha: str) -> GitCommit:
        """Return a single commit resolved from a full or abbreviated revision."""

        try:
            commit = self.repo.commit(sha)
        except (BadName, ValueError, GitCommandError) as exc:
            raise CommitNotFoundError(f"Commit revision could not be resolved: {sha}") from exc

        return self._to_git_commit(commit)

    @staticmethod
    def _open_repository(repo_path: Path) -> Repo:
        if not repo_path.exists():
            raise InvalidRepositoryError(f"Repository path does not exist: {repo_path}")
        if not repo_path.is_dir():
            raise InvalidRepositoryError(f"Repository path is not a directory: {repo_path}")

        try:
            return Repo(repo_path)
        except (GitPythonInvalidRepositoryError, NoSuchPathError) as exc:
            raise InvalidRepositoryError(f"Path is not a valid Git repository: {repo_path}") from exc

    def _to_git_commit(self, commit: Commit) -> GitCommit:
        return GitCommit(
            sha=commit.hexsha,
            message=self._text_value(commit.message),
            author_name=commit.author.name or "",
            author_email=commit.author.email,
            authored_at=commit.authored_datetime,
            committed_at=commit.committed_datetime,
            parent_shas=[parent.hexsha for parent in commit.parents],
            changed_files=self._changed_files_for_commit(commit),
        )

    def _changed_files_for_commit(self, commit: Commit) -> list[ChangedFile]:
        if commit.parents:
            diff_index = commit.diff(commit.parents[0], create_patch=True, R=True)
        else:
            diff_index = commit.diff(NULL_TREE, create_patch=True)
        stats = self._file_stats(commit)

        return [self._to_changed_file(diff, stats) for diff in diff_index]

    @staticmethod
    def _file_stats(commit: Commit) -> dict[str, dict[str, int]]:
        return {
            str(path): {
                "additions": int(values.get("insertions", 0)),
                "deletions": int(values.get("deletions", 0)),
            }
            for path, values in commit.stats.files.items()
        }

    def _to_changed_file(
        self,
        diff: Diff,
        stats: dict[str, dict[str, int]],
    ) -> ChangedFile:
        path = self._current_path(diff)
        stat = stats.get(path, {"additions": 0, "deletions": 0})

        return ChangedFile(
            path=path,
            change_type=self._change_type(diff),
            additions=stat["additions"],
            deletions=stat["deletions"],
            patch=self._patch_text(diff),
            old_path=diff.rename_from if diff.renamed_file else None,
        )

    @staticmethod
    def _current_path(diff: Diff) -> str:
        if diff.renamed_file and diff.rename_to:
            return diff.rename_to
        if diff.b_path:
            return diff.b_path
        if diff.a_path:
            return diff.a_path
        raise InvalidRepositoryError("Git diff did not contain a file path.")

    @staticmethod
    def _change_type(diff: Diff) -> FileChangeType:
        if diff.renamed_file:
            return FileChangeType.RENAMED
        if diff.new_file:
            return FileChangeType.ADDED
        if diff.deleted_file:
            return FileChangeType.DELETED
        return FileChangeType.MODIFIED

    @staticmethod
    def _patch_text(diff: Diff) -> str | None:
        if not diff.diff:
            return None
        if isinstance(diff.diff, bytes):
            try:
                text = diff.diff.decode("utf-8")
            except UnicodeDecodeError:
                return None
        else:
            text = diff.diff
        if not text:
            return None
        return text

    @staticmethod
    def _text_value(value: str | bytes) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value
