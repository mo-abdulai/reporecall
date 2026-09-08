import posixpath
from collections.abc import Iterable
from pathlib import PurePosixPath

from reporecall.models import (
    EngineeringEvent,
    EventActor,
    EventActorType,
    EventMetadata,
    GitCommit,
    GitHubUser,
)

EXTENSION_LANGUAGE_MAP = {
    ".c": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".cs": "C#",
    ".go": "Go",
    ".h": "C/C++",
    ".hpp": "C++",
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".php": "PHP",
    ".py": "Python",
    ".pyi": "Python",
    ".rb": "Ruby",
    ".rs": "Rust",
    ".swift": "Swift",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
}

_DOCUMENTATION_EXTENSIONS = frozenset({".adoc", ".md", ".rst"})
_DOCUMENTATION_NAMES = ("changelog", "contributing", "readme")
_CONFIGURATION_EXTENSIONS = frozenset(
    {".cfg", ".ini", ".tf", ".tfvars", ".toml", ".yaml", ".yml"}
)
_DEPENDENCY_FILENAMES = frozenset(
    {
        "build.gradle",
        "build.gradle.kts",
        "cargo.lock",
        "cargo.toml",
        "composer.json",
        "composer.lock",
        "gemfile",
        "gemfile.lock",
        "go.mod",
        "go.sum",
        "package-lock.json",
        "package.json",
        "pipfile",
        "pipfile.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "pom.xml",
        "pyproject.toml",
        "requirements.txt",
        "uv.lock",
        "yarn.lock",
    }
)


class EventMetadataExtractor:
    """Derive reproducible structural metadata from an engineering event."""

    def extract(self, event: EngineeringEvent) -> EventMetadata:
        """Return immutable metadata without mutating or performing I/O."""

        paths = _normalized_paths(event.changed_paths)
        extensions = _sorted_strings(
            extension
            for path in paths
            if (extension := PurePosixPath(path).suffix.lower())
        )
        test_paths = tuple(path for path in paths if _is_test_path(path))
        documentation_paths = tuple(path for path in paths if _is_documentation_path(path))
        configuration_paths = tuple(path for path in paths if _is_configuration_path(path))
        dependency_paths = tuple(path for path in paths if _is_dependency_path(path))
        authors, participants = _people(event)
        added_lines, deleted_lines = _diff_statistics(event)

        return EventMetadata(
            event_id=event.event_id,
            repository=event.repository,
            issue_numbers=tuple(event.issue_numbers),
            pull_request_numbers=tuple(event.pull_request_numbers),
            commit_shas=tuple(event.commit_shas),
            changed_paths=paths,
            authors=authors,
            participants=participants,
            labels=_labels(event),
            milestones=_milestones(event),
            languages=_sorted_strings(
                EXTENSION_LANGUAGE_MAP[extension]
                for extension in extensions
                if extension in EXTENSION_LANGUAGE_MAP
            ),
            file_extensions=extensions,
            directories=_directories(paths),
            added_lines=added_lines,
            deleted_lines=deleted_lines,
            changed_lines=added_lines + deleted_lines,
            has_tests=bool(test_paths),
            test_paths=test_paths,
            has_documentation_changes=bool(documentation_paths),
            documentation_paths=documentation_paths,
            has_configuration_changes=bool(configuration_paths),
            configuration_paths=configuration_paths,
            has_dependency_changes=bool(dependency_paths),
            dependency_paths=dependency_paths,
        )


def _labels(event: EngineeringEvent) -> tuple[str, ...]:
    values: list[str] = []
    for issue in event.issues:
        values.extend(label.name.strip() for label in issue.labels if label.name.strip())
    for pull_request in event.pull_requests:
        values.extend(
            label.name.strip() for label in pull_request.labels if label.name.strip()
        )
    return _sorted_strings(values)


def _milestones(event: EngineeringEvent) -> tuple[str, ...]:
    values: list[str] = []
    for issue in event.issues:
        if issue.milestone is not None and issue.milestone.title.strip():
            values.append(issue.milestone.title.strip())
    for pull_request in event.pull_requests:
        if pull_request.milestone is not None and pull_request.milestone.title.strip():
            values.append(pull_request.milestone.title.strip())
    return _sorted_strings(values)


def _people(event: EngineeringEvent) -> tuple[tuple[EventActor, ...], tuple[EventActor, ...]]:
    authors = [
        *(_github_actor(issue.author) for issue in event.issues),
        *(_github_actor(pull_request.author) for pull_request in event.pull_requests),
        *(
            _git_actor(commit.author_name, commit.author_email)
            for commit in event.github_commit_references
        ),
        *(
            _git_actor(commit.author_name, commit.author_email)
            for commit in event.local_commits
        ),
    ]
    contributors = [
        *authors,
        *(_github_actor(comment.author) for comment in event.issue_comments),
        *(_github_actor(comment.author) for comment in event.pull_request_comments),
        *(_github_actor(review.author) for review in event.pull_request_reviews),
        *(
            _github_actor(comment.author)
            for comment in event.pull_request_review_comments
        ),
    ]
    return _sorted_actors(authors), _sorted_actors(contributors)


def _github_actor(user: GitHubUser | None) -> EventActor | None:
    if user is None or not user.login.strip():
        return None
    login = user.login.strip()
    return EventActor(
        actor_type=EventActorType.GITHUB_USER,
        identifier=login,
        name=login,
    )


def _git_actor(name: str | None, email: str | None) -> EventActor | None:
    normalized_name = name.strip() if name is not None else ""
    normalized_email = email.strip() if email is not None else ""
    identifier = normalized_email or normalized_name
    if not identifier:
        return None
    return EventActor(
        actor_type=EventActorType.GIT_AUTHOR,
        identifier=identifier,
        name=normalized_name or None,
        email=normalized_email or None,
    )


def _sorted_actors(actors: Iterable[EventActor | None]) -> tuple[EventActor, ...]:
    actors_by_identity: dict[tuple[EventActorType, str], EventActor] = {}
    candidates = sorted(
        (actor for actor in actors if actor is not None),
        key=_actor_sort_key,
    )
    for actor in candidates:
        actors_by_identity.setdefault((actor.actor_type, actor.identifier), actor)
    return tuple(sorted(actors_by_identity.values(), key=_actor_sort_key))


def _actor_sort_key(actor: EventActor) -> tuple[str, str, str, str, str]:
    return (
        actor.actor_type.value,
        actor.identifier.casefold(),
        actor.identifier,
        actor.name or "",
        actor.email or "",
    )


def _normalized_paths(paths: Iterable[str]) -> tuple[str, ...]:
    normalized = {_normalize_path(path) for path in paths}
    return tuple(sorted((path for path in normalized if path), key=_string_sort_key))


def _normalize_path(path: str) -> str:
    normalized = posixpath.normpath(path.strip().replace("\\", "/"))
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")
    return "" if normalized == "." else normalized


def _directories(paths: Iterable[str]) -> tuple[str, ...]:
    directories: set[str] = set()
    for path in paths:
        parts = PurePosixPath(path).parent.parts
        for index in range(1, len(parts) + 1):
            directory = "/".join(parts[:index])
            if directory and directory != ".":
                directories.add(directory)
    return tuple(sorted(directories, key=_string_sort_key))


def _diff_statistics(event: EngineeringEvent) -> tuple[int, int]:
    pull_request_paths = {
        _normalize_path(file.filename) for file in event.pull_request_files
    }
    added_lines = sum(file.additions for file in event.pull_request_files)
    deleted_lines = sum(file.deletions for file in event.pull_request_files)

    local_commits = _unique_local_commits(event.local_commits)
    local_files = [
        file
        for commit in local_commits
        for file in commit.changed_files
        if _normalize_path(file.path) not in pull_request_paths
    ]
    if event.pull_request_files or local_files:
        return (
            added_lines + sum(file.additions for file in local_files),
            deleted_lines + sum(file.deletions for file in local_files),
        )

    pull_requests_with_statistics = [
        pull_request
        for pull_request in event.pull_requests
        if pull_request.additions is not None and pull_request.deletions is not None
    ]
    return (
        sum(pull_request.additions or 0 for pull_request in pull_requests_with_statistics),
        sum(pull_request.deletions or 0 for pull_request in pull_requests_with_statistics),
    )


def _unique_local_commits(commits: Iterable[GitCommit]) -> tuple[GitCommit, ...]:
    commits_by_sha: dict[str, GitCommit] = {}
    for commit in sorted(commits, key=lambda item: (item.sha, item.model_dump_json())):
        commits_by_sha.setdefault(commit.sha, commit)
    return tuple(commits_by_sha.values())


def _is_test_path(path: str) -> bool:
    pure_path = PurePosixPath(path.lower())
    filename = pure_path.name
    return (
        any(part in {"__tests__", "tests"} for part in pure_path.parts[:-1])
        or (filename.startswith("test_") and filename.endswith(".py"))
        or filename.endswith("_test.go")
        or ".test." in filename
        or ".spec." in filename
    )


def _is_documentation_path(path: str) -> bool:
    pure_path = PurePosixPath(path.lower())
    return (
        "docs" in pure_path.parts[:-1]
        or pure_path.suffix in _DOCUMENTATION_EXTENSIONS
        or pure_path.name.startswith(_DOCUMENTATION_NAMES)
    )


def _is_configuration_path(path: str) -> bool:
    pure_path = PurePosixPath(path.lower())
    filename = pure_path.name
    return (
        ".github" in pure_path.parts[:-1]
        or pure_path.suffix in _CONFIGURATION_EXTENSIONS
        or filename == ".env.example"
        or filename == "dockerfile"
        or filename.startswith(("dockerfile.", "docker-compose."))
        or (
            any(part in {"k8s", "kubernetes"} for part in pure_path.parts[:-1])
            and pure_path.suffix in {".yaml", ".yml"}
        )
    )


def _is_dependency_path(path: str) -> bool:
    filename = PurePosixPath(path.lower()).name
    return (
        filename in _DEPENDENCY_FILENAMES
        or (filename.startswith("requirements-") and filename.endswith(".txt"))
        or filename.endswith((".csproj", ".fsproj", ".vbproj"))
    )


def _sorted_strings(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values), key=_string_sort_key))


def _string_sort_key(value: str) -> tuple[str, str]:
    return value.casefold(), value
