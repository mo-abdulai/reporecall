from collections.abc import Iterable, Sequence

from reporecall.models import MetadataFilter, RetrievalChunk


class MetadataFilterMatcher:
    """Match chunk metadata offline without parsing or reclassifying source text."""

    def matches(self, chunk: RetrievalChunk, metadata_filter: MetadataFilter) -> bool:
        """Return true when every populated filter field matches the chunk."""

        metadata = chunk.metadata
        if (
            metadata_filter.repositories
            and chunk.repository not in metadata_filter.repositories
        ):
            return False
        if (
            metadata_filter.section_types
            and chunk.section_type not in metadata_filter.section_types
        ):
            return False
        if not _overlaps(metadata.issue_numbers, metadata_filter.issue_numbers):
            return False
        if not _overlaps(
            metadata.pull_request_numbers,
            metadata_filter.pull_request_numbers,
        ):
            return False
        if not _overlaps(metadata.commit_shas, metadata_filter.commit_shas):
            return False
        if not _casefold_overlaps(metadata.labels, metadata_filter.labels):
            return False
        if not _overlaps(metadata.milestones, metadata_filter.milestones):
            return False
        if not _casefold_overlaps(metadata.languages, metadata_filter.languages):
            return False
        if not _casefold_overlaps(
            metadata.file_extensions,
            metadata_filter.extensions,
        ):
            return False
        if not _overlaps(metadata.directories, metadata_filter.directories):
            return False
        if metadata_filter.path_prefixes and not any(
            _matches_path_prefix(path, prefix)
            for path in metadata.changed_paths
            for prefix in metadata_filter.path_prefixes
        ):
            return False

        event_actors = (*metadata.authors, *metadata.participants)
        if metadata_filter.actors and not any(
            actor in event_actors for actor in metadata_filter.actors
        ):
            return False

        boolean_fields = (
            (metadata_filter.has_test_changes, bool(metadata.test_paths)),
            (
                metadata_filter.has_documentation_changes,
                bool(metadata.documentation_paths),
            ),
            (
                metadata_filter.has_configuration_changes,
                bool(metadata.configuration_paths),
            ),
            (
                metadata_filter.has_dependency_changes,
                bool(metadata.dependency_paths),
            ),
        )
        if any(expected is not None and expected is not actual for expected, actual in boolean_fields):
            return False

        numeric_fields = (
            (
                metadata.added_lines,
                metadata_filter.min_added_lines,
                metadata_filter.max_added_lines,
            ),
            (
                metadata.deleted_lines,
                metadata_filter.min_deleted_lines,
                metadata_filter.max_deleted_lines,
            ),
            (
                metadata.changed_lines,
                metadata_filter.min_changed_lines,
                metadata_filter.max_changed_lines,
            ),
        )
        return all(
            _within_range(value, minimum, maximum)
            for value, minimum, maximum in numeric_fields
        )


def filter_chunk_ids(
    chunks: Iterable[RetrievalChunk],
    metadata_filter: MetadataFilter,
    *,
    matcher: MetadataFilterMatcher | None = None,
) -> tuple[str, ...]:
    """Return unique matching chunk IDs in caller-provided deterministic order."""

    active_matcher = matcher or MetadataFilterMatcher()
    candidate_ids: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen or not active_matcher.matches(chunk, metadata_filter):
            continue
        seen.add(chunk.chunk_id)
        candidate_ids.append(chunk.chunk_id)
    return tuple(candidate_ids)


def _overlaps[T](stored: Sequence[T], requested: Sequence[T]) -> bool:
    return not requested or bool(set(stored).intersection(requested))


def _casefold_overlaps(stored: Sequence[str], requested: Sequence[str]) -> bool:
    if not requested:
        return True
    normalized_stored = {value.casefold() for value in stored}
    return bool(normalized_stored.intersection(requested))


def _matches_path_prefix(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(f"{prefix}/")


def _within_range(
    value: int,
    minimum: int | None,
    maximum: int | None,
) -> bool:
    return (minimum is None or value >= minimum) and (
        maximum is None or value <= maximum
    )
