"""Translate canonical filter semantics to bound PostgreSQL predicates."""

from sqlalchemy import ColumnElement, and_, or_

from reporecall.models import MetadataFilter
from reporecall.persistence.orm import ChunkRecord


def metadata_predicates(
    metadata_filter: MetadataFilter | None,
) -> list[ColumnElement[bool]]:
    """AND across fields, OR within fields; eligibility precedes vector LIMIT."""
    if metadata_filter is None:
        return []
    filters = MetadataFilter.model_validate(metadata_filter.model_dump())
    conditions: list[ColumnElement[bool]] = []
    if filters.repositories:
        conditions.append(
            or_(
                *(
                    and_(
                        ChunkRecord.repository_owner == r.owner,
                        ChunkRecord.repository_name == r.name,
                    )
                    for r in filters.repositories
                )
            )
        )
    if filters.section_types:
        conditions.append(
            ChunkRecord.section_type.in_([s.value for s in filters.section_types])
        )
    fields = {
        "issue_numbers": "issue_numbers",
        "pull_request_numbers": "pull_request_numbers",
        "commit_shas": "commit_shas",
        "labels": "labels",
        "milestones": "milestones",
        "languages": "languages",
        "extensions": "file_extensions",
        "directories": "directories",
        "path_prefixes": "path_prefixes",
    }
    data = ChunkRecord.filter_metadata
    for name, key in fields.items():
        values = getattr(filters, name)
        if values:
            conditions.append(or_(*(data[key].contains([value]) for value in values)))
    if filters.actors:
        conditions.append(
            or_(
                *(
                    data["actors"].contains([a.model_dump(mode="json")])
                    for a in filters.actors
                )
            )
        )
    for name, key in (
        ("has_test_changes", "has_tests"),
        ("has_documentation_changes", "has_documentation_changes"),
        ("has_configuration_changes", "has_configuration_changes"),
        ("has_dependency_changes", "has_dependency_changes"),
    ):
        value = getattr(filters, name)
        if value is not None:
            conditions.append(data[key].as_boolean() == value)
    for key in ("added_lines", "deleted_lines", "changed_lines"):
        minimum, maximum = (
            getattr(filters, f"min_{key}"),
            getattr(filters, f"max_{key}"),
        )
        if minimum is not None:
            conditions.append(data[key].as_integer() >= minimum)
        if maximum is not None:
            conditions.append(data[key].as_integer() <= maximum)
    return conditions
