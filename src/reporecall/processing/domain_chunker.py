import hashlib
import re
from itertools import pairwise

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reporecall.models import (
    ArtifactReference,
    ArtifactType,
    RetrievalChunk,
    RetrievalDocument,
    RetrievalDocumentSection,
    RetrievalSectionType,
)

_PARAGRAPH_BOUNDARY = re.compile(r"\n(?:[ \t]*\n)+")
_WORD_BOUNDARY = re.compile(r"[ \t]+")

_PROSE_SECTION_TYPES = frozenset(
    {
        RetrievalSectionType.ISSUE,
        RetrievalSectionType.ISSUE_COMMENT,
        RetrievalSectionType.PULL_REQUEST,
        RetrievalSectionType.PULL_REQUEST_COMMENT,
        RetrievalSectionType.REVIEW,
        RetrievalSectionType.REVIEW_COMMENT,
        RetrievalSectionType.COMMIT,
    }
)

_STRUCTURED_SECTION_TYPES = frozenset(
    {
        RetrievalSectionType.OVERVIEW,
        RetrievalSectionType.METADATA,
        RetrievalSectionType.CHANGED_FILE,
        RetrievalSectionType.RELATIONSHIP,
        RetrievalSectionType.CONTEXTUAL_RELATIONSHIP,
    }
)


class ChunkingConfig(BaseModel):
    """Deterministic character and patch-line limits used by the chunker."""

    max_chars: int = Field(default=2_000, gt=0)
    overlap_chars: int = Field(default=200, ge=0)
    patch_max_lines: int = Field(default=120, gt=0)
    patch_overlap_lines: int = Field(default=12, ge=0)

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars.")
        if self.patch_overlap_lines >= self.patch_max_lines:
            raise ValueError(
                "patch_overlap_lines must be smaller than patch_max_lines."
            )
        return self


class DomainAwareChunker:
    """Split document sections deterministically without crossing domain boundaries."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, document: RetrievalDocument) -> list[RetrievalChunk]:
        """Return chunks in document-section and section-local index order."""

        chunks: list[RetrievalChunk] = []
        for section in document.sections:
            content = _normalize_content(section.content)
            if not content.strip():
                continue

            excerpts = self._split_section(section, content)
            chunks.extend(
                _build_chunk(document, section, excerpt, chunk_index)
                for chunk_index, excerpt in enumerate(excerpts)
            )
        return chunks

    def _split_section(
        self,
        section: RetrievalDocumentSection,
        content: str,
    ) -> tuple[str, ...]:
        if section.section_type is RetrievalSectionType.PATCH:
            return _split_patch(
                content,
                max_lines=self.config.patch_max_lines,
                overlap_lines=self.config.patch_overlap_lines,
            )
        if section.section_type in _PROSE_SECTION_TYPES:
            return _split_prose(
                content,
                max_chars=self.config.max_chars,
                overlap_chars=self.config.overlap_chars,
            )
        if section.section_type in _STRUCTURED_SECTION_TYPES:
            return _split_prose(
                content,
                max_chars=self.config.max_chars,
                overlap_chars=self.config.overlap_chars,
            )
        raise ValueError(f"Unsupported retrieval section type: {section.section_type}")


def _build_chunk(
    document: RetrievalDocument,
    section: RetrievalDocumentSection,
    content: str,
    chunk_index: int,
) -> RetrievalChunk:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    chunk_id = (
        f"{document.document_id}__{section.section_id}__"
        f"{chunk_index:04d}__{digest}"
    )
    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id=document.document_id,
        event_id=document.event_id,
        repository=document.repository,
        section_id=section.section_id,
        section_type=section.section_type,
        chunk_index=chunk_index,
        content=content,
        text=_retrieval_text(document, section, content),
        artifact=section.artifact,
        metadata=document.metadata,
    )


def _retrieval_text(
    document: RetrievalDocument,
    section: RetrievalDocumentSection,
    content: str,
) -> str:
    lines = [
        f"Repository: {document.repository.owner}/{document.repository.name}",
        f"Event ID: {document.event_id}",
        f"Document ID: {document.document_id}",
        f"Section: {_section_label(section.section_type)}",
    ]
    if section.artifact is not None:
        lines.append(f"Artifact: {_artifact_label(section.artifact)}")
    return "\n".join(lines) + "\n\n" + content


def _split_prose(
    content: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> tuple[str, ...]:
    if len(content) <= max_chars:
        return (content,)

    excerpts: list[str] = []
    start = 0
    while start < len(content):
        hard_end = min(start + max_chars, len(content))
        end = (
            len(content)
            if hard_end == len(content)
            else _preferred_prose_end(content, start, hard_end)
        )
        excerpt = content[start:end].strip("\n")
        if excerpt.strip():
            excerpts.append(excerpt)
        if end >= len(content):
            break

        excerpt_length = end - start
        start = (
            end - overlap_chars
            if overlap_chars and excerpt_length > overlap_chars
            else end
        )

    return tuple(excerpts)


def _preferred_prose_end(content: str, start: int, hard_end: int) -> int:
    paragraph_end = _last_match_end(_PARAGRAPH_BOUNDARY, content, start, hard_end)
    if paragraph_end is not None:
        return paragraph_end

    line_break = content.rfind("\n", start + 1, hard_end + 1)
    if line_break >= start:
        return line_break + 1

    word_end = _last_match_end(_WORD_BOUNDARY, content, start, hard_end)
    if word_end is not None:
        return word_end

    return hard_end


def _last_match_end(
    pattern: re.Pattern[str],
    content: str,
    start: int,
    end: int,
) -> int | None:
    match_end: int | None = None
    for match in pattern.finditer(content, start, end):
        if match.end() > start:
            match_end = match.end()
    return match_end


def _split_patch(
    content: str,
    *,
    max_lines: int,
    overlap_lines: int,
) -> tuple[str, ...]:
    lines = content.split("\n")
    if len(lines) <= max_lines:
        return (content,)

    excerpts: list[str] = []
    pending: list[str] = []
    for unit in _patch_units(lines):
        if len(unit) > max_lines:
            if pending:
                excerpts.append("\n".join(pending))
                pending = []
            excerpts.extend(
                _split_patch_lines(
                    unit,
                    max_lines=max_lines,
                    overlap_lines=overlap_lines,
                )
            )
            continue

        if pending and len(pending) + len(unit) > max_lines:
            excerpts.append("\n".join(pending))
            pending = []
        pending.extend(unit)

    if pending:
        excerpts.append("\n".join(pending))
    return tuple(excerpt for excerpt in excerpts if excerpt.strip())


def _patch_units(lines: list[str]) -> tuple[list[str], ...]:
    starts = [0]
    starts.extend(
        index
        for index, line in enumerate(lines[1:], start=1)
        if line.startswith(("diff --git ", "@@"))
    )
    starts.append(len(lines))
    return tuple(
        lines[start:end]
        for start, end in pairwise(starts)
        if start < end
    )


def _split_patch_lines(
    lines: list[str],
    *,
    max_lines: int,
    overlap_lines: int,
) -> list[str]:
    excerpts: list[str] = []
    start = 0
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        excerpts.append("\n".join(lines[start:end]))
        if end == len(lines):
            break
        start = end - overlap_lines
    return excerpts


def _normalize_content(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n").strip("\n")


def _section_label(section_type: RetrievalSectionType) -> str:
    return section_type.value.replace("_", " ").title()


def _artifact_label(artifact: ArtifactReference) -> str:
    prefixes = {
        ArtifactType.ISSUE: "Issue",
        ArtifactType.ISSUE_COMMENT: "Issue Comment",
        ArtifactType.PULL_REQUEST: "Pull Request",
        ArtifactType.PULL_REQUEST_FILE: "Pull Request File",
        ArtifactType.PULL_REQUEST_COMMENT: "Pull Request Comment",
        ArtifactType.PULL_REQUEST_REVIEW: "Review",
        ArtifactType.PULL_REQUEST_REVIEW_COMMENT: "Review Comment",
        ArtifactType.GITHUB_COMMIT_REFERENCE: "GitHub Commit",
        ArtifactType.LOCAL_GIT_COMMIT: "Local Commit",
        ArtifactType.LOCAL_CHANGED_FILE: "Local Changed File",
    }
    identifier = artifact.identifier
    numbered_types = {
        ArtifactType.ISSUE,
        ArtifactType.PULL_REQUEST,
    }
    prefixed_number_types = {
        ArtifactType.ISSUE_COMMENT: "issue-comment:",
        ArtifactType.PULL_REQUEST_COMMENT: "pr-comment:",
        ArtifactType.PULL_REQUEST_REVIEW: "review:",
        ArtifactType.PULL_REQUEST_REVIEW_COMMENT: "review-comment:",
    }
    if artifact.artifact_type in numbered_types:
        identifier = f"#{identifier}"
    elif artifact.artifact_type in prefixed_number_types:
        prefix = prefixed_number_types[artifact.artifact_type]
        identifier = f"#{identifier.removeprefix(prefix)}"
    return f"{prefixes[artifact.artifact_type]} {identifier}"
