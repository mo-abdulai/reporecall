import re
from itertools import pairwise

import pytest
from pydantic import ValidationError

import reporecall.processing.domain_chunker as chunker_module
from reporecall.github import GitHubRepository
from reporecall.models import (
    ArtifactReference,
    ArtifactType,
    EventMetadata,
    RetrievalDocument,
    RetrievalDocumentSection,
    RetrievalSectionType,
)
from reporecall.processing import ChunkingConfig, DomainAwareChunker


def test_chunking_config_defaults_are_immutable_and_model_independent():
    config = ChunkingConfig()

    assert config.max_chars == 2_000
    assert config.overlap_chars == 200
    assert config.patch_max_lines == 120
    assert config.patch_overlap_lines == 12
    with pytest.raises(ValidationError, match="frozen"):
        config.max_chars = 10


@pytest.mark.parametrize(
    "values",
    [
        {"max_chars": 0},
        {"max_chars": -1},
        {"overlap_chars": -1},
        {"max_chars": 10, "overlap_chars": 10},
        {"max_chars": 10, "overlap_chars": 11},
        {"patch_max_lines": 0},
        {"patch_overlap_lines": -1},
        {"patch_max_lines": 5, "patch_overlap_lines": 5},
        {"patch_max_lines": 5, "patch_overlap_lines": 6},
    ],
)
def test_chunking_config_rejects_invalid_limits(values: dict[str, int]):
    with pytest.raises(ValidationError):
        ChunkingConfig(**values)


def test_small_issue_section_stays_whole_with_retrieval_context():
    artifact = _artifact(ArtifactType.ISSUE, "421")
    content = "Title: Pool exhausted\n\nBody:\nConnectionError in retry_task"
    document = _document(
        [_section("issue-421", RetrievalSectionType.ISSUE, content, artifact)]
    )

    chunks = DomainAwareChunker(
        ChunkingConfig(max_chars=200, overlap_chars=0)
    ).chunk(document)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.content == content
    assert chunk.document_id == document.document_id
    assert chunk.event_id == document.event_id
    assert chunk.repository == document.repository
    assert chunk.section_id == "issue-421"
    assert chunk.section_type is RetrievalSectionType.ISSUE
    assert chunk.chunk_index == 0
    assert chunk.artifact == artifact
    assert chunk.metadata == document.metadata
    assert chunk.text == (
        "Repository: owner/repo\n"
        "Event ID: event\n"
        "Document ID: event__retrieval\n"
        "Section: Issue\n"
        "Artifact: Issue #421\n\n"
        f"{content}"
    )
    assert re.fullmatch(
        r"event__retrieval__issue-421__0000__[0-9a-f]{16}",
        chunk.chunk_id,
    )


def test_section_isolation_never_combines_small_domains():
    document = _document(
        [
            _section("issue-1", RetrievalSectionType.ISSUE, "AAA"),
            _section("pr-2", RetrievalSectionType.PULL_REQUEST, "BBB"),
            _section("review-3", RetrievalSectionType.REVIEW, "CCC"),
        ]
    )

    chunks = DomainAwareChunker(
        ChunkingConfig(max_chars=100, overlap_chars=0)
    ).chunk(document)

    assert [chunk.content for chunk in chunks] == ["AAA", "BBB", "CCC"]
    assert [chunk.section_id for chunk in chunks] == ["issue-1", "pr-2", "review-3"]
    assert [chunk.chunk_index for chunk in chunks] == [0, 0, 0]
    assert all(
        other not in chunk.content
        for chunk, others in zip(
            chunks,
            [("BBB", "CCC"), ("AAA", "CCC"), ("AAA", "BBB")],
            strict=True,
        )
        for other in others
    )


def test_large_prose_prefers_paragraph_boundaries_and_uses_overlap():
    first = "alpha paragraph one"
    second = "beta paragraph two!"
    third = "gamma paragraph end"
    content = f"{first}\n\n{second}\n\n{third}"
    document = _document(
        [_section("issue-1", RetrievalSectionType.ISSUE, content)]
    )
    config = ChunkingConfig(max_chars=45, overlap_chars=8)

    chunks = DomainAwareChunker(config).chunk(document)

    assert len(chunks) == 2
    assert chunks[0].content == f"{first}\n\n{second}"
    assert chunks[1].content.endswith(third)
    assert second[-6:] in chunks[1].content
    assert all(len(chunk.content) <= config.max_chars for chunk in chunks)
    assert first in chunks[0].content
    assert third in chunks[-1].content


def test_oversized_paragraph_prefers_lines_then_words():
    content = (
        "Traceback entry in worker.py\n"
        "retry_task raised ConnectionError\n"
        "sqlalchemy pool unavailable"
    )
    document = _document(
        [_section("comment-1", RetrievalSectionType.ISSUE_COMMENT, content)]
    )

    chunks = DomainAwareChunker(
        ChunkingConfig(max_chars=36, overlap_chars=0)
    ).chunk(document)

    assert len(chunks) == 3
    assert chunks[0].content == "Traceback entry in worker.py"
    assert chunks[1].content == "retry_task raised ConnectionError"
    assert chunks[2].content == "sqlalchemy pool unavailable"
    assert "worker.py" in chunks[0].content
    assert "ConnectionError" in chunks[1].content
    assert "sqlalchemy" in chunks[2].content


def test_hard_character_fallback_has_exact_overlap_and_loses_no_content():
    content = "0123456789ABCDEFGHIJKLMNOPQRSTUV"
    document = _document(
        [_section("commit-a", RetrievalSectionType.COMMIT, content)]
    )
    overlap = 3

    chunks = DomainAwareChunker(
        ChunkingConfig(max_chars=10, overlap_chars=overlap)
    ).chunk(document)

    assert [len(chunk.content) for chunk in chunks] == [10, 10, 10, 10, 4]
    assert all(
        current.content[-overlap:] == following.content[:overlap]
        for current, following in pairwise(chunks)
    )
    reconstructed = chunks[0].content + "".join(
        chunk.content[overlap:] for chunk in chunks[1:]
    )
    assert reconstructed == content


def test_code_block_indentation_and_special_characters_survive_chunking():
    content = (
        "Example:\n\n"
        "```python\n"
        "def retry_task(path: str) -> None:\n"
        "    raise ConnectionError(f\"failed: {path}\")\n"
        "```\n\n"
        "Path: src/workers/retry.py\n"
        "SHA: abc123/def456\n"
        "Unicode: café"
    )
    document = _document(
        [_section("pr-1", RetrievalSectionType.PULL_REQUEST, content)]
    )

    chunks = DomainAwareChunker(
        ChunkingConfig(max_chars=75, overlap_chars=10)
    ).chunk(document)
    rendered = "\n".join(chunk.content for chunk in chunks)

    assert "    raise ConnectionError" in rendered
    assert "`python" in rendered
    assert "src/workers/retry.py" in rendered
    assert "abc123/def456" in rendered
    assert "café" in rendered


def test_review_comment_preserves_artifact_and_parent_context():
    artifact = _artifact(
        ArtifactType.PULL_REQUEST_REVIEW_COMMENT,
        "review-comment:991",
    )
    content = "File: src/database/session.py\n\nRelease this in finally."
    document = _document(
        [
            _section(
                "review-comment-991",
                RetrievalSectionType.REVIEW_COMMENT,
                content,
                artifact,
            )
        ]
    )

    chunk = DomainAwareChunker().chunk(document)[0]

    assert chunk.content == content
    assert chunk.artifact == artifact
    assert chunk.section_type is RetrievalSectionType.REVIEW_COMMENT
    assert "Artifact: Review Comment #991" in chunk.text
    assert "src/database/session.py" in chunk.text


def test_long_commit_message_keeps_commit_and_section_identity():
    artifact = _artifact(ArtifactType.LOCAL_GIT_COMMIT, "abc123")
    content = (
        "SHA: abc123\n\n"
        "Release connection after retry failure.\n\n"
        "Keep DatabaseSession.close() in the finally block."
    )
    document = _document(
        [_section("commit-abc123", RetrievalSectionType.COMMIT, content, artifact)]
    )

    chunks = DomainAwareChunker(
        ChunkingConfig(max_chars=60, overlap_chars=5)
    ).chunk(document)

    assert len(chunks) > 1
    assert all(chunk.section_id == "commit-abc123" for chunk in chunks)
    assert all(chunk.artifact == artifact for chunk in chunks)
    assert "DatabaseSession.close()" in "\n".join(
        chunk.content for chunk in chunks
    )


def test_metadata_remains_structured_and_normally_uses_one_chunk():
    metadata = EventMetadata(
        event_id="event",
        repository=_repository(),
        labels=("bug",),
        languages=("Python",),
        changed_paths=("src/app.py",),
    )
    document = _document(
        [
            _section(
                "metadata",
                RetrievalSectionType.METADATA,
                "Labels: bug\nLanguages: Python\nChanged Paths:\n- src/app.py",
            )
        ],
        metadata=metadata,
    )

    chunks = DomainAwareChunker().chunk(document)

    assert len(chunks) == 1
    assert chunks[0].section_type is RetrievalSectionType.METADATA
    assert chunks[0].metadata is metadata
    assert chunks[0].metadata.languages == ("Python",)


def test_direct_and_contextual_relationships_remain_distinct():
    document = _document(
        [
            _section(
                "relationships",
                RetrievalSectionType.RELATIONSHIP,
                "PR #20 [pull_request_closes_issue] Issue #10",
            ),
            _section(
                "contextual-relationships",
                RetrievalSectionType.CONTEXTUAL_RELATIONSHIP,
                "PR #20 [pull_request_references_issue] Issue #99",
            ),
        ]
    )

    chunks = DomainAwareChunker().chunk(document)

    assert [chunk.section_type for chunk in chunks] == [
        RetrievalSectionType.RELATIONSHIP,
        RetrievalSectionType.CONTEXTUAL_RELATIONSHIP,
    ]
    assert chunks[0].section_id == "relationships"
    assert chunks[1].section_id == "contextual-relationships"
    assert "closes" not in chunks[1].content


def test_small_patch_stays_whole_and_preserves_diff_markers():
    content = (
        "Commit: abc123\n"
        "File: src/app.py\n\n"
        "Patch:\n"
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1 +1 @@\n"
        "-old_call()\n"
        "+new_call()"
    )
    document = _document(
        [_section("patch-a", RetrievalSectionType.PATCH, content)]
    )

    chunks = DomainAwareChunker(
        ChunkingConfig(patch_max_lines=20, patch_overlap_lines=2)
    ).chunk(document)

    assert len(chunks) == 1
    assert chunks[0].content == content
    for marker in ("diff --git", "--- ", "+++ ", "@@", "-old_call()", "+new_call()"):
        assert marker in chunks[0].content


def test_multiple_patch_hunks_are_packed_without_splitting_fitting_hunks():
    content = (
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-old_one\n"
        "+new_one\n"
        "@@ -10,2 +10,2 @@\n"
        "-old_two\n"
        "+new_two\n"
        "@@ -20,2 +20,2 @@\n"
        "-old_three\n"
        "+new_three"
    )
    document = _document(
        [_section("patch-a", RetrievalSectionType.PATCH, content)]
    )

    chunks = DomainAwareChunker(
        ChunkingConfig(patch_max_lines=6, patch_overlap_lines=1)
    ).chunk(document)

    assert [
        sum(line.startswith("@@") for line in chunk.content.splitlines())
        for chunk in chunks
    ] == [1, 2]
    assert all(len(chunk.content.splitlines()) <= 6 for chunk in chunks)
    assert "\n".join(chunk.content for chunk in chunks) == content


def test_oversized_hunk_uses_deterministic_line_overlap_without_loss():
    lines = [
        "@@ -1,6 +1,6 @@",
        " context_one",
        "-old_one",
        "+new_one",
        " context_two",
        "-old_two",
        "+new_two",
    ]
    content = "\n".join(lines)
    document = _document(
        [_section("patch-a", RetrievalSectionType.PATCH, content)]
    )
    overlap = 1

    chunks = DomainAwareChunker(
        ChunkingConfig(patch_max_lines=4, patch_overlap_lines=overlap)
    ).chunk(document)

    assert [chunk.content.splitlines() for chunk in chunks] == [
        lines[:4],
        lines[3:7],
    ]
    reconstructed = chunks[0].content.splitlines() + chunks[1].content.splitlines()[
        overlap:
    ]
    assert reconstructed == lines
    assert chunks == DomainAwareChunker(
        ChunkingConfig(patch_max_lines=4, patch_overlap_lines=overlap)
    ).chunk(document)


def test_patch_sections_for_same_path_keep_distinct_provenance_and_ids():
    sections = [
        _section(
            "patch-aaa-src%2Fapp.py",
            RetrievalSectionType.PATCH,
            "@@ first @@\n-old\n+new",
            _artifact(ArtifactType.LOCAL_CHANGED_FILE, "aaa:src/app.py"),
        ),
        _section(
            "patch-bbb-src%2Fapp.py",
            RetrievalSectionType.PATCH,
            "@@ second @@\n-before\n+after",
            _artifact(ArtifactType.LOCAL_CHANGED_FILE, "bbb:src/app.py"),
        ),
    ]

    chunks = DomainAwareChunker().chunk(_document(sections))

    assert [chunk.section_id for chunk in chunks] == [
        "patch-aaa-src%2Fapp.py",
        "patch-bbb-src%2Fapp.py",
    ]
    assert [chunk.artifact.identifier for chunk in chunks if chunk.artifact] == [
        "aaa:src/app.py",
        "bbb:src/app.py",
    ]
    assert len({chunk.chunk_id for chunk in chunks}) == 2


def test_domain_order_and_section_local_indices_are_stable_and_unique():
    document = _document(
        [
            _section("issue", RetrievalSectionType.ISSUE, "A" * 25),
            _section("pr", RetrievalSectionType.PULL_REQUEST, "BBB"),
            _section("patch", RetrievalSectionType.PATCH, "\n".join(f"+{i}" for i in range(7))),
        ]
    )
    chunker = DomainAwareChunker(
        ChunkingConfig(
            max_chars=10,
            overlap_chars=2,
            patch_max_lines=4,
            patch_overlap_lines=1,
        )
    )

    chunks = chunker.chunk(document)

    assert [(chunk.section_id, chunk.chunk_index) for chunk in chunks] == [
        ("issue", 0),
        ("issue", 1),
        ("issue", 2),
        ("pr", 0),
        ("patch", 0),
        ("patch", 1),
    ]
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert chunks == chunker.chunk(document)


def test_configuration_changes_chunk_content_and_digest_identity():
    content = "abcdefghijklmnopqrstuvwxyz0123456789"
    document = _document(
        [_section("issue", RetrievalSectionType.ISSUE, content)]
    )

    wider = DomainAwareChunker(
        ChunkingConfig(max_chars=14, overlap_chars=2)
    ).chunk(document)
    narrower = DomainAwareChunker(
        ChunkingConfig(max_chars=10, overlap_chars=2)
    ).chunk(document)

    assert [chunk.content for chunk in wider] != [chunk.content for chunk in narrower]
    assert {chunk.chunk_id for chunk in wider} != {
        chunk.chunk_id for chunk in narrower
    }


def test_chunking_does_not_mutate_document_sections_or_metadata():
    document = _document(
        [_section("issue", RetrievalSectionType.ISSUE, "A" * 25)]
    )
    original = document.model_dump()

    DomainAwareChunker(ChunkingConfig(max_chars=10, overlap_chars=2)).chunk(document)

    assert document.model_dump() == original


def test_blank_section_is_defensively_skipped():
    document = _document(
        [_section("blank", RetrievalSectionType.OVERVIEW, "   ")]
    )

    assert DomainAwareChunker().chunk(document) == []


def test_newline_normalization_produces_platform_independent_chunks():
    unix_document = _document(
        [_section("issue", RetrievalSectionType.ISSUE, "first\nsecond\nthird")]
    )
    windows_document = _document(
        [_section("issue", RetrievalSectionType.ISSUE, "first\r\nsecond\r\nthird")]
    )
    chunker = DomainAwareChunker(ChunkingConfig(max_chars=12, overlap_chars=0))

    unix_chunks = chunker.chunk(unix_document)
    windows_chunks = chunker.chunk(windows_document)

    assert windows_chunks == unix_chunks
    assert all("\r" not in chunk.content for chunk in windows_chunks)


def test_chunker_has_no_io_embedding_search_or_tokenizer_dependencies():
    module_attributes = vars(chunker_module)

    for forbidden_name in (
        "GitHubClient",
        "httpx",
        "git",
        "OpenAI",
        "Embedding",
        "SearchResult",
        "tiktoken",
        "transformers",
    ):
        assert forbidden_name not in module_attributes


def _document(
    sections: list[RetrievalDocumentSection],
    *,
    metadata: EventMetadata | None = None,
) -> RetrievalDocument:
    event_metadata = metadata or EventMetadata(
        event_id="event",
        repository=_repository(),
    )
    return RetrievalDocument(
        document_id="event__retrieval",
        event_id="event",
        repository=_repository(),
        title="Example Event",
        sections=sections,
        text="Canonical document text",
        metadata=event_metadata,
    )


def _section(
    section_id: str,
    section_type: RetrievalSectionType,
    content: str,
    artifact: ArtifactReference | None = None,
) -> RetrievalDocumentSection:
    return RetrievalDocumentSection(
        section_id=section_id,
        section_type=section_type,
        heading=section_type.value.replace("_", " ").upper(),
        content=content,
        artifact=artifact,
    )


def _artifact(artifact_type: ArtifactType, identifier: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_type=artifact_type,
        repository=_repository(),
        identifier=identifier,
    )


def _repository() -> GitHubRepository:
    return GitHubRepository(owner="owner", name="repo")
