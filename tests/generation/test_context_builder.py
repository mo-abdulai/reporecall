from copy import deepcopy

import pytest

from reporecall.generation import RAGContextBuilder, RAGContextError
from reporecall.github import GitHubRepository
from reporecall.models import (
    ArtifactReference,
    ArtifactType,
    EventMetadata,
    RetrievalChunk,
    RetrievalSectionType,
    VectorSearchHit,
)


def test_context_preserves_rank_order_scores_labels_and_identity():
    hits = [
        _hit("chunk-b", rank=1, score=0.684866, content="first evidence"),
        _hit("chunk-a", rank=2, score=0.25, content="second evidence"),
        _hit("chunk-c", rank=3, score=0.1, content="third evidence"),
    ]

    context = RAGContextBuilder().build(
        "database leak",
        hits,
        max_context_chars=10_000,
    )

    assert [item.evidence_id for item in context.evidence] == ["E1", "E2", "E3"]
    assert [item.chunk_id for item in context.evidence] == [
        "chunk-b",
        "chunk-a",
        "chunk-c",
    ]
    assert [item.rank for item in context.evidence] == [1, 2, 3]
    assert context.evidence[0].score == 0.684866
    assert context.text.index("EVIDENCE E1") < context.text.index("EVIDENCE E2")
    assert context.text.index("EVIDENCE E2") < context.text.index("EVIDENCE E3")
    assert "Similarity: 0.68486599999999997" in context.text
    assert "Repository: owner/repo" in context.text
    assert "Artifact: issue:10" in context.text
    assert context.retrieved_count == 3
    assert context.included_evidence_count == 3
    assert context.truncated is False


def test_context_orders_by_retrieval_rank_without_alphabetic_reordering():
    hits = [
        _hit("chunk-a", rank=2, score=0.2, content="rank two"),
        _hit("chunk-b", rank=1, score=0.8, content="rank one"),
    ]

    context = RAGContextBuilder().build("query", hits, max_context_chars=10_000)

    assert [item.chunk_id for item in context.evidence] == ["chunk-b", "chunk-a"]


def test_context_budget_excludes_lower_ranked_evidence_completely():
    first = _hit("chunk-1", rank=1, score=0.9, content="first")
    second = _hit("chunk-2", rank=2, score=0.8, content="second")
    builder = RAGContextBuilder()
    first_only = builder.build("query", [first], max_context_chars=10_000)

    context = builder.build(
        "query",
        [first, second],
        max_context_chars=len(first_only.text),
    )

    assert [item.chunk_id for item in context.evidence] == ["chunk-1"]
    assert "chunk-2" not in context.text
    assert context.retrieved_count == 2
    assert context.included_evidence_count == 1
    assert context.truncated is True


def test_oversized_first_hit_includes_deterministic_bounded_prefix():
    content = "0123456789" * 100
    hit = _hit("chunk-1", rank=1, score=0.9, content=content)
    full = RAGContextBuilder().build("query", [hit], max_context_chars=10_000)
    budget = len(full.text) - 250

    first = RAGContextBuilder().build("query", [hit], max_context_chars=budget)
    second = RAGContextBuilder().build("query", [hit], max_context_chars=budget)

    assert first == second
    assert len(first.text) == budget
    assert first.text == full.text[:budget]
    assert first.evidence[0].content == content[: len(first.evidence[0].content)]
    assert first.evidence[0].content_truncated is True
    assert first.truncated is True


def test_tiny_positive_budget_still_records_the_bounded_first_evidence():
    context = RAGContextBuilder().build(
        "query",
        [_hit("chunk-1", rank=1, score=1.0, content="source")],
        max_context_chars=1,
    )

    assert len(context.text) == 1
    assert context.evidence[0].content == ""
    assert context.evidence[0].content_truncated is True


def test_empty_hits_produce_empty_untruncated_context():
    context = RAGContextBuilder().build("query", [], max_context_chars=100)

    assert context.evidence == ()
    assert context.text == ""
    assert context.retrieved_count == 0
    assert context.included_evidence_count == 0
    assert context.truncated is False


@pytest.mark.parametrize("query", ["", "   "])
def test_context_rejects_blank_query(query: str):
    with pytest.raises(RAGContextError, match="must not be blank"):
        RAGContextBuilder().build(query, [], max_context_chars=100)


@pytest.mark.parametrize("budget", [0, -1])
def test_context_rejects_nonpositive_budget(budget: int):
    with pytest.raises(RAGContextError, match="greater than zero"):
        RAGContextBuilder().build("query", [], max_context_chars=budget)


def test_context_rejects_duplicate_retrieval_ranks():
    hits = [
        _hit("chunk-1", rank=1, score=0.9, content="first"),
        _hit("chunk-2", rank=1, score=0.8, content="second"),
    ]

    with pytest.raises(RAGContextError, match="unique ranks"):
        RAGContextBuilder().build("query", hits, max_context_chars=10_000)


def test_multiple_section_types_patch_and_stack_trace_remain_source_faithful():
    patch = "@@ -20,4 +20,8 @@\n- connection.release()\n+ connection.close()"
    stack = (
        "Traceback (most recent call last):\n"
        "  File \"worker.py\", line 42, in retry_worker\n"
        "ConnectionError: pool exhausted"
    )
    section_types = [
        RetrievalSectionType.ISSUE,
        RetrievalSectionType.REVIEW_COMMENT,
        RetrievalSectionType.PATCH,
        RetrievalSectionType.COMMIT,
    ]
    hits = [
        _hit(
            f"chunk-{index}",
            rank=index,
            score=1.0 / index,
            content=patch if section_type is RetrievalSectionType.PATCH else stack,
            section_type=section_type,
        )
        for index, section_type in enumerate(section_types, start=1)
    ]

    context = RAGContextBuilder().build("query", hits, max_context_chars=20_000)

    assert [item.section_type for item in context.evidence] == section_types
    assert patch in context.text
    assert stack in context.text


def test_context_construction_does_not_mutate_hits_or_chunks():
    hits = [_hit("chunk-1", rank=1, score=0.9, content="source")]
    original = deepcopy([hit.model_dump() for hit in hits])

    RAGContextBuilder().build("query", hits, max_context_chars=10_000)

    assert [hit.model_dump() for hit in hits] == original


def _hit(
    chunk_id: str,
    *,
    rank: int,
    score: float,
    content: str,
    section_type: RetrievalSectionType = RetrievalSectionType.ISSUE,
) -> VectorSearchHit:
    repository = GitHubRepository(owner="owner", name="repo")
    artifact = ArtifactReference(
        artifact_type=ArtifactType.ISSUE,
        repository=repository,
        identifier="10",
    )
    chunk = RetrievalChunk(
        chunk_id=chunk_id,
        document_id="document-1",
        event_id="event-1",
        repository=repository,
        section_id=f"section-{chunk_id}",
        section_type=section_type,
        chunk_index=0,
        content=content,
        text=f"Repository: owner/repo\n\n{content}",
        artifact=artifact,
        metadata=EventMetadata(event_id="event-1", repository=repository),
    )
    return VectorSearchHit(chunk=chunk, score=score, rank=rank)
