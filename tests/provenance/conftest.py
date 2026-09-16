import pytest

from reporecall.github import GitHubRepository
from reporecall.models import (
    ArtifactReference,
    ContextExpansionReason,
    EngineeringRelationship,
    EventMetadata,
    ExpandedContextChunk,
    ExpandedContextResult,
    RerankedSearchHit,
    RetrievalChunk,
    RetrievalDocument,
    RetrievalDocumentSection,
    RetrievalSource,
)
from reporecall.provenance import CitationBundleBuilder, CitationProvenanceIndex


@pytest.fixture
def evidence():
    repository = GitHubRepository(owner="owner", name="repo")
    metadata = EventMetadata(event_id="event", repository=repository)
    refs = tuple(
        ArtifactReference(
            artifact_type="pull_request", repository=repository, identifier=str(i)
        )
        for i in range(6)
    )
    sections = tuple(
        RetrievalDocumentSection(
            section_id=f"section-{i}",
            section_type="pull_request",
            heading="Untrusted heading",
            content="Original text",
            artifact=ref,
        )
        for i, ref in enumerate(refs)
    )
    chunks = tuple(
        RetrievalChunk(
            chunk_id=name,
            document_id="doc",
            event_id="event",
            repository=repository,
            section_id=section.section_id,
            section_type=section.section_type,
            chunk_index=0,
            content="Original text",
            text="Original rendered text",
            artifact=section.artifact,
            metadata=metadata,
        )
        for name, section in zip(("C", "A", "B", "D", "F", "E"), sections, strict=True)
    )
    sources = tuple(
        RetrievalSource(
            artifact=ref,
            url=f"https://example.test/source/{i}?raw=1#anchor",
            label=f"Stored label {i}",
        )
        for i, ref in enumerate(refs)
    )
    doc = RetrievalDocument(
        document_id="doc",
        event_id="event",
        repository=repository,
        metadata=metadata,
        title="Document",
        sections=sections,
        text="Original text",
        sources=sources,
    )
    hits = tuple(
        RerankedSearchHit(
            chunk=c,
            rank=i,
            reranker_score=3.123456789 - i,
            reranker_model="fake",
            previous_hybrid_rank=i,
            rrf_score=1 / (60 + i),
            dense_rank=i,
            dense_score=0.123456789,
            dense_rrf_contribution=1 / (60 + i),
            sources=("dense",),
        )
        for i, c in enumerate(chunks[:3], 1)
    )
    expanded = tuple(
        ExpandedContextChunk(
            chunk=c,
            reasons=tuple(
                ContextExpansionReason(
                    seed_chunk_id=hits[j].chunk.chunk_id,
                    relationship=EngineeringRelationship(
                        source=refs[j],
                        target=c.artifact,
                        relationship_type="pull_request_has_commit",
                        evidence_type="pr_commit_list",
                        evidence="Exact stored evidence",
                        source_field="commits",
                    ),
                    traversal_direction="outgoing",
                )
                for j in (0, 2)
            ),
        )
        for c in chunks[3:]
    )
    context = ExpandedContextResult(
        query="  preserve query\n",
        seed_hits=hits,
        expanded_chunks=expanded,
        truncated=True,
    )
    return doc, chunks, context


@pytest.fixture
def bundle(evidence):
    doc, chunks, context = evidence
    return CitationBundleBuilder(
        provenance_index=CitationProvenanceIndex(documents=[doc], chunks=chunks)
    ).build(context)
