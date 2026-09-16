import pytest

from reporecall.github import GitHubRepository
from reporecall.provenance import CitationProvenanceError, CitationProvenanceIndex


def test_duplicate_identical_records_allowed_and_conflicts_rejected(evidence):
    doc, chunks, _ = evidence
    index = CitationProvenanceIndex(documents=[doc, doc], chunks=[*chunks, chunks[0]])
    assert index.sources_for_chunk(chunks[0])
    with pytest.raises(CitationProvenanceError, match="duplicate document"):
        CitationProvenanceIndex(
            documents=[doc, doc.model_copy(update={"title": "Conflict"})], chunks=[]
        )
    with pytest.raises(CitationProvenanceError, match="duplicate chunk"):
        CitationProvenanceIndex(
            documents=[doc],
            chunks=[chunks[0], chunks[0].model_copy(update={"text": "Conflict"})],
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("document_id", "absent"),
        ("event_id", "wrong"),
        ("section_id", "absent"),
        ("section_type", "patch"),
        ("artifact", None),
    ],
)
def test_chunk_document_consistency(evidence, field, value):
    doc, chunks, _ = evidence
    with pytest.raises(CitationProvenanceError):
        CitationProvenanceIndex(
            documents=[doc], chunks=[chunks[0].model_copy(update={field: value})]
        )


def test_missing_or_stale_chunk(evidence):
    doc, chunks, _ = evidence
    with pytest.raises(CitationProvenanceError, match="Missing document"):
        CitationProvenanceIndex(documents=[], chunks=chunks)
    index = CitationProvenanceIndex(documents=[doc], chunks=chunks)
    with pytest.raises(CitationProvenanceError, match="Missing indexed chunk"):
        index.sources_for_chunk(chunks[0].model_copy(update={"chunk_id": "unknown"}))
    with pytest.raises(CitationProvenanceError, match="conflicts"):
        index.sources_for_chunk(chunks[0].model_copy(update={"text": "stale"}))
    with pytest.raises(CitationProvenanceError, match="Missing document"):
        index.artifact_sources("missing", chunks[0].artifact)


def test_repository_consistency(evidence):
    doc, chunks, _ = evidence
    other = GitHubRepository(owner="owner", name="other")
    with pytest.raises(CitationProvenanceError, match="event/repository"):
        CitationProvenanceIndex(
            documents=[doc], chunks=[chunks[0].model_copy(update={"repository": other})]
        )
    source = doc.sources[0].model_copy(
        update={
            "artifact": doc.sources[0].artifact.model_copy(update={"repository": other})
        }
    )
    with pytest.raises(CitationProvenanceError, match="Source repository"):
        CitationProvenanceIndex(
            documents=[doc.model_copy(update={"sources": (source,)})], chunks=[]
        )


def test_document_event_and_repository_isolation(evidence):
    doc, chunks, _ = evidence
    other_repo = GitHubRepository(owner="owner", name="other")
    other_artifact = chunks[0].artifact.model_copy(update={"repository": other_repo})
    source = doc.sources[0].model_copy(update={"artifact": other_artifact})
    other_doc = doc.model_copy(
        update={
            "document_id": "other",
            "event_id": "other-event",
            "repository": other_repo,
            "sources": (source,),
        }
    )
    local = doc.model_copy(update={"sources": ()})
    index = CitationProvenanceIndex(documents=[local, other_doc], chunks=chunks)
    assert index.sources_for_chunk(chunks[0]) == ()
    assert index.artifact_sources("doc", other_artifact) == ()
    # Even the identical artifact in another document/event cannot supply URLs.
    other_doc = doc.model_copy(
        update={"document_id": "other", "event_id": "other-event"}
    )
    first = CitationProvenanceIndex(documents=[local, other_doc], chunks=chunks)
    second = CitationProvenanceIndex(
        documents=[other_doc, local], chunks=list(reversed(chunks))
    )
    assert (
        first.sources_for_chunk(chunks[0]) == second.sources_for_chunk(chunks[0]) == ()
    )
