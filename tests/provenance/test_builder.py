import pytest
from pydantic import ValidationError

from reporecall.models import (
    CitationBundle,
    ExpandedContextResult,
    ExpandedEvidenceCitation,
    RetrievedEvidenceCitation,
)
from reporecall.provenance import (
    CitationBundleBuilder,
    CitationProvenanceError,
    CitationProvenanceIndex,
)


def test_order_diagnostics_relationships_query_and_truncation(evidence, bundle):
    _, _, context = evidence
    assert [c.citation.label for c in bundle.retrieved] == ["R1", "R2", "R3"]
    assert [c.hit.chunk.chunk_id for c in bundle.retrieved] == ["C", "A", "B"]
    assert [c.citation.label for c in bundle.expanded] == ["X1", "X2", "X3"]
    assert [c.expanded_chunk.chunk.chunk_id for c in bundle.expanded] == ["D", "F", "E"]
    assert tuple(c.hit for c in bundle.retrieved) == context.seed_hits
    assert tuple(c.expanded_chunk for c in bundle.expanded) == context.expanded_chunks
    assert bundle.query == context.query
    assert bundle.context_truncated
    assert bundle.retrieved_count == bundle.expanded_count == 3
    for item in bundle.expanded:
        assert [(r.seed_chunk_id, r.citation.label) for r in item.seed_references] == [
            ("C", "R1"),
            ("B", "R3"),
        ]
    assert CitationBundle.model_validate_json(bundle.model_dump_json()) == bundle


def test_exact_sources_and_urls(evidence, bundle):
    document, _, _ = evidence
    for i, item in enumerate((*bundle.retrieved, *bundle.expanded)):
        assert len(item.sources) == 1
        assert item.sources[0].source == document.sources[i]
        assert item.urls == (document.sources[i].url,)


def test_missing_urls_and_sources_are_valid(evidence):
    doc, chunks, context = evidence
    source = doc.sources[0].model_copy(update={"url": None})
    doc = doc.model_copy(update={"sources": (source,)})
    bundle = CitationBundleBuilder(
        provenance_index=CitationProvenanceIndex(documents=[doc], chunks=chunks)
    ).build(context)
    assert len(bundle.retrieved[0].sources) == 1
    assert all(item.urls == () for item in (*bundle.retrieved, *bundle.expanded))
    assert bundle.retrieved[1].sources == ()


def test_sources_urls_dedup_order_and_no_mutation(evidence):
    doc, chunks, context = evidence
    base = doc.sources[0]
    alternative = base.model_copy(update={"url": "https://example.test/alternative"})
    same_url_other_record = base.model_copy(update={"label": "Different stored label"})
    doc = doc.model_copy(
        update={"sources": (*doc.sources, base, alternative, same_url_other_record)}
    )
    before = (
        doc.model_dump(),
        tuple(c.model_dump() for c in chunks),
        context.model_dump(),
    )
    build = lambda d, c: CitationBundleBuilder(
        provenance_index=CitationProvenanceIndex(documents=d, chunks=c)
    ).build(context)
    first = build([doc], chunks)
    second = build(
        [doc.model_copy(update={"sources": tuple(reversed(doc.sources))})],
        tuple(reversed(chunks)),
    )
    assert first == second
    assert len(first.retrieved[0].sources) == 3
    assert first.retrieved[0].urls == tuple(sorted((base.url, alternative.url)))
    assert before == (
        doc.model_dump(),
        tuple(c.model_dump() for c in chunks),
        context.model_dump(),
    )


def test_artifactless_keeps_identity_without_unrelated_sources(evidence):
    doc, chunks, context = evidence
    section = doc.sections[0].model_copy(update={"artifact": None})
    doc = doc.model_copy(update={"sections": (section, *doc.sections[1:])})
    c = chunks[0].model_copy(update={"artifact": None})
    hit = context.seed_hits[0].model_copy(update={"chunk": c})
    context = ExpandedContextResult(query="query", seed_hits=(hit,), expanded_chunks=())
    result = CitationBundleBuilder(
        provenance_index=CitationProvenanceIndex(documents=[doc], chunks=[c])
    ).build(context)
    assert result.retrieved[0].hit.chunk == c
    assert result.retrieved[0].sources == ()
    assert result.retrieved[0].urls == ()


def test_empty_bundle():
    context = ExpandedContextResult(query="empty", seed_hits=(), expanded_chunks=())
    bundle = CitationBundleBuilder(
        provenance_index=CitationProvenanceIndex(documents=[], chunks=[])
    ).build(context)
    assert bundle.retrieved_count == bundle.expanded_count == 0
    assert not bundle.context_truncated


def test_missing_seed_reason_fails_clearly(evidence):
    doc, chunks, context = evidence
    reason = (
        context.expanded_chunks[0]
        .reasons[0]
        .model_copy(update={"seed_chunk_id": "missing"})
    )
    expanded = context.expanded_chunks[0].model_copy(update={"reasons": (reason,)})
    invalid = context.model_copy(update={"expanded_chunks": (expanded,)})
    with pytest.raises(CitationProvenanceError, match="Invalid expanded context"):
        CitationBundleBuilder(
            provenance_index=CitationProvenanceIndex(documents=[doc], chunks=chunks)
        ).build(invalid)


@pytest.mark.parametrize(
    "field", ["rank", "reranker_score", "rrf_score", "dense_score", "keyword_score"]
)
def test_no_expanded_scores_or_rank(bundle, field):
    assert field not in ExpandedEvidenceCitation.model_fields
    with pytest.raises(ValidationError):
        ExpandedEvidenceCitation.model_validate(
            bundle.expanded[0].model_dump() | {field: 1}
        )


@pytest.mark.parametrize("retrieved", [True, False])
def test_wrong_kind(bundle, retrieved):
    item = bundle.retrieved[0] if retrieved else bundle.expanded[0]
    cls = RetrievedEvidenceCitation if retrieved else ExpandedEvidenceCitation
    kind = "expanded" if retrieved else "retrieved"
    with pytest.raises(ValidationError):
        cls.model_validate(item.model_dump() | {"citation": {"kind": kind, "index": 1}})


@pytest.mark.parametrize(
    "field,value",
    [
        ("document_id", "bad"),
        ("event_id", "bad"),
        ("chunk_id", "bad"),
        ("section_id", "bad"),
        ("section_type", "patch"),
    ],
)
def test_source_identity_mismatch(bundle, field, value):
    item = bundle.retrieved[0]
    source = item.sources[0].model_dump() | {field: value}
    with pytest.raises(ValidationError, match="complete chunk provenance"):
        RetrievedEvidenceCitation.model_validate(
            item.model_dump() | {"sources": [source]}
        )


def test_invalid_bundle_links_order_and_duplicates(bundle):
    data = bundle.model_dump()
    with pytest.raises(ValidationError):
        CitationBundle.model_validate(
            data | {"retrieved": list(reversed(data["retrieved"]))}
        )
    with pytest.raises(ValidationError):
        CitationBundle.model_validate(data | {"expanded": [data["expanded"][0]] * 2})
    data["expanded"][0]["seed_references"][0]["citation"]["index"] = 2
    with pytest.raises(ValidationError, match="original chunk"):
        CitationBundle.model_validate(data)


def test_missing_or_duplicate_seed_links_and_no_reasons(bundle):
    item = bundle.expanded[0].model_dump()
    for refs in ([], item["seed_references"][:1], item["seed_references"] * 2):
        with pytest.raises(ValidationError):
            ExpandedEvidenceCitation.model_validate(item | {"seed_references": refs})
    item["expanded_chunk"]["reasons"] = []
    with pytest.raises(ValidationError):
        ExpandedEvidenceCitation.model_validate(item)


def test_immutable_models(bundle):
    for item, field, value in [
        (bundle, "query", "changed"),
        (bundle.retrieved[0], "sources", ()),
        (bundle.expanded[0], "sources", ()),
    ]:
        with pytest.raises(ValidationError):
            setattr(item, field, value)
