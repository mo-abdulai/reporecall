"""Real offline orchestration with supplied synthetic dense/neural outputs."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from reporecall.api.schemas import CitationBundleResponse
from reporecall.models import MetadataFilter, VectorSearchHit
from reporecall.provenance import CitationBundleBuilder, CitationProvenanceIndex
from reporecall.retrieval import (
    BM25Index,
    CrossEncoderReranker,
    CrossEncoderRerankerConfig,
    HybridRetrievalConfig,
    HybridRetriever,
    KeywordRetriever,
    ReciprocalRankFusion,
    RelationshipContextExpander,
    RelationshipContextExpansionConfig,
    RelationshipContextIndex,
)
from reporecall.services.search_service import SearchService
from tests.persistence.fixtures import corpus_data


@pytest.fixture
def pipeline():
    corpus = corpus_data()
    calls = []

    class Dense:
        def search(self, query, *, k=5, metadata_filter=None):
            calls.append((query, metadata_filter))
            return [
                VectorSearchHit(chunk=chunk, score=0.9 - i / 100, rank=i + 1)
                for i, chunk in enumerate(corpus.chunks[:k])
            ]

    class Scores:
        model_name = "synthetic"

        def score(self, query, passages):
            return [0.8 - i / 100 for i, _ in enumerate(passages)]

    bm25 = BM25Index()
    bm25.build(corpus.chunks)
    service = SearchService(
        hybrid=HybridRetriever(
            vector_retriever=Dense(),
            keyword_retriever=KeywordRetriever(
                index=bm25, chunks={c.chunk_id: c for c in corpus.chunks}
            ),
            config=HybridRetrievalConfig(dense_k=8, keyword_k=8),
        ),
        fusion=ReciprocalRankFusion(),
        reranker=CrossEncoderReranker(
            backend=Scores(),
            config=CrossEncoderRerankerConfig(model_name="synthetic", top_k=8),
        ),
        expander=RelationshipContextExpander(
            index=RelationshipContextIndex(events=corpus.events, chunks=corpus.chunks),
            config=RelationshipContextExpansionConfig(
                seed_k=8, max_total_expanded_chunks=1
            ),
        ),
        citations=CitationBundleBuilder(
            provenance_index=CitationProvenanceIndex(
                documents=corpus.documents, chunks=corpus.chunks
            )
        ),
    )
    return SimpleNamespace(service=service, corpus=corpus, calls=calls)


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("citations", [False, True])
def test_pipeline_optional_stages(pipeline, expand, citations):
    query = " session\ncleanup  "
    before = pipeline.corpus
    result = pipeline.service.search(
        query, top_k=1, expand_relationships=expand, include_citations=citations
    )
    assert result.context.query == query
    assert pipeline.calls == [(query, None)]
    assert len(result.context.seed_hits) == 1
    hit = result.context.seed_hits[0]
    assert hit.reranker_score == 0.8
    assert hit.dense_score is not None
    assert hit.rrf_score > 0
    assert hit.chunk in before.chunks
    assert bool(result.context.expanded_chunks) is expand
    assert (result.citations is not None) is citations
    if citations:
        rendered = CitationBundleResponse.from_domain(result.citations)
        assert rendered.retrieved[0].label == "R1"
        assert rendered.retrieved[0].evidence.hit == hit
        if expand:
            assert rendered.expanded[0].label == "X1"
            assert rendered.expanded[0].seed_references[0].label == "R1"
    assert pipeline.corpus == before


def test_concurrent_requests_do_not_mutate_shared_config(pipeline):
    def run(k):
        return pipeline.service.search("session", top_k=k, expand_relationships=False)

    with ThreadPoolExecutor(max_workers=2) as executor:
        one, three = executor.map(run, (1, 3))
    assert len(one.context.seed_hits) == 1
    assert len(three.context.seed_hits) == 3
    assert pipeline.service.reranker.config.top_k == 8
    assert run(1) == one


def test_metadata_filter_object_passed_without_interpretation(pipeline):
    metadata_filter = MetadataFilter(languages=("Python",))
    pipeline.service.search("session", metadata_filter=metadata_filter)
    assert pipeline.calls[0][1] is metadata_filter


@pytest.mark.parametrize(("query", "k"), [("", 1), ("x", 0), ("x", 9)])
def test_invalid_service_input(pipeline, query, k):
    with pytest.raises(ValueError):
        pipeline.service.search(query, top_k=k)


def test_truncation_preserved(pipeline, monkeypatch):
    context = pipeline.service.search("session", top_k=3).context.model_copy(
        update={"truncated": True}
    )
    monkeypatch.setattr(pipeline.service.expander, "expand", lambda result: context)
    result = pipeline.service.search("session", top_k=3)
    assert result.context.truncated
    assert result.citations.context_truncated
