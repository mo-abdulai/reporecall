"""Versioned synchronous HTTP handlers for blocking application operations."""

from typing import Annotated

from fastapi import APIRouter, Depends

from reporecall.api.dependencies import get_services
from reporecall.api.schemas import (
    CitationBundleResponse,
    QueryRequest,
    RepositoryResponse,
    SearchRequest,
    SearchResponse,
    StatusResponse,
)
from reporecall.models import (
    EngineeringEvent,
    RetrievalChunk,
    RetrievalDocument,
    UnderstoodQuery,
)
from reporecall.persistence.catalog import CorpusStatistics
from reporecall.services.runtime import ServiceResources

router = APIRouter(prefix="/api/v1")
Services = Annotated[ServiceResources, Depends(get_services)]


@router.get("/health", tags=["Status"])
def health() -> StatusResponse:
    """Process liveness; does not contact the database or load models."""
    return StatusResponse(status="ok")


@router.get("/ready", tags=["Status"])
def ready(services: Services) -> StatusResponse:
    """Check shared services and database schema without inference."""
    services.corpus.check_ready()
    return StatusResponse(status="ready")


@router.get("/corpus", tags=["Corpus"])
def corpus(services: Services) -> CorpusStatistics:
    return services.corpus.statistics()


@router.get("/repositories", tags=["Corpus"])
def repositories(services: Services) -> RepositoryResponse:
    return RepositoryResponse(repositories=services.corpus.repositories())


@router.get("/events/{event_id}", tags=["Corpus"])
def event(event_id: str, services: Services) -> EngineeringEvent:
    return services.corpus.event(event_id)


@router.get("/documents/{document_id}", tags=["Corpus"])
def document(document_id: str, services: Services) -> RetrievalDocument:
    return services.corpus.document(document_id)


@router.get("/chunks/{chunk_id}", tags=["Corpus"])
def chunk(chunk_id: str, services: Services) -> RetrievalChunk:
    return services.corpus.chunk(chunk_id)


@router.post("/query/understand", tags=["Query"])
def understand(body: QueryRequest, services: Services) -> UnderstoodQuery:
    """Interpret constraints only; does not execute search or answer generation."""
    return services.query.understand(body.query)


@router.post("/search", tags=["Search"])
def search(body: SearchRequest, services: Services) -> SearchResponse:
    """Rank hybrid evidence, optionally expand relationships and resolve citations.

    Scores are raw diagnostics, not probabilities. Expanded context is unranked.
    Query understanding is never invoked implicitly.
    """
    result = services.search.search(
        query=body.query,
        metadata_filter=body.metadata_filter,
        top_k=body.top_k,
        expand_relationships=body.expand_relationships,
        include_citations=body.include_citations,
    )
    return SearchResponse(
        query=result.context.query,
        hits=result.context.seed_hits,
        expanded_context=result.context.expanded_chunks,
        context_truncated=result.context.truncated,
        citations=CitationBundleResponse.from_domain(result.citations)
        if result.citations is not None
        else None,
    )
