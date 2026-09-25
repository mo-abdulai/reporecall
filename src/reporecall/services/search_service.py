"""Coordinate existing evidence retrieval components without generating answers."""

from threading import Lock

from pydantic import BaseModel, ConfigDict

from reporecall.models import (
    CitationBundle,
    ExpandedContextResult,
    MetadataFilter,
    RerankedRetrievalResult,
)
from reporecall.provenance import CitationBundleBuilder
from reporecall.retrieval import (
    CrossEncoderReranker,
    HybridRetriever,
    ReciprocalRankFusion,
    RelationshipContextExpander,
)


class SearchServiceResult(BaseModel):
    """Final ranked seeds and structural context, with optional canonical citations."""

    context: ExpandedContextResult
    citations: CitationBundle | None
    model_config = ConfigDict(frozen=True)


class SearchService:
    """Reuse shared components and serialize local model access without request state."""

    def __init__(
        self,
        *,
        hybrid: HybridRetriever,
        fusion: ReciprocalRankFusion,
        reranker: CrossEncoderReranker,
        expander: RelationshipContextExpander,
        citations: CitationBundleBuilder,
    ) -> None:
        self.hybrid = hybrid
        self.fusion = fusion
        self.reranker = reranker
        self.expander = expander
        self.citations = citations
        self._lock = Lock()

    def search(
        self,
        query: str,
        *,
        metadata_filter: MetadataFilter | None = None,
        top_k: int = 5,
        expand_relationships: bool = True,
        include_citations: bool = True,
    ) -> SearchServiceResult:
        """Preserve raw query/filter; expansion is never mixed into ranked hits."""
        if not query.strip() or not 1 <= top_k <= self.reranker.config.top_k:
            raise ValueError("Invalid search query or result count.")
        with self._lock:
            candidates = self.hybrid.search(query, metadata_filter=metadata_filter)
            for candidate in candidates.candidates:
                self.expander.index.validate_seed(candidate.chunk)
            fused = self.fusion.fuse(candidates)
            ranked = self.reranker.rerank(fused)
            hits = ranked.hits[:top_k]
            limited = RerankedRetrievalResult.model_validate(
                {
                    **ranked.model_dump(),
                    "hits": hits,
                    "top_k": top_k,
                    "returned_count": len(hits),
                }
            )
            context = (
                self.expander.expand(limited)
                if expand_relationships
                else ExpandedContextResult(
                    query=query, seed_hits=hits, expanded_chunks=(), truncated=False
                )
            )
            return SearchServiceResult(
                context=context,
                citations=self.citations.build(context) if include_citations else None,
            )
