"""Bounded one-hop context discovery over explicit relationships only."""

from pydantic import BaseModel, ConfigDict, Field

from reporecall.models.context_expansion import (
    ContextExpansionReason,
    ExpandedContextChunk,
    ExpandedContextResult,
    RelationshipTraversalDirection,
)
from reporecall.models.relationships import ArtifactReference, RelationshipType
from reporecall.models.reranking import RerankedRetrievalResult, RerankedSearchHit
from reporecall.models.retrieval_chunks import RetrievalChunk
from reporecall.retrieval.relationship_context_index import (
    RelationshipContextIndex,
    chunk_order,
    relationship_order,
)

STRONG_CONTEXT_RELATIONSHIPS = frozenset(
    {
        RelationshipType.ISSUE_HAS_COMMENT,
        RelationshipType.PULL_REQUEST_HAS_COMMENT,
        RelationshipType.PULL_REQUEST_HAS_REVIEW,
        RelationshipType.REVIEW_HAS_COMMENT,
        RelationshipType.PULL_REQUEST_HAS_REVIEW_COMMENT,
        RelationshipType.PULL_REQUEST_CHANGES_FILE,
        RelationshipType.LOCAL_COMMIT_CHANGES_FILE,
        RelationshipType.PULL_REQUEST_HAS_COMMIT,
        RelationshipType.GITHUB_COMMIT_MATCHES_LOCAL_COMMIT,
        RelationshipType.PULL_REQUEST_CLOSES_ISSUE,
        RelationshipType.COMMIT_CLOSES_ISSUE,
    }
)
CONTEXTUAL_RELATIONSHIPS = frozenset(
    {
        RelationshipType.PULL_REQUEST_REFERENCES_ISSUE,
        RelationshipType.COMMIT_REFERENCES_ISSUE,
    }
)


class RelationshipContextExpansionConfig(BaseModel):
    """Positive chunk budgets; seed_k defines scope, not truncation."""

    seed_k: int = Field(default=5, gt=0, strict=True)
    max_related_chunks_per_seed: int = Field(default=5, gt=0, strict=True)
    max_chunks_per_related_artifact: int = Field(default=2, gt=0, strict=True)
    max_total_expanded_chunks: int = Field(default=20, gt=0, strict=True)
    include_contextual_relationships: bool = False

    model_config = ConfigDict(frozen=True, extra="forbid")


class RelationshipContextExpander:
    """Expand seeds once, keeping original scores exclusively on seed hits.

    Strong edges precede contextual edges across all seeds. Within each class,
    selection uses seed rank, relationship enum value and canonical chunk order.
    Bounds count distinct non-seed chunks, never reasons. All discovered reasons
    for retained chunks survive budgets, even when a later seed's budget is full.
    Section IDs are the stable fallback because chunks carry no section ordinal.
    """

    def __init__(
        self,
        *,
        index: RelationshipContextIndex,
        config: RelationshipContextExpansionConfig | None = None,
    ) -> None:
        self.index = index
        self.config = config or RelationshipContextExpansionConfig()

    def expand(self, result: RerankedRetrievalResult) -> ExpandedContextResult:
        """Return offline structural context without searching or parsing text."""
        seeds = result.hits[: self.config.seed_k]
        seed_ids = {hit.chunk.chunk_id for hit in seeds}
        candidates: list[
            tuple[RerankedSearchHit, RetrievalChunk, ContextExpansionReason]
        ] = []
        for seed in seeds:
            self.index.validate_seed(seed.chunk)
            artifact = seed.chunk.artifact
            if artifact is None:
                continue
            for relationship, event_contextual in self.index.related_edges(
                seed.chunk.event_id, artifact
            ):
                kind = relationship.relationship_type
                contextual = event_contextual or kind in CONTEXTUAL_RELATIONSHIPS
                if kind not in STRONG_CONTEXT_RELATIONSHIPS | CONTEXTUAL_RELATIONSHIPS:
                    continue
                if contextual and not self.config.include_contextual_relationships:
                    continue
                direction = (
                    RelationshipTraversalDirection.OUTGOING
                    if relationship.source == artifact
                    else RelationshipTraversalDirection.INCOMING
                )
                reason = ContextExpansionReason(
                    seed_chunk_id=seed.chunk.chunk_id,
                    relationship=relationship,
                    traversal_direction=direction,
                    contextual=contextual,
                )
                if reason.related_artifact.repository != seed.chunk.repository:
                    continue
                for chunk in self.index.artifact_chunks(
                    seed.chunk.event_id, reason.related_artifact
                ):
                    if chunk.chunk_id not in seed_ids:
                        candidates.append((seed, chunk, reason))
        candidates.sort(
            key=lambda item: (
                item[2].contextual,
                item[0].rank,
                item[2].relationship_type.value,
                chunk_order(item[1]),
                relationship_order(item[2].relationship),
                item[2].traversal_direction.value,
            )
        )
        selected: dict[str, RetrievalChunk] = {}
        reasons: dict[str, list[ContextExpansionReason]] = {}
        per_seed: dict[str, set[str]] = {}
        per_artifact: dict[tuple[str, ArtifactReference], set[str]] = {}
        truncated = False
        for seed, chunk, reason in candidates:
            chunk_id = chunk.chunk_id
            chunk_reasons = reasons.setdefault(chunk_id, [])
            if reason not in chunk_reasons:
                chunk_reasons.append(reason)
            seed_chunks = per_seed.setdefault(seed.chunk.chunk_id, set())
            artifact_key = (seed.chunk.chunk_id, reason.related_artifact)
            artifact_chunks = per_artifact.setdefault(artifact_key, set())
            if (
                chunk_id not in artifact_chunks
                and len(artifact_chunks) >= self.config.max_chunks_per_related_artifact
            ):
                truncated = True
                continue
            if (
                chunk_id not in seed_chunks
                and len(seed_chunks) >= self.config.max_related_chunks_per_seed
            ):
                truncated = True
                continue
            artifact_chunks.add(chunk_id)
            seed_chunks.add(chunk_id)
            if (
                chunk_id not in selected
                and len(selected) >= self.config.max_total_expanded_chunks
            ):
                truncated = True
                continue
            selected.setdefault(chunk_id, chunk)
        return ExpandedContextResult(
            query=result.query,
            seed_hits=seeds,
            expanded_chunks=tuple(
                ExpandedContextChunk(chunk=chunk, reasons=tuple(reasons[chunk_id]))
                for chunk_id, chunk in selected.items()
            ),
            truncated=truncated,
        )
