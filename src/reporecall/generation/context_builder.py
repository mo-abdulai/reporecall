from collections.abc import Sequence

from reporecall.generation.exceptions import RAGContextError
from reporecall.models import RAGContext, RAGEvidence, VectorSearchHit


class RAGContextBuilder:
    """Render ranked retrieval hits into bounded source-faithful evidence."""

    def build(
        self,
        query: str,
        hits: Sequence[VectorSearchHit],
        *,
        max_context_chars: int,
    ) -> RAGContext:
        """Include complete hits in rank order, prefix-truncating only the first."""

        if not query.strip():
            raise RAGContextError("RAG context query must not be blank.")
        if max_context_chars <= 0:
            raise RAGContextError("Maximum context characters must be greater than zero.")

        ordered_hits = sorted(hits, key=lambda hit: hit.rank)
        ranks = [hit.rank for hit in ordered_hits]
        if len(set(ranks)) != len(ranks):
            raise RAGContextError("Retrieval hits must have unique ranks.")

        included: list[RAGEvidence] = []
        rendered: list[str] = []
        used_chars = 0
        truncated = False

        for hit in ordered_hits:
            evidence_id = f"E{len(included) + 1}"
            evidence = _evidence_from_hit(hit, evidence_id=evidence_id)
            block = _render_evidence(evidence)
            separator = "" if not rendered else "\n\n"
            required_chars = len(separator) + len(block)
            if used_chars + required_chars <= max_context_chars:
                rendered.append(f"{separator}{block}")
                included.append(evidence)
                used_chars += required_chars
                continue

            truncated = True
            if not included:
                available = max_context_chars - len(separator)
                rendered.append(block[:available])
                content_start = len(_evidence_header(evidence))
                included_content_chars = max(
                    0,
                    min(len(evidence.content), available - content_start),
                )
                included.append(
                    evidence.model_copy(
                        update={
                            "content": evidence.content[:included_content_chars],
                            "content_truncated": (
                                included_content_chars < len(evidence.content)
                            ),
                        }
                    )
                )
            break

        if len(included) < len(ordered_hits):
            truncated = True

        return RAGContext(
            query=query,
            evidence=tuple(included),
            text="".join(rendered),
            retrieved_count=len(ordered_hits),
            included_evidence_count=len(included),
            truncated=truncated,
        )


def _evidence_from_hit(hit: VectorSearchHit, *, evidence_id: str) -> RAGEvidence:
    chunk = hit.chunk
    return RAGEvidence(
        evidence_id=evidence_id,
        rank=hit.rank,
        score=hit.score,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        event_id=chunk.event_id,
        repository=chunk.repository,
        section_id=chunk.section_id,
        section_type=chunk.section_type,
        artifact=chunk.artifact,
        content=chunk.content,
    )


def _render_evidence(evidence: RAGEvidence) -> str:
    return (
        f"{_evidence_header(evidence)}"
        f"{evidence.content}"
        f"\n--- END REPOSITORY CONTENT {evidence.evidence_id} ---"
    )


def _evidence_header(evidence: RAGEvidence) -> str:
    artifact = (
        "None"
        if evidence.artifact is None
        else f"{evidence.artifact.artifact_type.value}:{evidence.artifact.identifier}"
    )
    return (
        f"=== EVIDENCE {evidence.evidence_id} ===\n"
        f"Rank: {evidence.rank}\n"
        f"Similarity: {format(evidence.score, '.17g')}\n"
        f"Repository: {evidence.repository.owner}/{evidence.repository.name}\n"
        f"Section: {evidence.section_type.value}\n"
        f"Artifact: {artifact}\n"
        f"Chunk ID: {evidence.chunk_id}\n"
        f"Document ID: {evidence.document_id}\n"
        f"Event ID: {evidence.event_id}\n"
        f"Section ID: {evidence.section_id}\n"
        f"--- BEGIN REPOSITORY CONTENT {evidence.evidence_id} ---\n"
    )
