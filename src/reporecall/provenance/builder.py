"""Build traceable citations from finalized evidence without executing retrieval."""

from pydantic import ValidationError

from reporecall.models.context_expansion import ExpandedContextResult
from reporecall.models.provenance import (
    CitationBundle,
    CitationIdentifier,
    CitationKind,
    ExpandedEvidenceCitation,
    RetrievedEvidenceCitation,
    SeedCitationReference,
)
from reporecall.provenance.index import CitationProvenanceError, CitationProvenanceIndex


class CitationBundleBuilder:
    """Assign bundle-local labels and join stored source provenance offline."""

    def __init__(self, *, provenance_index: CitationProvenanceIndex) -> None:
        self.provenance_index = provenance_index

    def build(self, context: ExpandedContextResult) -> CitationBundle:
        """Preserve context ordering, query, diagnostics, reasons, and truncation."""
        try:
            # Validate even model_copy/model_construct inputs at this boundary.
            ExpandedContextResult.model_validate(context.model_dump())
        except ValidationError as exc:
            raise CitationProvenanceError(
                "Invalid expanded context provenance."
            ) from exc
        retrieved = tuple(
            RetrievedEvidenceCitation(
                citation=CitationIdentifier(
                    kind=CitationKind.RETRIEVED, index=hit.rank
                ),
                hit=hit,
                sources=self.provenance_index.sources_for_chunk(hit.chunk),
            )
            for hit in context.seed_hits
        )
        labels = {item.hit.chunk.chunk_id: item.citation for item in retrieved}
        expanded = tuple(
            ExpandedEvidenceCitation(
                citation=CitationIdentifier(kind=CitationKind.EXPANDED, index=index),
                expanded_chunk=item,
                sources=self.provenance_index.sources_for_chunk(item.chunk),
                seed_references=tuple(
                    sorted(
                        (
                            SeedCitationReference(
                                seed_chunk_id=seed_id, citation=labels[seed_id]
                            )
                            for seed_id in {
                                reason.seed_chunk_id for reason in item.reasons
                            }
                        ),
                        key=lambda ref: ref.citation.index,
                    )
                ),
            )
            for index, item in enumerate(context.expanded_chunks, 1)
        )
        return CitationBundle(
            query=context.query,
            retrieved=retrieved,
            expanded=expanded,
            context_truncated=context.truncated,
        )
