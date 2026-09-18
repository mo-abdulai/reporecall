"""Deterministic citation membership and score-free judge evidence rendering."""

from reporecall.models.provenance import CitationBundle
from reporecall.models.rag_evaluation import JudgeEvidenceItem, RAGCitationDiagnostics
from reporecall.provenance.validation import validate_citation_labels


def citation_diagnostics(answer: str, bundle: CitationBundle) -> RAGCitationDiagnostics:
    """Reuse canonical Phase 18 parsing; repeated labels count only once."""
    result = validate_citation_labels(answer, bundle)
    unknown = set(result.unknown_labels)
    return RAGCitationDiagnostics(
        cited_labels=result.referenced_labels,
        valid_cited_labels=tuple(
            label for label in result.referenced_labels if label not in unknown
        ),
        unknown_cited_labels=result.unknown_labels,
    )


def render_judge_evidence(bundle: CitationBundle) -> tuple[JudgeEvidenceItem, ...]:
    """Preserve exact chunk text and relationship provenance, excluding scores."""
    retrieved = tuple(
        JudgeEvidenceItem(
            citation=c.citation,
            content=c.hit.chunk.text,
            artifact=c.hit.chunk.artifact,
            section_type=c.hit.chunk.section_type,
        )
        for c in bundle.retrieved
    )
    expanded = tuple(
        JudgeEvidenceItem(
            citation=c.citation,
            content=c.expanded_chunk.chunk.text,
            artifact=c.expanded_chunk.chunk.artifact,
            section_type=c.expanded_chunk.chunk.section_type,
            expanded_from=tuple(ref.citation.label for ref in c.seed_references),
            expansion_reasons=c.expanded_chunk.reasons,
        )
        for c in bundle.expanded
    )
    return retrieved + expanded
