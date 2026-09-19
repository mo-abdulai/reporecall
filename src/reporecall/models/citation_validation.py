"""Minimal reference syntax checks, independent of answers or generation."""

import re

from pydantic import BaseModel, ConfigDict

from reporecall.models.provenance import CitationBundle, CitationIdentifier

# Reject identifier subscripts (array[X1]), nested brackets and escaped markers.
_REFERENCE = re.compile(r"(?<![\w\[\\])\[([RX][1-9][0-9]*)\](?!\])")


class CitationValidationResult(BaseModel):
    """Unique references in occurrence order and their unresolved subset."""

    referenced_labels: tuple[str, ...]
    unknown_labels: tuple[str, ...]
    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def valid(self) -> bool:
        return not self.unknown_labels


def render_citation_reference(citation: CitationIdentifier) -> str:
    """Render only the canonical bracketed citation marker."""
    return f"[{citation.label}]"


def validate_citation_labels(
    text: str, bundle: CitationBundle
) -> CitationValidationResult:
    """Report unknown canonical markers without modifying the supplied text.

    This is a syntax/membership check, not an assessment of claim support or a
    Markdown parser. Bare identifiers, array subscripts, escaped markers and
    noncanonical indices such as [R01] are not citation references.
    """
    referenced = tuple(dict.fromkeys(_REFERENCE.findall(text)))
    known = {item.citation.label for item in bundle.retrieved} | {
        item.citation.label for item in bundle.expanded
    }
    return CitationValidationResult(
        referenced_labels=referenced,
        unknown_labels=tuple(label for label in referenced if label not in known),
    )
