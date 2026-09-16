import pytest
from pydantic import ValidationError

from reporecall.models import CitationIdentifier, CitationKind


@pytest.mark.parametrize(
    "kind,prefix", [(CitationKind.RETRIEVED, "R"), (CitationKind.EXPANDED, "X")]
)
def test_labels_are_derived_and_identifiers_frozen(kind, prefix):
    identifier = CitationIdentifier(kind=kind, index=3)
    assert identifier.label == f"{prefix}3"
    with pytest.raises(ValidationError):
        identifier.index = 4
    with pytest.raises(ValidationError):
        CitationIdentifier(kind=kind, index=3, label="R4")


@pytest.mark.parametrize("index", [0, -1, True, 1.5, "1"])
def test_invalid_index(index):
    with pytest.raises(ValidationError):
        CitationIdentifier(kind="retrieved", index=index)


def test_invalid_kind():
    with pytest.raises(ValidationError):
        CitationIdentifier(kind="evidence", index=1)
