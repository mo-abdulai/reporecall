import pytest

from reporecall.provenance import render_citation_reference, validate_citation_labels


@pytest.mark.parametrize(
    "text,referenced,unknown",
    [
        ("Previously fixed [R1]. Related context [X1].", ("R1", "X1"), ()),
        ("See [R9].", ("R9",), ("R9",)),
        ("See [R1], [X1], and [X99].", ("R1", "X1", "X99"), ("X99",)),
        ("[R1] ... [R1] [X1] [R9] [R9]", ("R1", "X1", "R9"), ("R9",)),
        (r"variable R1 array[X1] [R0] [R01] [x1] [[R1]] \[X1]", (), ()),
        ("[R1][X1]", ("R1", "X1"), ()),
        ("No references", (), ()),
    ],
)
def test_reference_validation(bundle, text, referenced, unknown):
    original = text
    result = validate_citation_labels(text, bundle)
    assert result.referenced_labels == referenced
    assert result.unknown_labels == unknown
    assert result.valid == (not unknown)
    assert text == original


def test_render(bundle):
    assert render_citation_reference(bundle.retrieved[0].citation) == "[R1]"
    assert render_citation_reference(bundle.expanded[0].citation) == "[X1]"
