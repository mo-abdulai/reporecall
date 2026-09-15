import pytest
from pydantic import ValidationError

from reporecall.models import MetadataFilter, UnderstoodQuery


def test_understood_query_preserves_original_query_and_is_immutable():
    metadata_filter = MetadataFilter(languages=("Python",))
    understood = UnderstoodQuery(
        original_query="  Find Python ConnectionResetError fixes in src/API  ",
        retrieval_query="ConnectionResetError fixes",
        metadata_filter=metadata_filter,
        extracted_constraints=("language:python",),
        model_name="fake-model",
    )

    assert understood.original_query == (
        "  Find Python ConnectionResetError fixes in src/API  "
    )
    assert understood.retrieval_query == "ConnectionResetError fixes"
    assert understood.metadata_filter == metadata_filter
    assert understood.extracted_constraints == ("language:python",)

    with pytest.raises(ValidationError, match="frozen"):
        understood.retrieval_query = "rewritten"


@pytest.mark.parametrize("field", ["original_query", "retrieval_query"])
def test_understood_query_rejects_blank_query_text(field: str):
    values = {
        "original_query": "authentication regression",
        "retrieval_query": "authentication regression",
    }
    values[field] = "   "

    with pytest.raises(ValidationError, match="must not be blank"):
        UnderstoodQuery(**values)


def test_understood_query_rejects_blank_diagnostics():
    with pytest.raises(ValidationError, match="diagnostics"):
        UnderstoodQuery(
            original_query="authentication regression",
            retrieval_query="authentication regression",
            warnings=(" ",),
        )
