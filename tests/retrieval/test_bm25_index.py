import hashlib
from collections.abc import Sequence

import numpy as np
import pytest
from pydantic import ValidationError

import reporecall.retrieval.bm25_index as bm25_index_module
from reporecall.github import GitHubRepository
from reporecall.models import EventMetadata, RetrievalChunk, RetrievalSectionType
from reporecall.retrieval import BM25Config, BM25Index, BM25IndexError


def test_bm25_dependency_is_loaded_only_when_index_is_instantiated():
    assert "bm25s" not in vars(bm25_index_module)


def test_missing_bm25_dependency_has_actionable_error(monkeypatch):
    def missing_module(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(bm25_index_module, "import_module", missing_module)

    with pytest.raises(BM25IndexError, match="bm25s"):
        BM25Index()


def test_bm25_configuration_has_standard_immutable_defaults():
    config = BM25Config()

    assert config.k1 == 1.5
    assert config.b == 0.75
    with pytest.raises(ValidationError, match="frozen"):
        config.k1 = 2.0


@pytest.mark.parametrize(
    "values",
    [
        {"k1": 0},
        {"k1": -1},
        {"b": -0.1},
        {"b": 1.1},
    ],
)
def test_bm25_configuration_rejects_invalid_values(values: dict[str, float]):
    with pytest.raises(ValidationError):
        BM25Config(**values)


def test_index_preserves_input_row_order_tokens_and_source_hashes():
    chunks = [
        _chunk("chunk-c", "ConnectionResetError"),
        _chunk("chunk-a", "src/database/session.py"),
    ]
    index = BM25Index()

    index.build(chunks)
    matches = index.search(("connectionreseterror",), k=2)

    assert index.chunk_ids == ("chunk-c", "chunk-a")
    assert "src/database/session.py" in index.tokenized_corpus[1]
    assert matches[0].source_text_sha256 == hashlib.sha256(
        chunks[0].text.encode("utf-8")
    ).hexdigest()


def test_exact_technical_term_ranks_matching_chunk_first():
    index = BM25Index()
    index.build(
        [
            _chunk("a", "ConnectionResetError occurred in retry_worker()"),
            _chunk("b", "Database timeout after transaction commit"),
            _chunk("c", "Update CSS spacing"),
        ]
    )

    matches = index.search(("connectionreseterror",), k=3)

    assert [match.chunk_id for match in matches] == ["a"]
    assert matches[0].rank == 1
    assert matches[0].score > 0


def test_search_excludes_zero_score_documents():
    index = BM25Index()
    index.build([_chunk("a", "database"), _chunk("b", "frontend")])

    assert index.search(("unseen-token",), k=2) == []


def test_candidate_filtering_happens_before_top_k_and_reassigns_ranks():
    index = BM25Index()
    index.build(
        [
            _chunk("a", "needle needle needle needle needle"),
            _chunk("b", "needle needle needle needle"),
            _chunk("c", "needle needle"),
            _chunk("d", "needle"),
        ]
    )

    matches = index.search(
        ("needle",),
        k=2,
        candidate_chunk_ids=("d", "c"),
    )

    assert [match.chunk_id for match in matches] == ["c", "d"]
    assert [match.rank for match in matches] == [1, 2]


def test_candidate_input_order_does_not_change_tie_ordering():
    index = BM25Index()
    index.build(
        [
            _chunk("chunk-c", "shared"),
            _chunk("chunk-a", "shared"),
            _chunk("chunk-b", "shared"),
        ]
    )

    matches = index.search(
        ("shared",),
        k=3,
        candidate_chunk_ids=("chunk-c", "chunk-b", "chunk-a"),
    )

    assert [match.chunk_id for match in matches] == [
        "chunk-a",
        "chunk-b",
        "chunk-c",
    ]


def test_empty_build_resets_index_and_empty_search_is_safe():
    index = BM25Index()
    index.build([_chunk("a", "database")])

    index.build([])

    assert index.is_empty is True
    assert index.chunk_ids == ()
    assert index.tokenized_corpus == ()
    assert index.search((), k=5) == []


def test_index_rejects_duplicate_chunk_ids():
    index = BM25Index()

    with pytest.raises(BM25IndexError, match="Duplicate chunk IDs"):
        index.build([_chunk("duplicate", "first"), _chunk("duplicate", "second")])


def test_index_rejects_chunks_that_tokenize_to_no_terms():
    index = BM25Index()
    chunk = _chunk("punctuation", "source content").model_copy(
        update={"text": "... !!!"}
    )

    with pytest.raises(BM25IndexError, match="produced no BM25 tokens"):
        index.build([chunk])


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_nonpositive_k(k: int):
    index = BM25Index()
    index.build([_chunk("a", "database")])

    with pytest.raises(BM25IndexError, match="greater than zero"):
        index.search(("database",), k=k)


def test_search_with_no_query_tokens_returns_no_matches():
    index = BM25Index()
    index.build([_chunk("a", "database")])

    assert index.search((), k=1) == []


def test_k_larger_than_matching_corpus_returns_only_positive_matches():
    index = BM25Index()
    index.build([_chunk("a", "database"), _chunk("b", "database pool")])

    matches = index.search(("database",), k=10)

    assert len(matches) == 2


def test_candidate_ids_must_be_nonblank_unique_and_indexed():
    index = BM25Index()
    index.build([_chunk("a", "database")])

    with pytest.raises(BM25IndexError, match="must not be blank"):
        index.search(("database",), k=1, candidate_chunk_ids=(" ",))
    with pytest.raises(BM25IndexError, match="must be unique"):
        index.search(("database",), k=1, candidate_chunk_ids=("a", "a"))
    with pytest.raises(BM25IndexError, match="missing from the BM25 index"):
        index.search(("database",), k=1, candidate_chunk_ids=("missing",))


def test_empty_candidate_selection_returns_without_scoring():
    index = BM25Index()
    index.build([_chunk("a", "database")])
    index._model = _FailingModel()

    assert index.search(("database",), k=1, candidate_chunk_ids=()) == []


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        (np.array([1.0, 2.0, 3.0]), "invalid score array shape"),
        (np.array([float("nan"), 1.0]), "non-finite lexical score"),
        (np.array([float("inf"), 1.0]), "non-finite lexical score"),
    ],
)
def test_index_rejects_invalid_library_scores(
    scores: np.ndarray,
    message: str,
):
    index = BM25Index()
    index.build([_chunk("a", "database"), _chunk("b", "worker")])
    index._model = _ScoreModel(scores)

    with pytest.raises(BM25IndexError, match=message):
        index.search(("database",), k=2)


def test_library_scoring_failure_is_wrapped():
    index = BM25Index()
    index.build([_chunk("a", "database")])
    index._model = _FailingModel()

    with pytest.raises(BM25IndexError, match="failed while scoring"):
        index.search(("database",), k=1)


class _ScoreModel:
    def __init__(self, scores: np.ndarray) -> None:
        self.scores = scores

    def get_scores(self, query_tokens_single: list[str]) -> object:
        return self.scores


class _FailingModel:
    def get_scores(self, query_tokens_single: list[str]) -> object:
        raise RuntimeError("scoring failed")


def _chunk(
    chunk_id: str,
    content: str,
    *,
    languages: Sequence[str] = (),
) -> RetrievalChunk:
    repository = GitHubRepository(owner="owner", name="repo")
    event_id = f"event-{chunk_id}"
    metadata = EventMetadata(
        event_id=event_id,
        repository=repository,
        languages=tuple(languages),
    )
    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        event_id=event_id,
        repository=repository,
        section_id=f"section-{chunk_id}",
        section_type=RetrievalSectionType.ISSUE,
        chunk_index=0,
        content=content,
        text=f"Repository: owner/repo\n{content}",
        metadata=metadata,
    )
