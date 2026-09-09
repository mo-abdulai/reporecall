import pytest
from pydantic import ValidationError

from reporecall.github import GitHubRepository
from reporecall.models import (
    EventMetadata,
    KeywordIndexMatch,
    KeywordSearchHit,
    RetrievalChunk,
    RetrievalSectionType,
)


def test_keyword_index_match_is_immutable_and_serializable():
    match = KeywordIndexMatch(
        chunk_id="chunk-a",
        source_text_sha256="a" * 64,
        score=2.5,
        rank=1,
    )

    assert match.model_dump(mode="json") == {
        "chunk_id": "chunk-a",
        "source_text_sha256": "a" * 64,
        "score": 2.5,
        "rank": 1,
    }
    with pytest.raises(ValidationError, match="frozen"):
        match.rank = 2


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_keyword_index_match_rejects_nonfinite_scores(score: float):
    with pytest.raises(ValidationError):
        KeywordIndexMatch(
            chunk_id="chunk-a",
            source_text_sha256="a" * 64,
            score=score,
            rank=1,
        )


@pytest.mark.parametrize("rank", [0, -1])
def test_keyword_index_match_rejects_nonpositive_ranks(rank: int):
    with pytest.raises(ValidationError):
        KeywordIndexMatch(
            chunk_id="chunk-a",
            source_text_sha256="a" * 64,
            score=1.0,
            rank=rank,
        )


def test_keyword_index_match_rejects_invalid_source_hash():
    with pytest.raises(ValidationError):
        KeywordIndexMatch(
            chunk_id="chunk-a",
            source_text_sha256="not-a-sha256",
            score=1.0,
            rank=1,
        )


def test_keyword_search_hit_preserves_full_chunk_and_is_immutable():
    chunk = _chunk()
    hit = KeywordSearchHit(chunk=chunk, score=1.75, rank=1)

    assert hit.chunk is chunk
    assert hit.chunk.model_dump() == chunk.model_dump()
    assert hit.score == 1.75
    assert hit.rank == 1
    with pytest.raises(ValidationError, match="frozen"):
        hit.score = 2.0


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_keyword_search_hit_rejects_nonfinite_scores(score: float):
    with pytest.raises(ValidationError):
        KeywordSearchHit(chunk=_chunk(), score=score, rank=1)


@pytest.mark.parametrize("rank", [0, -1])
def test_keyword_search_hit_rejects_nonpositive_ranks(rank: int):
    with pytest.raises(ValidationError):
        KeywordSearchHit(chunk=_chunk(), score=1.0, rank=rank)


def _chunk() -> RetrievalChunk:
    repository = GitHubRepository(owner="owner", name="repo")
    return RetrievalChunk(
        chunk_id="chunk-a",
        document_id="document-a",
        event_id="event-a",
        repository=repository,
        section_id="issue-10",
        section_type=RetrievalSectionType.ISSUE,
        chunk_index=0,
        content="ConnectionResetError in retry_worker()",
        text=(
            "Repository: owner/repo\n"
            "Event: event-a\n"
            "ConnectionResetError in retry_worker()"
        ),
        metadata=EventMetadata(event_id="event-a", repository=repository),
    )
