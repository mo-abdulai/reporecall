import pytest
from pydantic import ValidationError

from reporecall.generation import DEFAULT_RAG_MODEL, RAGConfig
from reporecall.github import GitHubRepository
from reporecall.models import (
    EventMetadata,
    RAGAnswer,
    RAGContext,
    RAGEvidence,
    RAGPrompt,
    RetrievalChunk,
    RetrievalSectionType,
)


def test_rag_config_defaults_are_valid_and_immutable():
    config = RAGConfig()

    assert config.top_k == 5
    assert config.max_context_chars == 12_000
    assert config.model_name == DEFAULT_RAG_MODEL
    assert config.temperature == 0.0
    with pytest.raises(ValidationError, match="frozen"):
        config.top_k = 10


@pytest.mark.parametrize(
    "update",
    [
        {"top_k": 0},
        {"top_k": -1},
        {"max_context_chars": 0},
        {"max_context_chars": -1},
        {"model_name": ""},
        {"model_name": "   "},
        {"temperature": -0.1},
    ],
)
def test_rag_config_rejects_invalid_values(update: dict[str, object]):
    with pytest.raises(ValidationError):
        RAGConfig(**update)


def test_rag_evidence_is_immutable_and_serializable():
    evidence = _evidence()

    assert evidence.model_dump(mode="json")["section_type"] == "issue"
    assert evidence.score == 0.684866
    with pytest.raises(ValidationError, match="frozen"):
        evidence.rank = 2


@pytest.mark.parametrize("evidence_id", ["", "1", "E0", "E-1", "E01"])
def test_rag_evidence_rejects_invalid_evidence_id(evidence_id: str):
    values = _evidence().model_dump()
    values["evidence_id"] = evidence_id

    with pytest.raises(ValidationError):
        RAGEvidence(**values)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_rag_evidence_rejects_nonfinite_score(score: float):
    values = _evidence().model_dump()
    values["score"] = score

    with pytest.raises(ValidationError):
        RAGEvidence(**values)


def test_rag_context_validates_counts_ids_order_and_text():
    first = _evidence()
    second = first.model_copy(update={"evidence_id": "E2", "rank": 2})
    context = RAGContext(
        query="question",
        evidence=(first, second),
        text="evidence text",
        retrieved_count=3,
        included_evidence_count=2,
        truncated=True,
    )

    assert context.retrieved_count == 3
    assert context.included_evidence_count == 2

    for update in (
        {"included_evidence_count": 1},
        {"retrieved_count": 1},
        {"evidence": (second, first)},
        {"text": ""},
    ):
        with pytest.raises(ValidationError):
            RAGContext(**(context.model_dump() | update))


def test_empty_rag_context_is_valid():
    context = RAGContext(
        query="question",
        evidence=(),
        text="",
        retrieved_count=0,
        included_evidence_count=0,
        truncated=False,
    )

    assert context.evidence == ()


def test_rag_prompt_is_immutable_and_requires_text():
    prompt = RAGPrompt(system_prompt="system", user_prompt="user")

    with pytest.raises(ValidationError, match="frozen"):
        prompt.user_prompt = "changed"
    with pytest.raises(ValidationError):
        RAGPrompt(system_prompt="", user_prompt="user")


def test_rag_answer_preserves_counts_model_and_evidence():
    answer = RAGAnswer(
        query="question",
        answer="A historical fix exists [E1].",
        model_name="fake-model",
        evidence=(_evidence(),),
        retrieved_count=2,
        included_evidence_count=1,
        context_truncated=True,
        insufficient_evidence=False,
    )

    assert answer.evidence[0].score == 0.684866
    assert answer.model_dump(mode="json")["evidence"][0]["chunk_id"] == "chunk-1"


@pytest.mark.parametrize(
    "update",
    [
        {"query": "   "},
        {"answer": ""},
        {"model_name": "   "},
        {"model_name": None},
        {"included_evidence_count": 0},
        {"retrieved_count": 0},
    ],
)
def test_generated_rag_answer_rejects_invalid_state(update: dict[str, object]):
    values = {
        "query": "question",
        "answer": "answer",
        "model_name": "fake-model",
        "evidence": (_evidence(),),
        "retrieved_count": 1,
        "included_evidence_count": 1,
        "context_truncated": False,
        "insufficient_evidence": False,
    }
    values.update(update)

    with pytest.raises(ValidationError):
        RAGAnswer(**values)


def test_insufficient_evidence_answer_can_omit_model_and_evidence():
    answer = RAGAnswer(
        query="question",
        answer="No repository evidence was found.",
        model_name=None,
        evidence=(),
        retrieved_count=0,
        included_evidence_count=0,
        context_truncated=False,
        insufficient_evidence=True,
    )

    assert answer.insufficient_evidence is True


def _evidence() -> RAGEvidence:
    chunk = _chunk()
    return RAGEvidence(
        evidence_id="E1",
        rank=1,
        score=0.684866,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        event_id=chunk.event_id,
        repository=chunk.repository,
        section_id=chunk.section_id,
        section_type=chunk.section_type,
        artifact=chunk.artifact,
        content=chunk.content,
    )


def _chunk() -> RetrievalChunk:
    repository = GitHubRepository(owner="owner", name="repo")
    return RetrievalChunk(
        chunk_id="chunk-1",
        document_id="document-1",
        event_id="event-1",
        repository=repository,
        section_id="issue-10",
        section_type=RetrievalSectionType.ISSUE,
        chunk_index=0,
        content="Database connection leak",
        text="Repository: owner/repo\n\nDatabase connection leak",
        artifact=None,
        metadata=EventMetadata(event_id="event-1", repository=repository),
    )
