from collections.abc import Sequence

import pytest

from reporecall.generation import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    LLMBackendError,
    RAGConfig,
    RAGContextBuilder,
    RAGPipeline,
    RAGPromptBuilder,
    RAGResponseError,
)
from reporecall.github import GitHubRepository
from reporecall.models import (
    EventMetadata,
    RetrievalChunk,
    RetrievalSectionType,
    VectorSearchHit,
)


class FakeRetriever:
    def __init__(self, hits: Sequence[VectorSearchHit]) -> None:
        self.hits = list(hits)
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, k: int = 5) -> list[VectorSearchHit]:
        self.calls.append((query, k))
        return self.hits[:k]


class FakeLLMBackend:
    def __init__(
        self,
        answer: str = "A similar fix exists [E1].",
        *,
        error: Exception | None = None,
    ) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[tuple[str, str]] = []

    @property
    def model_name(self) -> str:
        return "fake-model"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self.error is not None:
            raise self.error
        return self.answer


def test_pipeline_runs_retrieval_context_prompt_and_generation_end_to_end():
    hits = [
        _hit("chunk-b", rank=1, score=0.684866, content="connection cleanup"),
        _hit("chunk-a", rank=2, score=0.2, content="retry worker"),
    ]
    retriever = FakeRetriever(hits)
    llm = FakeLLMBackend("A prior cleanup fix exists [E1].")
    pipeline = _pipeline(
        retriever,
        llm,
        config=RAGConfig(top_k=2, model_name="fake-model"),
    )
    query = "Have we fixed a database connection leak before?"

    answer = pipeline.answer(query)

    assert retriever.calls == [(query, 2)]
    assert len(llm.calls) == 1
    assert query in llm.calls[0][1]
    assert answer.query == query
    assert answer.answer == "A prior cleanup fix exists [E1]."
    assert answer.model_name == "fake-model"
    assert [item.chunk_id for item in answer.evidence] == ["chunk-b", "chunk-a"]
    assert [item.score for item in answer.evidence] == [0.684866, 0.2]
    assert answer.retrieved_count == 2
    assert answer.included_evidence_count == 2
    assert answer.context_truncated is False
    assert answer.insufficient_evidence is False


def test_empty_retrieval_returns_deterministic_answer_without_llm_call():
    retriever = FakeRetriever([])
    llm = FakeLLMBackend()
    pipeline = _pipeline(retriever, llm)

    answer = pipeline.answer("question")

    assert answer.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert answer.model_name is None
    assert answer.evidence == ()
    assert answer.retrieved_count == 0
    assert answer.included_evidence_count == 0
    assert answer.context_truncated is False
    assert answer.insufficient_evidence is True
    assert llm.calls == []


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_fails_before_retrieval_or_llm(query: str):
    retriever = FakeRetriever([])
    llm = FakeLLMBackend()

    with pytest.raises(RAGResponseError, match="must not be blank"):
        _pipeline(retriever, llm).answer(query)

    assert retriever.calls == []
    assert llm.calls == []


def test_context_limit_exposes_only_evidence_supplied_to_llm():
    hits = [
        _hit("chunk-1", rank=1, score=0.9, content="first"),
        _hit("chunk-2", rank=2, score=0.8, content="second"),
    ]
    single_context = RAGContextBuilder().build(
        "query",
        [hits[0]],
        max_context_chars=10_000,
    )
    config = RAGConfig(
        top_k=2,
        max_context_chars=len(single_context.text),
        model_name="fake-model",
    )

    answer = _pipeline(FakeRetriever(hits), FakeLLMBackend(), config=config).answer(
        "query"
    )

    assert [item.chunk_id for item in answer.evidence] == ["chunk-1"]
    assert answer.retrieved_count == 2
    assert answer.included_evidence_count == 1
    assert answer.context_truncated is True


def test_pipeline_rejects_empty_llm_answer():
    with pytest.raises(RAGResponseError, match="empty answer"):
        _pipeline(
            FakeRetriever([_hit("chunk-1", rank=1, score=1.0, content="source")]),
            FakeLLMBackend("   "),
        ).answer("query")


def test_pipeline_preserves_focused_llm_backend_failure():
    error = LLMBackendError("provider failed")
    pipeline = _pipeline(
        FakeRetriever([_hit("chunk-1", rank=1, score=1.0, content="source")]),
        FakeLLMBackend(error=error),
    )

    with pytest.raises(LLMBackendError) as exc_info:
        pipeline.answer("query")

    assert exc_info.value is error


@pytest.mark.parametrize("invalid_label", ["[E0]", "[E2]", "[E7]", "[E01]"])
def test_pipeline_rejects_unknown_evidence_labels(invalid_label: str):
    pipeline = _pipeline(
        FakeRetriever([_hit("chunk-1", rank=1, score=1.0, content="source")]),
        FakeLLMBackend(f"Unsupported citation {invalid_label}."),
    )

    with pytest.raises(RAGResponseError, match="unknown evidence"):
        pipeline.answer("query")


def test_pipeline_accepts_available_evidence_labels_and_uncited_text():
    hit = _hit("chunk-1", rank=1, score=1.0, content="source")

    cited = _pipeline(FakeRetriever([hit]), FakeLLMBackend("Supported [E1].")).answer(
        "query"
    )
    uncited = _pipeline(FakeRetriever([hit]), FakeLLMBackend("Limited evidence.")).answer(
        "query"
    )

    assert cited.answer == "Supported [E1]."
    assert uncited.answer == "Limited evidence."


def test_pipeline_is_deterministic_and_does_not_mutate_hits():
    hits = [_hit("chunk-1", rank=1, score=1.0, content="source")]
    original = [hit.model_dump() for hit in hits]
    pipeline = _pipeline(FakeRetriever(hits), FakeLLMBackend())

    first = pipeline.answer("query")
    second = pipeline.answer("query")

    assert first == second
    assert [hit.model_dump() for hit in hits] == original


def _pipeline(
    retriever,
    llm_backend: FakeLLMBackend,
    *,
    config: RAGConfig | None = None,
) -> RAGPipeline:
    return RAGPipeline(
        retriever=retriever,
        context_builder=RAGContextBuilder(),
        prompt_builder=RAGPromptBuilder(),
        llm_backend=llm_backend,
        config=config or RAGConfig(model_name="fake-model"),
    )


def _hit(
    chunk_id: str,
    *,
    rank: int,
    score: float,
    content: str,
) -> VectorSearchHit:
    repository = GitHubRepository(owner="owner", name="repo")
    chunk = RetrievalChunk(
        chunk_id=chunk_id,
        document_id="document-1",
        event_id="event-1",
        repository=repository,
        section_id=f"section-{chunk_id}",
        section_type=RetrievalSectionType.ISSUE,
        chunk_index=0,
        content=content,
        text=f"Repository: owner/repo\n\n{content}",
        artifact=None,
        metadata=EventMetadata(event_id="event-1", repository=repository),
    )
    return VectorSearchHit(chunk=chunk, score=score, rank=rank)
