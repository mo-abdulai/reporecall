"""Explicitly synthetic data and fixed judge outputs; no real quality labels."""

from reporecall.evaluation.rag_judge_backend import RAG_JUDGE_RUBRIC
from reporecall.github import GitHubRepository
from reporecall.models import (
    ArtifactReference,
    CitationBundle,
    CitationIdentifier,
    CitationSupportAssessment,
    ContextExpansionReason,
    EngineeringRelationship,
    EventMetadata,
    ExpandedContextChunk,
    ExpandedEvidenceCitation,
    ExpectedFact,
    ExpectedFactAssessment,
    RAGEvaluationCase,
    RAGEvaluationSample,
    RAGJudgeResult,
    RerankedSearchHit,
    RetrievalChunk,
    RetrievedEvidenceCitation,
    SeedCitationReference,
)


def make_case(facts=("Synthetic fact A.", "Synthetic fact B."), case_id="synthetic-1"):
    return RAGEvaluationCase(
        case_id=case_id,
        query="  What happened in the synthetic example?\n",
        expected_facts=tuple(
            ExpectedFact(fact_id=f"F{i}", text=text) for i, text in enumerate(facts, 1)
        ),
        reference_answer="Supplemental synthetic reference.",
    )


def make_bundle(query=None, text="Synthetic evidence: fact A."):
    repo = GitHubRepository(owner="synthetic", name="fixture")
    pr = ArtifactReference(
        artifact_type="pull_request", repository=repo, identifier="1"
    )
    commit = ArtifactReference(
        artifact_type="local_git_commit", repository=repo, identifier="abc"
    )

    def chunk(identifier, artifact):
        return RetrievalChunk(
            chunk_id=identifier,
            document_id="doc",
            event_id="event",
            repository=repo,
            section_id=identifier,
            section_type="overview",
            chunk_index=0,
            content=text,
            text=text,
            artifact=artifact,
            metadata=EventMetadata(event_id="event", repository=repo),
        )

    seed = chunk("seed", pr)
    hit = RerankedSearchHit(
        chunk=seed,
        rank=1,
        reranker_score=123.456,
        reranker_model="synthetic-reranker",
        previous_hybrid_rank=1,
        rrf_score=1 / 61,
        dense_rank=1,
        dense_score=0.9,
        dense_rrf_contribution=1 / 61,
        sources=("dense",),
    )
    reason = ContextExpansionReason(
        seed_chunk_id="seed",
        relationship=EngineeringRelationship(
            source=pr,
            target=commit,
            relationship_type="pull_request_has_commit",
            evidence_type="pr_commit_list",
            evidence="Synthetic structural evidence",
        ),
        traversal_direction="outgoing",
    )
    r = RetrievedEvidenceCitation(
        citation=CitationIdentifier(kind="retrieved", index=1), hit=hit
    )
    x = ExpandedEvidenceCitation(
        citation=CitationIdentifier(kind="expanded", index=1),
        expanded_chunk=ExpandedContextChunk(
            chunk=chunk("expanded", commit), reasons=(reason,)
        ),
        seed_references=(
            SeedCitationReference(seed_chunk_id="seed", citation=r.citation),
        ),
    )
    return CitationBundle(
        query=query if query is not None else make_case().query,
        retrieved=(r,),
        expanded=(x,),
        context_truncated=True,
    )


def make_sample(answer="Synthetic fact A. [R1]", case=None, bundle=None):
    case = case or make_case()
    return RAGEvaluationSample(
        case_id=case.case_id,
        answer=answer,
        citation_bundle=bundle or make_bundle(case.query),
        answer_model="synthetic-answer-model",
    )


def judgment(
    statuses=("supported", "missing"),
    citations=(("R1", "supported"),),
    relevance=4,
    faithfulness=4,
    model="synthetic-judge",
    rubric=RAG_JUDGE_RUBRIC,
):
    return RAGJudgeResult(
        judge_model=model,
        rubric=rubric,
        answer_relevance=relevance,
        faithfulness=faithfulness,
        fact_assessments=tuple(
            ExpectedFactAssessment(fact_id=f"F{i}", status=status)
            for i, status in enumerate(statuses, 1)
        ),
        citation_assessments=tuple(
            CitationSupportAssessment(citation_label=label, status=status)
            for label, status in citations
        ),
    )


class FakeRAGEvaluationJudgeBackend:
    """Return exactly injected results or errors; never infer semantic verdicts."""

    def __init__(self, result, model_name="synthetic-judge", error=None):
        self.result = result
        self.model_name = model_name
        self.error = error
        self.requests = []

    def judge(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.result
