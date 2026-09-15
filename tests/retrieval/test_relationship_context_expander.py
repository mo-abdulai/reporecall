import ast
import inspect

import pytest
from pydantic import ValidationError

from reporecall.github import GitHubRepository
from reporecall.models import (
    ArtifactReference,
    ContextExpansionReason,
    EngineeringEvent,
    EngineeringRelationship,
    EventMetadata,
    ExpandedContextChunk,
    ExpandedContextResult,
    RelationshipEvidenceType,
    RelationshipType,
    RerankedRetrievalResult,
    RerankedSearchHit,
    RetrievalChunk,
)
from reporecall.retrieval import (
    RelationshipContextError,
    RelationshipContextExpander,
    RelationshipContextExpansionConfig,
    RelationshipContextIndex,
)

REPO = GitHubRepository(owner="owner", name="repo")


def artifact(kind="pull_request", identifier="20", repository=REPO):
    return ArtifactReference(
        artifact_type=kind, identifier=identifier, repository=repository
    )


def chunk(identifier, ref=None, event_id="event", repository=REPO, index=0):
    return RetrievalChunk(
        chunk_id=identifier,
        document_id="doc",
        event_id=event_id,
        repository=repository,
        section_id="section",
        section_type="overview",
        chunk_index=index,
        content="source",
        text="exact source",
        artifact=ref,
        metadata=EventMetadata(event_id=event_id, repository=repository),
    )


def edge(
    source, target, kind="pull_request_has_commit", evidence="pr_commit_list", **kwargs
):
    return EngineeringRelationship(
        source=source,
        target=target,
        relationship_type=kind,
        evidence_type=evidence,
        **kwargs,
    )


def event(edges=(), contextual=(), event_id="event", repository=REPO):
    return EngineeringEvent(
        event_id=event_id,
        repository=repository,
        anchor=artifact(repository=repository),
        relationships=list(edges),
        contextual_relationships=list(contextual),
    )


def ranked(chunks):
    hits = tuple(
        RerankedSearchHit(
            chunk=c,
            rank=i,
            reranker_score=100 - i,
            reranker_model="test",
            previous_hybrid_rank=i,
            rrf_score=1 / (60 + i),
            dense_rank=i,
            dense_score=0.5,
            dense_rrf_contribution=1 / (60 + i),
            sources=("dense",),
        )
        for i, c in enumerate(chunks, 1)
    )
    return RerankedRetrievalResult(
        query="raw query",
        model_name="test",
        input_candidate_count=len(hits),
        reranked_candidate_count=len(hits),
        returned_count=len(hits),
        candidate_k=max(1, len(hits)),
        top_k=max(1, len(hits)),
        hits=hits,
    )


def expand(seeds, chunks, edges=(), contextual=(), **config):
    return RelationshipContextExpander(
        index=RelationshipContextIndex(
            events=[event(edges, contextual)], chunks=chunks
        ),
        config=RelationshipContextExpansionConfig(**config),
    ).expand(ranked(seeds))


RELATIONS = [
    ("issue_has_comment", "issue", "issue_comment", "parent_identifier"),
    (
        "pull_request_has_comment",
        "pull_request",
        "pull_request_comment",
        "parent_identifier",
    ),
    (
        "pull_request_has_review",
        "pull_request",
        "pull_request_review",
        "parent_identifier",
    ),
    (
        "review_has_comment",
        "pull_request_review",
        "pull_request_review_comment",
        "parent_identifier",
    ),
    (
        "pull_request_has_review_comment",
        "pull_request",
        "pull_request_review_comment",
        "parent_identifier",
    ),
    ("pull_request_changes_file", "pull_request", "pull_request_file", "pr_file_list"),
    (
        "local_commit_changes_file",
        "local_git_commit",
        "local_changed_file",
        "local_commit_file_list",
    ),
    (
        "pull_request_has_commit",
        "pull_request",
        "github_commit_reference",
        "pr_commit_list",
    ),
    (
        "github_commit_matches_local_commit",
        "github_commit_reference",
        "local_git_commit",
        "exact_sha_match",
    ),
    ("pull_request_closes_issue", "pull_request", "issue", "closing_keyword"),
    ("commit_closes_issue", "local_git_commit", "issue", "closing_keyword"),
]


@pytest.mark.parametrize("kind,source_type,target_type,evidence", RELATIONS)
@pytest.mark.parametrize("incoming", [False, True])
def test_all_strong_edges_both_directions(
    kind, source_type, target_type, evidence, incoming
):
    source, target = artifact(source_type, "abc123"), artifact(target_type, "abc123")
    relationship = edge(source, target, kind, evidence, source_field="body").model_copy(
        update={"evidence": "original evidence"}
    )
    a, b = chunk("a", source), chunk("b", target)
    seed, related = (b, a) if incoming else (a, b)
    result = expand([seed], [a, b], [relationship])
    assert result.seed_hits == ranked([seed]).hits
    assert result.expanded_chunks[0].chunk == related
    reason = result.expanded_chunks[0].reasons[0]
    assert reason.relationship == relationship
    assert reason.evidence_types == (RelationshipEvidenceType(evidence),)
    assert reason.traversal_direction.value == ("incoming" if incoming else "outgoing")
    assert not result.truncated
    assert "rank" not in result.expanded_chunks[0].model_dump()
    assert "score" not in result.expanded_chunks[0].model_dump()


def test_one_hop_and_same_event_are_not_transitive():
    pr, commit, issue = (
        artifact(),
        artifact("github_commit_reference", "abc"),
        artifact("issue", "10"),
    )
    a, b, c = chunk("a", pr), chunk("b", commit), chunk("c", issue)
    edges = [
        edge(pr, commit),
        edge(commit, issue, "commit_closes_issue", "closing_keyword"),
    ]
    assert [x.chunk for x in expand([a], [a, b, c], edges).expanded_chunks] == [b]
    assert expand([a], [a, b, c]).expanded_count == 0


@pytest.mark.parametrize(
    "kind", ["pull_request_references_issue", "commit_references_issue"]
)
def test_weak_references_require_opt_in_and_keep_evidence(kind):
    a, b = artifact(), artifact("issue", "10")
    x, y = chunk("x", a), chunk("y", b)
    relationship = edge(a, b, kind, "text_reference")
    assert expand([x], [x, y], [relationship]).expanded_count == 0
    result = expand(
        [x], [x, y], contextual=[relationship], include_contextual_relationships=True
    )
    assert result.expanded_chunks[0].reasons[0].contextual
    assert result.expanded_chunks[0].reasons[0].relationship_type == RelationshipType(
        kind
    )


def test_strong_context_beats_weak_even_from_earlier_seed():
    a, b, c, d = [artifact(identifier=str(i)) for i in range(4)]
    chunks = [chunk(str(i), ref) for i, ref in enumerate([a, b, c, d])]
    edges = [edge(a, c, "pull_request_references_issue", "text_reference"), edge(b, d)]
    result = expand(
        chunks[:2],
        chunks,
        edges,
        include_contextual_relationships=True,
        max_total_expanded_chunks=1,
    )
    assert result.expanded_chunks[0].chunk == chunks[3]
    assert result.truncated


def test_multiple_reasons_seeds_and_full_evidence_survive_global_limit():
    a, b, c = [artifact(identifier=str(i)) for i in range(3)]
    x, y, z = chunk("x", a), chunk("y", b), chunk("z", c)
    relations = [
        edge(a, c),
        edge(b, c),
        edge(a, c, evidence="github_commit_associated_pull_request"),
    ]
    relations.append(relations[0].model_copy(update={"evidence": "additional payload"}))
    result = expand([x, y], [x, y, z], relations, max_total_expanded_chunks=1)
    assert result.expanded_count == 1
    assert len(result.expanded_chunks[0].reasons) == 4
    assert not result.truncated
    assert expand([x, z], [x, y, z], [edge(a, c)]).expanded_count == 0


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"max_chunks_per_related_artifact": 2}, 2),
        ({"max_chunks_per_related_artifact": 10, "max_related_chunks_per_seed": 3}, 3),
        (
            {
                "max_chunks_per_related_artifact": 10,
                "max_related_chunks_per_seed": 10,
                "max_total_expanded_chunks": 1,
            },
            1,
        ),
    ],
)
def test_limits_and_canonical_chunk_order(config, expected):
    a, b = artifact(), artifact("local_changed_file", "path")
    seed = chunk("seed", a)
    related = [chunk(f"chunk-{9 - i}", b, index=i) for i in range(5)]
    result = expand([seed], [seed, *reversed(related)], [edge(a, b)], **config)
    assert [item.chunk for item in result.expanded_chunks] == related[:expected]
    assert result.truncated


def test_empty_artifactless_missing_target_and_seed_k():
    a = chunk("a", artifact())
    b = chunk("b")
    assert expand([], []).seed_count == 0
    assert expand([b], [b]).seed_hits[0].chunk == b
    result = expand(
        [a, b], [a, b], [edge(a.artifact, artifact("issue", "missing"))], seed_k=1
    )
    assert result.seed_count == 1
    assert result.expanded_count == 0
    assert not result.truncated


def test_determinism_and_no_mutation():
    a, b, c = [artifact(identifier=str(i)) for i in range(3)]
    chunks = [chunk(str(i), ref) for i, ref in enumerate([a, b, c])]
    relations = [edge(a, b), edge(a, c)]
    original = event(relations)
    before = original.model_dump()
    first = RelationshipContextExpander(
        index=RelationshipContextIndex(events=[original], chunks=chunks)
    ).expand(ranked(chunks[:1]))
    original.relationships.clear()  # Index holds a snapshot, not this mutable list.
    second = expand(chunks[:1], list(reversed(chunks)), list(reversed(relations)))
    assert first == second
    assert before["relationships"] == [r.model_dump() for r in relations]


def test_event_and_repository_isolation():
    ref, target = artifact(), artifact("issue", "10")
    seed = chunk("seed", ref)
    other = chunk("other", target, event_id="other")
    index = RelationshipContextIndex(
        events=[event([edge(ref, target)]), event(event_id="other")],
        chunks=[seed, other],
    )
    assert (
        RelationshipContextExpander(index=index).expand(ranked([seed])).expanded_count
        == 0
    )
    repo_b = GitHubRepository(owner="owner", name="repo-b")
    foreign = artifact("issue", "10", repo_b)
    index = RelationshipContextIndex(
        events=[
            event(
                contextual=[
                    edge(
                        ref, foreign, "pull_request_references_issue", "text_reference"
                    )
                ]
            ),
            event(event_id="b", repository=repo_b),
        ],
        chunks=[seed, chunk("b", foreign, "b", repo_b)],
    )
    result = RelationshipContextExpander(
        index=index,
        config=RelationshipContextExpansionConfig(
            include_contextual_relationships=True
        ),
    ).expand(ranked([seed]))
    assert result.expanded_count == 0
    assert not result.truncated


@pytest.mark.parametrize(
    "field",
    [
        "seed_k",
        "max_related_chunks_per_seed",
        "max_chunks_per_related_artifact",
        "max_total_expanded_chunks",
    ],
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_limits(field, value):
    with pytest.raises(ValidationError):
        RelationshipContextExpansionConfig(**{field: value})


def test_index_consistency_errors():
    a = chunk("a", artifact())
    with pytest.raises(RelationshipContextError, match="Duplicate chunk"):
        RelationshipContextIndex(events=[event()], chunks=[a, a])
    with pytest.raises(RelationshipContextError, match="Conflicting duplicate"):
        RelationshipContextIndex(
            events=[event(), event([edge(artifact(), artifact("issue", "1"))])],
            chunks=[],
        )
    with pytest.raises(RelationshipContextError, match="Missing"):
        RelationshipContextExpander(
            index=RelationshipContextIndex(events=[], chunks=[])
        ).expand(ranked([a]))
    with pytest.raises(RelationshipContextError, match="Missing"):
        RelationshipContextIndex(events=[], chunks=[a])
    bad = a.model_copy(update={"repository": GitHubRepository(owner="x", name="y")})
    with pytest.raises(RelationshipContextError, match="repository"):
        RelationshipContextIndex(events=[event()], chunks=[bad])
    index = RelationshipContextIndex(events=[event()], chunks=[a])
    with pytest.raises(RelationshipContextError, match="conflicts"):
        index.validate_seed(a.model_copy(update={"text": "stale"}))
    with pytest.raises(RelationshipContextError, match="repository"):
        index.validate_seed(bad)


def test_reason_and_result_validation_and_immutability():
    a, b = artifact(), artifact("issue", "10")
    x, y = chunk("x", a), chunk("y", b)
    reason = ContextExpansionReason(
        seed_chunk_id="x", relationship=edge(a, b), traversal_direction="outgoing"
    )
    for update in [
        {"seed_chunk_id": " "},
        {"traversal_direction": "sideways"},
        {"relationship": {"source": a, "target": b, "relationship_type": "bad"}},
    ]:
        with pytest.raises(ValidationError):
            ContextExpansionReason.model_validate(reason.model_dump() | update)
    with pytest.raises(ValidationError):
        ExpandedContextChunk(chunk=y, reasons=())
    with pytest.raises(ValidationError):
        ExpandedContextChunk(chunk=x, reasons=(reason,))
    item = ExpandedContextChunk(chunk=y, reasons=(reason,))
    result = ExpandedContextResult(
        query="q", seed_hits=ranked([x]).hits, expanded_chunks=(item,)
    )
    assert result.seed_count == result.expanded_count == 1
    with pytest.raises(ValidationError):
        result.truncated = True
    with pytest.raises(ValidationError):
        ExpandedContextResult(query="q", seed_hits=(), expanded_chunks=(item,))


def test_expansion_modules_have_no_execution_dependencies():
    import reporecall.models.context_expansion as models
    import reporecall.retrieval.relationship_context_expander as expander
    import reporecall.retrieval.relationship_context_index as index

    forbidden = {
        "VectorRetriever",
        "KeywordRetriever",
        "HybridRetriever",
        "FaissVectorIndex",
        "BM25Index",
        "EmbeddingBackend",
        "CrossEncoderReranker",
        "OpenAI",
        "LLMBackend",
        "RAGPipeline",
        "QueryUnderstandingBackend",
    }
    for module in (models, expander, index):
        tree = ast.parse(inspect.getsource(module))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not (names | imports) & forbidden
