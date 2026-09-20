"""Small explicitly synthetic corpus with known vectors and rich metadata."""

from reporecall.github import GitHubRepository
from reporecall.models import (
    ArtifactReference,
    ChunkEmbedding,
    EngineeringEvent,
    EngineeringRelationship,
    EventActor,
    EventMetadata,
    RetrievalChunk,
    RetrievalDocument,
    RetrievalDocumentSection,
    RetrievalSource,
)
from reporecall.persistence import PersistentCorpus
from reporecall.persistence.mapping import source_hash

MODEL = "synthetic-384"


def vector(x=1.0, y=0.0):
    return (x, y, *((0.0,) * 382))


class FakeEmbeddingBackend:
    model_name = MODEL
    normalized = True

    def __init__(self, output=None):
        self.output = [vector()] if output is None else output
        self.inputs = []

    def embed(self, texts):
        self.inputs.append(tuple(texts))
        return self.output


def corpus_data():
    events, documents, chunks, embeddings = [], [], [], []
    for index, (name, path, language, values) in enumerate(
        (
            ("a", "src/database_old.py", "Java", vector()),
            ("b", "src/database/session.py", "Python", vector(0.8, 0.6)),
            ("c", "tests/session_test.py", "Python", vector(0.6, 0.8)),
            ("d", "src/database/session.py", "Python", vector(-1.0, 0.0)),
        )
    ):
        repo = GitHubRepository(
            owner="synthetic", name="other" if name == "d" else "fixture"
        )
        event_id, document_id = f"event-{name}", f"doc-{name}"
        pr = ArtifactReference(
            artifact_type="pull_request",
            repository=repo,
            identifier=str(10 + index if name != "d" else 10),
        )
        commit = ArtifactReference(
            artifact_type="local_git_commit", repository=repo, identifier=f"sha-{name}"
        )
        relationship = EngineeringRelationship(
            source=pr,
            target=commit,
            relationship_type="pull_request_has_commit",
            evidence_type="pr_commit_list",
            evidence="Synthetic source record",
        )
        events.append(
            EngineeringEvent(
                event_id=event_id,
                repository=repo,
                anchor=pr,
                relationships=[relationship],
            )
        )
        metadata = EventMetadata(
            event_id=event_id,
            repository=repo,
            issue_numbers=(10,),
            pull_request_numbers=(10,),
            commit_shas=(f"sha-{name}",),
            labels=("BUG", "Straße") if name == "b" else ("feature",),
            milestones=("Release A",),
            changed_paths=(path,),
            languages=(language,),
            file_extensions=(".PY",),
            directories=(path.rsplit("/", 1)[0],),
            authors=(
                EventActor(actor_type="github_user", identifier="alice", name="Alice"),
            ),
            participants=(
                EventActor(
                    actor_type="git_author",
                    identifier="local",
                    email="synthetic@example.test",
                ),
            ),
            added_lines=index + 1,
            deleted_lines=index,
            changed_lines=2 * index + 1,
            has_tests=name == "c",
            test_paths=(path,) if name == "c" else (),
            has_documentation_changes=name == "b",
            documentation_paths=("docs/example.md",) if name == "b" else (),
            has_configuration_changes=name == "b",
            configuration_paths=("config.toml",) if name == "b" else (),
            has_dependency_changes=name == "b",
            dependency_paths=("requirements.txt",) if name == "b" else (),
        )
        sections = tuple(
            RetrievalDocumentSection(
                section_id=f"section-{name}-{suffix}",
                section_type=kind,
                heading="Synthetic section",
                content=f"Synthetic session cleanup {name} {suffix}",
                artifact=artifact,
            )
            for suffix, kind, artifact in (
                ("pr", "pull_request", pr),
                ("commit", "commit", commit),
            )
        )
        document = RetrievalDocument(
            document_id=document_id,
            event_id=event_id,
            repository=repo,
            title="Synthetic event",
            sections=sections,
            text="\n".join(s.content for s in sections),
            metadata=metadata,
            sources=(
                RetrievalSource(
                    artifact=pr,
                    url=f"https://example.test/{name}/pr/10",
                    label="Stored PR",
                ),
                RetrievalSource(artifact=commit, url=None, label="Local commit"),
            ),
        )
        documents.append(document)
        for suffix, section in zip(("pr", "commit"), sections, strict=True):
            chunk = RetrievalChunk(
                chunk_id=f"{name}-{suffix}",
                document_id=document_id,
                event_id=event_id,
                repository=repo,
                section_id=section.section_id,
                section_type=section.section_type,
                chunk_index=0,
                content=section.content,
                text=section.content,
                artifact=section.artifact,
                metadata=metadata,
            )
            chunks.append(chunk)
            embeddings.append(
                ChunkEmbedding(
                    chunk_id=chunk.chunk_id,
                    document_id=document_id,
                    event_id=event_id,
                    repository=repo,
                    section_id=chunk.section_id,
                    section_type=chunk.section_type,
                    model_name=MODEL,
                    dimension=384,
                    normalized=True,
                    source_text_sha256=source_hash(chunk.text),
                    vector=values,
                )
            )
    return PersistentCorpus(
        events=tuple(events),
        documents=tuple(documents),
        chunks=tuple(chunks),
        embeddings=tuple(embeddings),
    )


def store_data(repository, corpus=None):
    corpus = corpus or corpus_data()
    repository.store(
        events=corpus.events,
        documents=corpus.documents,
        chunks=corpus.chunks,
        embeddings=corpus.embeddings,
    )
    return corpus
