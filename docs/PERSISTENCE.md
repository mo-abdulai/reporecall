# PostgreSQL corpus persistence

Persistence is optional and synchronous. Existing in-memory retrieval remains available.
No connection or migration occurs when importing the package.

## Setup

Install project dependencies with `uv sync`. Supply your own PostgreSQL connection
in `DATABASE_URL`, using the `postgresql+psycopg://` driver prefix and an explicit
database name. Keep credentials in the environment or your ignored `.env` file.
`DATABASE_ECHO` defaults to false; SQL parameters are hidden even with logging enabled.

The PostgreSQL server must have pgvector installed. Run:

```sh
uv run alembic upgrade head
```

The migration enables `vector` in the public schema and creates the corpus tables.
The migration role must be permitted to create the extension, or an administrator
must enable it first. Migration versions and tables use the current schema.
`alembic downgrade base` removes the corpus tables but retains the potentially
shared vector extension. Downgrading destroys stored corpus data; use only on
an intentional rollback or disposable database.

## Schema and integrity

Four tables store engineering events, retrieval documents, retrieval chunks, and
chunk embeddings. Full canonical Pydantic JSON payloads preserve relationship
evidence, real source URLs, artifact identities, and metadata. Relational columns
support identities, foreign keys, filtering, and sorting. Mappers validate these
projections against payloads when writing and loading records.

The vector column is `vector(384)`. Changing dimensions requires a schema migration
and reindex; there is no runtime dimension switch. Multiple model versions may
coexist under `(chunk_id, model_name)`. Retrieval explicitly selects one model;
a nonempty embedding store without that model raises an error.

The JSON embedding payload preserves original Python float values for exact domain
round trips. The pgvector projection uses float32, matching FAISS's storage precision.
Unit norm validation uses the FAISS tolerances (`rtol=1e-4`, `atol=1e-5`).

`store()` owns one transaction for the supplied logical unit. Identical records
are idempotent. Conflicting records, including changed content under an existing
chunk ID, fail without partial commits. Parent identities, sections, metadata,
embedding hashes, dimensions, normalization, and finite values are validated.
Concurrent transaction conflicts fail explicitly; automatic retries and mutable
replacement/reindexing workflows are not part of this layer.

## Store and reconstruct

```python
from reporecall.config import get_settings
from reporecall.persistence import (
    DatabaseConfig,
    PersistentCorpusRepository,
    create_database_engine,
    create_session_factory,
)

settings = get_settings()
if not settings.database_url:
    raise ValueError("Set DATABASE_URL first")

engine = create_database_engine(
    DatabaseConfig(url=settings.database_url, echo=settings.database_echo)
)
sessions = create_session_factory(engine)
repository = PersistentCorpusRepository(session_factory=sessions)

# Supply existing domain objects produced by the ingestion/processing layers.
repository.store(
    events=events,
    documents=documents,
    chunks=chunks,
    embeddings=embeddings,
)

# This can run in a fresh process, with no Git/GitHub ingestion.
corpus = repository.load_corpus()
```

`load_corpus()` returns a consistent repeatable-read snapshot and cross-validates
parent relationships. Individual `load_events()`, `load_documents()`,
`load_chunks()`, and `load_embeddings()` methods are also available; separate
calls are separate snapshots. Collection ordering is explicit canonical identity
order using PostgreSQL's C collation.

Rebuild existing components from the snapshot:

```python
from reporecall.retrieval import BM25Index, FaissVectorIndex, RelationshipContextIndex
from reporecall.provenance import CitationProvenanceIndex

bm25 = BM25Index()
bm25.build(corpus.chunks)
faiss = FaissVectorIndex()
# Select one model if the store contains multiple model versions.
faiss.build([e for e in corpus.embeddings if e.model_name == embedding_backend.model_name])
relationships = RelationshipContextIndex(events=corpus.events, chunks=corpus.chunks)
provenance = CitationProvenanceIndex(documents=corpus.documents, chunks=corpus.chunks)
```

Dispose of the engine when the application no longer needs it.

## Exact dense search

```python
from reporecall.models import MetadataFilter
from reporecall.persistence import PostgresVectorRetriever

retriever = PostgresVectorRetriever(
    session_factory=sessions,
    embedding_backend=embedding_backend,
    model_name=embedding_backend.model_name,
)
hits = retriever.search(
    "database session cleanup",
    k=10,
    metadata_filter=MetadataFilter(languages=("Python",)),
)
```

The raw query reaches the supplied embedding backend unchanged. Query vectors
are transient. The SQL query applies metadata eligibility before vector top-k,
orders by pgvector's negative inner product (`<#>`) and canonical chunk ID, and
negates the distance for the returned similarity. Negative similarity remains
possible for vectors pointing in opposite directions. Ranks are assigned after
ordering. No HNSW or IVFFlat indexes are created.

Filters reuse `MetadataFilter`: AND across fields, OR within each field. SQL
uses bound values and structured metadata, never chunk text. Case-insensitive
fields use persisted Python `casefold()` projections so Unicode behavior matches
the in-memory matcher. Paths use exact component prefixes, including literal `%`
and `_`, rather than substring or unescaped LIKE matching. Actor identity includes
all canonical fields, including explicit nulls.

## Tests

Normal offline tests:

```sh
uv run pytest
```

PostgreSQL integration tests require an explicit `TEST_DATABASE_URL` pointing to
a disposable PostgreSQL/pgvector database. They never fall back to `DATABASE_URL`.
The test role needs permission to create temporary schemas and enable pgvector.
Each test applies migrations in a unique temporary schema and removes only that
schema. The shared extension is retained. Do not point these tests at production.

```sh
uv run pytest -m postgres tests/persistence
```

Without `TEST_DATABASE_URL`, these tests skip with an explicit reason. With it,
connection, extension, migration, and assertion failures fail the tests.
Tests cover rollback, round trips, schema downgrade/upgrade, filtered FAISS parity
with absolute score tolerance `1e-6`, and reconstruction of BM25, relationships,
and citations without ingestion. These synthetic tests establish implementation
correctness, not real retrieval quality.
