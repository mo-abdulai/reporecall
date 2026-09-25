# HTTP API

RepoRecall serves persisted repository evidence through FastAPI. It does not generate
answers or ingest/index repositories through HTTP. Search returns ranked evidence,
optional one-hop relationship context, and source-faithful citations.

## Local setup

Install Python dependencies with `uv sync`. Start PostgreSQL with pgvector available,
set `DATABASE_URL` to a `postgresql+psycopg://...` URL in your environment or local
`.env`, and apply migrations:

```bash
uv run alembic upgrade head
uv run uvicorn reporecall.api.app:create_app --factory --reload
```

The default server listens on `127.0.0.1:8000`. Use Uvicorn's `--host` and `--port`
flags to change this. Do not expose this unauthenticated development service to
untrusted networks. The API does not run migrations automatically.

Populate the database using the existing `PersistentCorpusRepository` Python API.
A migrated, empty database is valid: corpus counts and search results will be empty.
For a nonempty corpus, the selected embedding model must cover every chunk.
`API_EMBEDDING_MODEL` defaults to `sentence-transformers/all-MiniLM-L6-v2`; persisted
vectors must be normalized and 384-dimensional. Multiple stored models are allowed,
but search uses this one explicit model.

## Configuration

- `DATABASE_URL`: required for corpus-backed services.
- `API_EMBEDDING_MODEL`: must match the persisted embeddings exactly.
- `API_RERANKER_MODEL`: defaults to `cross-encoder/ms-marco-MiniLM-L6-v2`.
- `API_CANDIDATE_K`: internal dense/BM25/fusion/reranker depth, 50–200; default 50.
- `API_QUERY_MODEL`: independently configurable query-understanding model.
- `OPENAI_API_KEY`: needed only when calling query understanding.
- `API_CORS_ORIGINS`: JSON array of explicit browser origins; default `[]`.
- `API_DOCS_ENABLED`: defaults to `true`.

Models load lazily on the first nonempty search. This may download model weights;
preprovision them for offline deployment. Startup builds the BM25 and relationship/
provenance indexes once from persisted data. Restart after corpus updates; hot reload
of persisted evidence is not implemented. Each worker owns its own resources and
serializes neural search access. Database sessions remain scoped to operations.

## Endpoints

All application endpoints use `/api/v1`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Cheap process liveness, independent of database/models |
| GET | `/ready` | Shared service initialization and database/schema readiness |
| GET | `/corpus` | Counts and explicit stored embedding model summaries |
| GET | `/repositories` | Deterministically ordered repository identities |
| GET | `/events/{event_id}` | Canonical engineering event |
| GET | `/documents/{document_id}` | Source-faithful retrieval document |
| GET | `/chunks/{chunk_id}` | Source-faithful retrieval chunk |
| POST | `/query/understand` | Interpret query constraints without searching |
| POST | `/search` | Reranked hybrid evidence search |

Interactive schemas: `/docs`, `/redoc`, and `/openapi.json`. Disable these with
`API_DOCS_ENABLED=false`. No `/ask` or `/chat` endpoint exists.

```bash
curl http://127.0.0.1:8000/api/v1/ready
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"database session cleanup","top_k":5,"metadata_filter":{"languages":["Python"]},"expand_relationships":true,"include_citations":true}'
```

Search preserves the query exactly and accepts 1–50 results. Filters use existing
metadata AND/OR semantics; unknown fields are rejected. Query understanding is
never called implicitly. To use it, call `/query/understand`, review the returned
constraints, then explicitly supply your chosen query/filter to `/search`.

Search runs PostgreSQL exact dense retrieval and BM25, hybrid union, reciprocal
rank fusion, cross-encoder reranking, optional relationship expansion, then optional
citation resolution. `hits` contain original ranked chunks and diagnostics; scores
are not probabilities. `expanded_context` contains separate unranked chunks and
relationship reasons, never fabricated retrieval scores. `context_truncated` preserves
expansion completeness diagnostics.

Citation wrappers expose canonical `label` (`R1`, `X1`), original structured evidence,
unique real `urls`, and expanded seed references. Missing URLs stay empty. Disabling
citations returns `null`; enabling citations on empty evidence returns an empty
bundle. Disabling expansion returns `[]` and still allows retrieved citations.

## Errors and readiness

Errors consistently return `{"error":{"code":"...","message":"..."}}` without
provider messages, database URLs, credentials, or tracebacks.

- 422: invalid input.
- 404: unknown resource/route.
- 503: unavailable database, required schema, embedding/reranker, or corpus service.
- 502: query-understanding provider or structured-output failure.
- 500: unexpected internal failure.

Missing database configuration or a failed initial database connection keeps
liveness available and readiness at 503. Restart after correcting startup failures.
Readiness checks the database on each call but deliberately does not download models,
run inference, or contact OpenAI. It therefore does not promise provider availability.

## Tests

```bash
UV_CACHE_DIR=/tmp/reporecall-uv-cache uv run pytest
TEST_DATABASE_URL='postgresql+psycopg://USER:PASSWORD@localhost/TEST_DB' \
  UV_CACHE_DIR=/tmp/reporecall-uv-cache uv run pytest tests/api/integration
```

Integration tests create and remove isolated schemas in the explicitly selected test
database. They use real PostgreSQL/pgvector and fake embedding/reranker backends;
no OpenAI requests or model downloads occur. Without `TEST_DATABASE_URL`, these
tests skip. Synthetic fixtures demonstrate pipeline behavior, not retrieval quality.
