# RepoRecall — Codex Instructions

RepoRecall is an engineering-history intelligence system that helps developers find similar historical bugs, understand how they were previously fixed, and retrieve supporting evidence from repository history.

The project is being built incrementally to learn production-grade RAG, information retrieval, metadata search, repository ingestion, and evaluation.

Do not skip ahead or implement future phases unless the current task explicitly requests them.

---

## 1. Project Goal

RepoRecall should eventually answer questions such as:

* Have we fixed a similar bug before?
* Find previous authentication-related fixes.
* How was this database connection leak fixed historically?
* Show past fixes that modified a specific file.
* Find Python bugs from the last two years where tests were added.
* Does this stack trace resemble a previously fixed issue?

The long-term system will ingest and connect:

* Git commits
* code diffs
* changed files
* pull requests
* GitHub issues
* issue comments
* pull request comments
* reviews
* labels
* tests
* source-code metadata

It will later support:

* metadata-aware retrieval
* vector retrieval
* keyword/BM25 retrieval
* hybrid retrieval
* rank fusion
* reranking
* query understanding
* relationship-aware context construction
* grounded LLM generation
* citations
* retrieval evaluation
* RAG evaluation

---

## 2. Current Development Philosophy

RepoRecall must be built one phase at a time.

For each task:

1. Implement only the requested phase.
2. Keep the implementation understandable.
3. Avoid premature abstractions.
4. Avoid unnecessary dependencies.
5. Add tests for important behavior.
6. Run validation before considering the task complete.
7. Explain major design decisions.
8. Stop after completing the requested task.

Do not automatically continue into the next phase.

The purpose of this project is not only to produce working software. The code should remain understandable enough that the project owner can explain the architecture and implementation during technical interviews.

---

## 3. Technology Stack

Current baseline:

* Python 3.12
* `uv` for Python dependency and environment management
* `src/` package layout
* Pydantic for structured domain models
* Pydantic Settings for application configuration
* GitPython for local Git operations
* HTTPX for HTTP/API communication
* pytest for testing
* Ruff for linting
* mypy for static type checking

Future technologies may include:

* GitHub REST API
* LangChain components where useful
* Sentence Transformers or another embedding model
* FAISS for the initial vector-search baseline
* BM25
* reranking models
* OpenAI
* PostgreSQL
* pgvector
* FastAPI
* Next.js
* Docker
* Kubernetes

Do not introduce future technologies until their phase explicitly requires them.

---

## 4. Dependency Management

`pyproject.toml` is the main dependency declaration.

`uv.lock` is the reproducible lock file.

`requirements.txt` exists for deployment compatibility.

Add production dependencies using:

```bash
uv add <package>
```

Add development dependencies using:

```bash
uv add --dev <package>
```

Do not manually add package versions directly to `requirements.txt`.

After changing production dependencies, regenerate deployment requirements using:

```bash
uv export --format requirements.txt --no-hashes -o requirements.txt
```

Keep:

* `pyproject.toml`
* `uv.lock`
* `requirements.txt`

in sync.

Do not introduce a dependency when equivalent functionality can reasonably be implemented using the Python standard library or an existing project dependency.

---

## 5. Expected Repository Layout

The project will grow over time. Do not create future modules merely because they appear below.

Target architecture:

```text
reporecall/
│
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── requirements.txt
├── .python-version
├── .env
├── .env.example
├── .gitignore
│
├── src/
│   └── reporecall/
│       ├── __init__.py
│       ├── config.py
│       │
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── git_loader.py
│       │   ├── github_loader.py
│       │   └── repository_loader.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   └── records.py
│       │
│       ├── processing/
│       ├── documents/
│       ├── embeddings/
│       ├── storage/
│       ├── retrieval/
│       ├── generation/
│       ├── evaluation/
│       ├── api/
│       └── utils/
│
├── tests/
│
├── data/
│   ├── repositories/
│   └── cache/
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── ROADMAP.md
│   ├── DATA_MODEL.md
│   └── RETRIEVAL_DESIGN.md
│
└── scripts/
```

Only create directories and files required for the current phase.

---

## 6. Architectural Boundaries

Keep responsibilities separated.

### Ingestion

Responsible for retrieving raw source data.

Examples:

* Git repository history
* GitHub API data
* pull requests
* issues
* comments
* reviews

Ingestion code must not contain RAG or LLM logic.

GitHub relationship-evidence loaders may perform network requests, but they must
normalize API responses into domain models before passing evidence to processing.

### Models

Contains structured domain models.

Models should not depend directly on:

* GitPython objects
* HTTPX response objects
* LangChain
* OpenAI
* database clients

Convert external data into RepoRecall domain models at system boundaries.

### Processing

Responsible for tasks such as:

* normalization
* entity linking
* metadata extraction
* code parsing

Deterministic relationship linkers and enrichers belong in this layer.

Relationship linking and enrichment must:

* operate only on normalized records and evidence supplied by the caller
* avoid Git, GitHub API, HTTP, database, and external-service calls
* preserve explicit evidence for every emitted relationship
* preserve provenance when multiple evidence sources support a relationship
* keep LLM or semantic inference separate from deterministic extraction
* represent direct evidence without creating transitive relationship conclusions

Engineering event construction also belongs in processing and must:

* operate entirely offline over normalized records and direct relationships
* use an explicit, deterministic set of event-forming relationship types
* keep weak references contextual rather than merging events through them
* preserve relationship direction and avoid creating transitive relationships
* use stable anchors, event identifiers, artifact ordering, and event ordering
* defer semantic event classification to metadata processing

Deterministic event metadata extraction must:

* operate entirely offline without mutating `EngineeringEvent`
* return a separate immutable metadata model
* derive values only from normalized event records
* use explicit path classifiers and extension-to-language mappings
* preserve source identity instead of reconciling people heuristically
* define precedence where GitHub and local Git expose overlapping statistics
* remain separate from semantic classification and retrieval-document generation

Retrieval-document generation must:

* operate entirely offline from a supplied `EngineeringEvent` and matching `EventMetadata`
* produce one deterministic document per event without chunking
* preserve typed section boundaries as the source of canonical rendered text
* preserve source-authored bodies, comments, commit messages, stack traces, and patches without semantic rewriting
* retain lightweight artifact provenance and real source URLs without fabricating links
* prevent obvious duplicate GitHub/local commit and changed-file rendering while retaining richer local patches and both source identities
* group multiple evidence sources for readability without mutating or collapsing the underlying relationships
* keep direct and contextual relationships visibly distinct
* avoid current-time fields, semantic inference, embeddings, retrieval, database, and external-service calls

Retrieval chunking must:

* operate entirely offline over structured `RetrievalDocument.sections`
* keep every chunk within exactly one document section and preserve its section ID and type
* use deterministic paragraph, line, word, and character boundaries for prose without semantic rewriting
* preserve diff headers, hunk boundaries, line markers, and configurable line overlap for patches
* retain document, event, repository, artifact, and structured metadata provenance
* use stable content-derived chunk identities and section-local chunk indices
* remain separate from tokenizers, embeddings, vector storage, search, ranking, and generation

### Documents

Responsible for converting normalized engineering history into retrieval-ready documents.

### Embeddings

Responsible for converting retrieval-ready chunk text into validated dense vectors.

Chunk embedding must:

* embed `RetrievalChunk.text` exactly, without rewriting or preprocessing it
* keep ML-library model execution behind a small backend interface
* preserve chunk, document, event, repository, section, model, normalization, and source-text identities
* validate output count, dimensions, and finite vector values before producing records
* keep unit tests network-independent and free from real model downloads
* remain separate from vector persistence, indexing, similarity search, and query retrieval

### Storage

Responsible for:

* vector indexes
* relational persistence
* metadata persistence

### Retrieval

Responsible for:

* semantic search
* keyword search
* metadata filtering
* hybrid retrieval
* rank fusion
* reranking
* query parsing

The initial dense retrieval baseline must:

* use exact inner-product search over L2-normalized vectors
* require explicit model, dimension, and normalization compatibility
* embed raw queries with the same backend and model used for indexed chunks
* preserve deterministic FAISS row-to-chunk identity and complete chunk provenance
* validate finite vectors, finite scores, duplicate identities, and stale source-text hashes
* keep dense retrieval separate from metadata filtering, keyword retrieval, reranking, context construction, and RAG

Structured metadata filtering must:

* select eligible chunks from existing `EventMetadata` before vector ranking
* avoid parsing chunk text or repeating metadata extraction during retrieval
* use AND semantics across populated fields and OR semantics within one field
* constrain eligibility without boosting or otherwise modifying dense similarity scores
* preserve the raw query independently from filter values
* leave natural-language filter extraction to future query-understanding logic

The initial lexical retrieval baseline must:

* index source-faithful `RetrievalChunk.text` without rewriting chunk content
* use deterministic engineering-aware tokenization that preserves complete technical identifiers
* remain independently usable without embeddings, vector similarity, or FAISS
* apply structured metadata eligibility before lexical top-k selection
* preserve raw BM25 scores as lexical relevance scores distinct from dense similarity scores
* leave dense and lexical score or rank fusion to a separate retrieval layer

### Generation

Responsible for:

* context construction
* prompts
* LLM answer generation
* grounding
* citations

Grounded generation must:

* run only after retrieval and answer from the retrieved repository evidence
* treat repository content as untrusted data rather than executable instructions
* preserve evidence identifiers and provenance through prompts and answers
* keep provider-specific API logic behind a small LLM backend abstraction
* keep retrieval and generation independently testable
* return a deterministic insufficient-evidence result when retrieval is empty
* avoid real LLM API calls in automated tests

### Evaluation

Responsible for:

* retrieval metrics
* RAG evaluation
* latency measurements
* experimental comparison

Do not mix these responsibilities unnecessarily.

---

## 7. Important Domain Relationships

RepoRecall should eventually understand software history as connected engineering events.

The core relationship is:

```text
Issue
  ↓
Pull Request
  ↓
Commit
  ↓
Changed File
  ↓
Code Diff
  ↓
Tests
```

Do not treat these only as disconnected text documents.

Where reliable relationships exist, preserve identifiers such as:

* repository name
* issue number
* PR number
* commit SHA
* file path

These identifiers will later be used for context expansion, metadata filtering, and citations.

---

## 8. Metadata Is a First-Class Feature

RepoRecall is not merely semantic search over text.

Metadata must eventually be searchable and filterable.

Examples of useful deterministic metadata:

* repository
* commit SHA
* author
* date
* file path
* directory
* programming language
* additions
* deletions
* change type
* issue number
* PR number
* issue labels
* PR labels
* tests changed
* tests added

Possible future inferred metadata:

* component
* bug category
* root cause
* fix strategy
* severity

Keep deterministic metadata separate from LLM-inferred metadata when practical.

Never present inferred metadata as unquestionably factual.

---

## 9. Coding Standards

Use Python 3.12 syntax.

Use type hints for:

* function parameters
* return values
* class attributes where appropriate

Prefer:

* small focused functions
* small focused classes
* explicit data transformations
* descriptive variable names
* clear control flow
* dependency injection where it meaningfully improves testing

Avoid:

* deeply nested logic
* unnecessary inheritance
* generic "manager" classes with unrelated responsibilities
* giant utility modules
* large god objects
* premature factory patterns
* unnecessary framework abstractions

Public classes and important public functions should have concise docstrings explaining purpose rather than restating the function name.

---

## 10. Error Handling

Raise meaningful exceptions.

Examples:

* invalid repository path
* invalid Git repository
* failed clone
* Git command failure
* malformed API response
* authentication failure
* GitHub rate limiting
* missing required configuration

Do not use broad exception handling such as:

```python
try:
    ...
except Exception:
    pass
```

Do not silently ignore important failures.

Preserve the underlying cause using exception chaining when useful:

```python
raise RepositoryCloneError(...) from exc
```

---

## 11. Configuration and Secrets

Never hardcode:

* GitHub tokens
* OpenAI API keys
* database passwords
* private repository credentials
* deployment secrets

Read configuration from environment variables.

Use `.env` only for local development.

`.env` must never be committed.

`.env.example` should contain variable names and safe placeholder/default values.

Example:

```env
GITHUB_TOKEN=
GITHUB_API_URL=https://api.github.com
REPO_DATA_DIR=data/repositories
CACHE_DIR=data/cache
```

---

## 12. GitHub and Git Separation

Local Git operations and GitHub API operations must remain separate.

### Local Git

Use local repository history for:

* commits
* SHAs
* parent relationships
* file changes
* diffs
* commit metadata

### GitHub API

Use GitHub for:

* pull requests
* issues
* labels
* comments
* reviews
* PR metadata
* issue metadata
* GitHub-specific relationships

Do not require GitHub API access for functionality that can be obtained directly from Git.

---

## 13. Testing Requirements

Important behavior should be tested.

Prefer tests that are:

* deterministic
* fast
* isolated
* network-independent unless explicitly testing integration behavior

For Git tests, create temporary repositories using pytest temporary directories rather than depending on a real public GitHub repository.

Unit tests should not call GitHub.com.

Mock external APIs where appropriate.

Future integration tests may use external services only when explicitly requested.

Tests should cover:

* normal behavior
* boundary cases
* invalid input
* meaningful failure modes

---

## 14. Validation Commands

Before considering a task complete, run:

```bash
uv run pytest
```

Then:

```bash
uv run ruff check .
```

Then:

```bash
uv run mypy src
```

Where appropriate, also run:

```bash
uv run pytest --cov=reporecall
```

If dependencies changed, regenerate:

```bash
uv export --format requirements.txt --no-hashes -o requirements.txt
```

Do not claim tests passed unless they were actually run successfully.

If a validation command fails, explain the failure and fix it when it is within the current task scope.

---

## 15. Formatting and Linting

Use Ruff for linting.

Prefer code that passes:

```bash
uv run ruff check .
```

If formatting configuration is added, use Ruff formatting rather than introducing another formatter unless there is a clear reason.

Avoid disabling linting rules merely to silence valid warnings.

---

## 16. Type Safety

Use mypy to catch avoidable typing problems.

Avoid unnecessary use of:

```python
Any
```

when a meaningful type can be expressed.

External library boundaries may require `Any`; keep such usage localized.

Domain models should have strong types.

---

## 17. RAG Development Rules

Do not implement RAG before the repository ingestion and engineering-data model are solid.

The intended progression is:

```text
Raw repository data
        ↓
normalized engineering records
        ↓
relationships
        ↓
metadata
        ↓
retrieval documents
        ↓
chunking
        ↓
embeddings
        ↓
baseline vector retrieval
        ↓
metadata filtering
        ↓
keyword retrieval
        ↓
hybrid retrieval
        ↓
reranking
        ↓
context construction
        ↓
LLM generation
        ↓
citations
        ↓
evaluation
```

Do not jump directly from repository data to OpenAI.

---

## 18. Chunking Philosophy

When the chunking phase begins, do not blindly split every record using fixed character counts.

Chunking should respect engineering structure when practical.

Examples:

* issue title/body/comments
* PR description
* PR discussion
* commit message
* file-level diffs
* function-level diffs
* class-level code
* test changes

Future code-aware chunking may use:

* Python AST
* Tree-sitter

The chunking strategy should be measurable and documented.

---

## 19. Retrieval Philosophy

The long-term retrieval system should not rely only on vector similarity.

Software engineering data contains exact identifiers and technical terms that keyword retrieval handles well.

Examples:

* exception names
* function names
* class names
* filenames
* commit SHAs
* package names
* error codes

The intended future architecture is:

```text
User Query
    ↓
Query Understanding
    ↓
┌──────────────────┬──────────────────┐
│ Vector Retrieval │ Keyword Retrieval│
└─────────┬────────┴──────────┬───────┘
          ↓                   ↓
         Metadata Filtering
                 ↓
              Fusion
                 ↓
             Reranking
                 ↓
          Best Evidence
```

Implementation should be evidence-driven through evaluation rather than assuming a more complex retrieval method is automatically better.

---

## 20. Evaluation Philosophy

RepoRecall should eventually prove retrieval quality quantitatively.

Future evaluation metrics may include:

* Recall@K
* Precision@K
* Mean Reciprocal Rank
* NDCG
* retrieval latency
* reranking latency
* generation latency
* end-to-end latency
* answer faithfulness
* context precision
* context recall
* token usage
* cost per query

When evaluating retrieval approaches, preserve experiment results so strategies such as these can be compared:

* vector only
* BM25 only
* hybrid
* hybrid + metadata
* hybrid + reranker

Do not make unsupported claims that one retrieval strategy is superior without evaluation.

---

## 21. LLM Usage Rules

When LLM functionality is introduced:

* use the LLM after retrieval rather than as a replacement for retrieval
* provide retrieved evidence as explicit context
* instruct the model not to invent unsupported answers
* preserve source provenance
* return citations
* distinguish retrieved facts from model interpretation

The system should be able to state that insufficient historical evidence was found.

Do not force an answer when retrieval provides poor evidence.

---

## 22. Deployment Requirements

RepoRecall is intended to be deployable.

Maintain:

```text
requirements.txt
```

for deployment compatibility.

Production dependencies should not unnecessarily include development tooling.

Do not commit:

* `.env`
* API keys
* local cloned repositories
* caches
* vector indexes unless explicitly needed
* database files
* large generated artifacts

Deployment-specific configuration should come from environment variables.

---

## 23. Local Data

Local repositories should live under:

```text
data/repositories/
```

Temporary/cache data should live under:

```text
data/cache/
```

These directories should generally be ignored by Git.

The project must not accidentally commit cloned third-party repositories.

---

## 24. Documentation

Keep the README focused on:

* what RepoRecall does
* why the problem matters
* architecture
* setup
* usage
* evaluation results
* screenshots/demo
* limitations

As the system becomes more complex, use:

```text
docs/ARCHITECTURE.md
docs/ROADMAP.md
docs/DATA_MODEL.md
docs/RETRIEVAL_DESIGN.md
```

for deeper technical documentation.

When making major architectural changes, update relevant documentation.


## README Maintenance

README.md is a living technical document and must remain synchronized with
the actual implementation.

After completing work that changes RepoRecall's capabilities or architecture,
update README.md where appropriate.

Keep the README product-focused. Do not expose internal development phase
numbers or a "Current Phase" / "Project Status" section.

Update relevant sections such as:

- Overview
- Architecture
- Current capabilities
- Data model
- Project structure
- Usage examples
- Tech stack
- Testing
- Evaluation results
- Deployment instructions
- Limitations
- Roadmap

Only describe functionality as implemented when it actually exists.

Planned functionality must be clearly labeled as planned or future work.

Do not add phase-completion checklists to README.md.

Do not modify README.md for routine phases, internal models, tests, implementation
details, or coverage changes. Update it only when the public purpose, setup, public
usage, major architecture, or a major user-facing capability materially changes.

---

## 25. Current Roadmap

The intended progression is:

### Phase 0

Project setup

### Phase 1A

Configuration and local Git domain models

### Phase 1B

Local Git commit/diff ingestion

### Phase 1C

Repository cloning and updates

### Phase 2A

GitHub API client

### Phase 2B

Issue ingestion

### Phase 2C

Pull request ingestion

### Phase 2D

Comments and reviews

### Phase 3

Link issues, PRs, commits, files, and tests

### Phase 4

Normalize records into engineering events

### Phase 5

Metadata extraction

### Phase 6

Retrieval-document generation

### Phase 7

Domain-aware chunking

### Phase 8

Embeddings

### Phase 9

FAISS baseline

### Phase 10

Baseline RAG

### Phase 11

Metadata filtering

### Phase 12

BM25 keyword retrieval

### Phase 13

Hybrid retrieval

### Phase 14

Rank fusion

### Phase 15

Reranking

### Phase 16

Query understanding

### Phase 17

Relationship-aware context building

### Phase 18

Citations and provenance

### Phase 19

Retrieval evaluation

### Phase 20

RAG evaluation

### Phase 21

PostgreSQL and pgvector

### Phase 22

FastAPI backend

### Phase 23

Next.js frontend

### Phase 24

Stack-trace search

### Phase 25

AST / Tree-sitter code-aware parsing

### Phase 26

Incremental indexing

### Phase 27

GitHub webhooks

### Phase 28

Docker and CI/CD

### Phase 29

Kubernetes deployment

### Phase 30

Observability

This roadmap is directional, not permission to implement future phases.

Only implement the phase explicitly requested by the user.

---

## 26. Current Phase Guardrail

Before modifying code, determine which phase the current request belongs to.

Do not implement functionality from later phases simply because it seems helpful.

For example, during local Git ingestion, do not add:

* OpenAI
* LangChain
* embeddings
* FAISS
* PostgreSQL
* FastAPI
* frontend code
* GitHub issue ingestion

unless explicitly requested.

Prefer a working, tested implementation of the current layer over speculative future architecture.

---

## 27. Completion Format

After completing a coding task, respond with:

### What changed

Brief summary of the implementation.

### Files changed

List files created or modified.

### Design decisions

Explain the most important choices and why they were made.

### Validation

Report commands actually run and their results.

Examples:

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

### Limitations

Mention meaningful limitations of the current implementation.

### Next phase

State what the next logical phase would be, but do not implement it unless explicitly requested.

---

## 28. Core Principle

RepoRecall should become a serious information-retrieval system, not merely a "chat with GitHub" demo.

The important engineering work is:

```text
ingestion
→ data modeling
→ relationships
→ metadata
→ parsing
→ chunking
→ retrieval
→ ranking
→ context construction
→ grounding
→ evaluation
```

LLM generation is only one part of the system.

Optimize for correctness, explainability, testability, retrieval quality, and maintainability rather than maximizing framework usage.
