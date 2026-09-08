# RepoRecall

**RepoRecall is an engineering-history retrieval system for finding how similar software problems were previously investigated and resolved.**

Engineering knowledge is often scattered across Git commits, diffs, issues, pull requests, reviews, comments, and tests. RepoRecall connects these artifacts into structured engineering history so developers can eventually search past problems and retrieve the fixes, discussions, code changes, and evidence behind them.

## What It Does

RepoRecall currently ingests local Git history and GitHub repository data, normalizes related engineering artifacts, links direct relationships, groups connected records into engineering events, extracts deterministic metadata, and renders source-aware retrieval documents.

It is being designed to answer questions such as:

> Have we seen this problem before?

> What caused a similar failure in the past?

> Which files were changed to fix it?

> Were tests added or modified?

> What issue, pull request, review, or commit explains the change?

Search and answer generation are planned. The current implementation stops at deterministic retrieval-document generation.

## Architecture

The implemented pipeline is:

```text
Local Git + GitHub API
          ↓
 Normalized Records
          +
 Normalized GitHub Evidence
          ↓
 Relationship Linking
   and Enrichment
          ↓
Engineering Relationships
          ↓
  Engineering Events
          ↓
 Deterministic Metadata
          ↓
  Retrieval Documents
```

The planned retrieval path builds on those documents:

```text
Retrieval Documents
         ↓
Domain-Aware Chunks
         ↓
Vector + Keyword Search
         ↓
Hybrid Retrieval + Reranking
         ↓
Grounded Answer Generation
         ↓
Answers with Source Evidence
```

This separation keeps repository understanding deterministic and testable instead of treating raw GitHub text as an undifferentiated document collection.

## Core Design

RepoRecall models engineering history across:

* Git commits and diffs
* changed files and textual patches
* GitHub issues and comments
* pull requests and changed files
* pull request commit references
* reviews and review comments
* GitHub-authoritative relationship evidence
* deterministic engineering relationships
* engineering events
* searchable deterministic metadata
* source-aware retrieval documents

Relationships preserve evidence provenance, allowing RepoRecall to distinguish structural links, exact commit matches, GitHub-provided evidence, closing keywords, and contextual text references.

Network access remains in ingestion. Relationship linking, event assembly, metadata extraction, and retrieval-document generation operate offline over normalized models.

## Engineering Events

Related repository artifacts are grouped into `EngineeringEvent` records representing coherent structural episodes in engineering history.

```text
EngineeringEvent
├── Issues
│   └── Comments
├── Pull Requests
│   ├── Conversation Comments
│   ├── Reviews and Review Comments
│   └── Changed Files
├── Commits
│   └── Changed Files and Patches
├── Direct Relationships
└── Contextual Relationships
```

Strong direct relationships can connect artifacts into the same event. Weak references remain contextual so unrelated repository history does not collapse into large groups. Event assembly does not create transitive conclusions or semantic classifications.

## Metadata

`EventMetadataExtractor` derives immutable metadata directly from each engineering event, including:

* issue and pull request numbers
* commit SHAs and changed paths
* authors and participants
* labels and milestones
* languages, extensions, and directories
* additions, deletions, and changed-line totals
* test, documentation, configuration, and dependency paths

GitHub users and local Git authors remain separate identities. Languages and path categories come from explicit deterministic mappings rather than semantic inference.

## Retrieval Documents

`RetrievalDocumentBuilder` transforms one `EngineeringEvent` and its matching `EventMetadata` into one immutable `RetrievalDocument`.

Each document preserves:

* deterministic document and event IDs
* repository identity
* a source-derived title
* typed sections for each engineering domain
* canonical human-readable text
* deterministic metadata
* artifact identities and source URLs

Sections can represent issues, comments, pull requests, reviews, commits, changed files, patches, direct relationships, and contextual relationships. Empty domains are omitted.

Original issue bodies, discussions, stack traces, commit messages, technical identifiers, and patches remain source-faithful. Exact local and GitHub representations are deduplicated for rendering without discarding their separate provenance.

```text
Pull Request #20 [pull_request_closes_issue] Issue #10
Evidence:
- closing_keyword: Fixes #10
- github_timeline_closed_event
```

Multiple evidence sources are grouped for readability without changing the underlying relationship records.

## Retrieval Design

The planned retrieval system will combine multiple signals rather than relying on vector similarity alone:

```text
Dense Retrieval
      +
Keyword / BM25 Retrieval
      +
Metadata Filtering
      ↓
 Hybrid Retrieval
      ↓
 Rank Fusion
      ↓
 Cross-Encoder Reranking
      ↓
 Relationship-Aware Context
```

This is intended to support semantic and exact search across repository identity, language, file paths, labels, authors, directories, tests, exception names, symbols, and commit SHAs. These retrieval components are not implemented yet.

## Tech Stack

Current:

* **Language:** Python 3.12
* **Package management:** uv
* **Models and validation:** Pydantic
* **Configuration:** Pydantic Settings
* **Git:** GitPython
* **GitHub API:** HTTPX and the GitHub REST API
* **Testing:** pytest and pytest-cov
* **Quality:** Ruff and mypy

Planned technologies include vector search, BM25, PostgreSQL with pgvector, FastAPI, Docker, and Kubernetes. They will be added only when the corresponding capability is implemented.

## Setup

Clone the repository:

```bash
git clone https://github.com/mo-abdulai/reporecall.git
cd reporecall
```

Install Python 3.12 and synchronize dependencies:

```bash
uv python install 3.12
uv sync
```

Create a local environment file:

```bash
cp .env.example .env
```

Add a GitHub token when authenticated GitHub API access is needed:

```env
GITHUB_TOKEN=your_github_token
```

Never commit the `.env` file or credentials.

## Usage

After normalized records and relationships have been assembled into events, extract metadata and build a retrieval document explicitly:

```python
from reporecall.processing import EventMetadataExtractor, RetrievalDocumentBuilder

event = events[0]
metadata = EventMetadataExtractor().extract(event)
document = RetrievalDocumentBuilder().build(event, metadata)

print(document.document_id)
print(document.title)
print(document.text)

for source in document.sources:
    print(source.label, source.url)
```

Both processing steps are deterministic, offline, and non-mutating.

## Testing

Run the test suite:

```bash
uv run pytest
```

Run all quality checks:

```bash
uv run ruff check .
uv run mypy src
uv run pytest --cov=reporecall
```

Automated tests are deterministic and network-independent. GitHub API behavior is tested with HTTPX mock transports.

## Project Structure

```text
src/reporecall/
├── github/        # GitHub API client, models, and API errors
├── ingestion/     # Local Git and GitHub data ingestion
├── models/        # Normalized records, relationships, events, and documents
├── processing/    # Linking, enrichment, assembly, metadata, and rendering
└── utils/         # Shared utilities as they become necessary
```

Retrieval, generation, storage, API, and evaluation packages will be introduced only when their capabilities are implemented.

## Design Principles

* Preserve source provenance.
* Keep deterministic processing separate from LLM inference.
* Treat metadata as searchable data.
* Avoid implicit API enrichment and N+1 requests.
* Keep retrieval independently measurable.
* Prefer evidence-backed relationships over heuristics.
* Preserve GitHub and local Git identities without inventing matches.
* Represent direct evidence without adding transitive conclusions.
* Use LLMs after retrieval, not as a substitute for retrieval.

## Current Limitations

RepoRecall does not yet provide:

* document chunking
* embeddings or vector indexing
* keyword or hybrid retrieval
* reranking
* persistence
* API or frontend surfaces
* LLM answer generation
* citations in generated answers

The current implementation produces deterministic, source-aware retrieval documents that these later layers can consume.

## Example Goal

A future RepoRecall query may look like:

```text
Our FastAPI service starts leaking database connections
after a retry worker crashes.

Have we fixed anything like this before?
```

RepoRecall is being designed to retrieve the most relevant historical engineering events and surface the associated issue, pull request, discussion, code changes, tests, commits, and source evidence.
