# RepoRecall

**RepoRecall is an engineering-history intelligence system that helps developers discover whether a bug or technical problem has happened before, understand how it was previously fixed, and trace the evidence back through issues, pull requests, commits, code changes, and tests.**

Instead of treating a Git repository as only source code, RepoRecall treats its history as a searchable engineering knowledge base.

```text
"What happened before?"
        ↓
Repository History
        ↓
Issues + PRs + Commits + Diffs + Tests
        ↓
Metadata-Aware Retrieval
        ↓
Historical Fixes
        ↓
Evidence + Explanation + Sources
```
---

# Why RepoRecall?

Software teams accumulate enormous amounts of engineering knowledge over time:

* bug reports
* GitHub issues
* pull requests
* commit messages
* code reviews
* code diffs
* test changes
* release notes
* technical discussions

Much of this knowledge becomes difficult to discover later.

A developer investigating a database connection leak might search documentation or the current codebase while an almost identical problem was solved two years earlier in:

```text
Issue #184
      ↓
PR #206
      ↓
Commit 7f82a31
      ↓
src/database/session.py
      ↓
tests/test_session.py
```

RepoRecall aims to make that historical engineering knowledge searchable.

A future RepoRecall query might look like:

```text
Have we fixed a SQLAlchemy connection leak after worker failures before?
```

Instead of returning a generic LLM answer, RepoRecall should eventually return:

```text
Similar Historical Fix

PR #842 — Fix session leak after worker failure

Root cause:
Database sessions were not closed when execution exited
through an exception.

Fix:
Session cleanup was moved into a finally block.

Changed files:
- src/database/session.py
- src/workers/retry.py

Tests:
- tests/test_session_cleanup.py

Evidence:
- Issue #801
- PR #842
- Commit a81fc93
```

The objective is not simply to build a "chat with GitHub" application.

The objective is to build a real **engineering information-retrieval system**.

---

# Core Product Idea

RepoRecall will eventually answer questions such as:

```text
Have we fixed a similar authentication timeout before?
```

```text
Find previous database connection leaks.
```

```text
Show bug fixes that modified session.py.
```

```text
Find Python bugs from the last two years where tests were added.
```

```text
How did we previously handle Redis connection failures?
```

```text
Does this stack trace resemble a previously fixed issue?
```

```text
Find past race-condition fixes involving the authentication service.
```

The key idea is that RepoRecall will search both:

```text
CONTENT
+
METADATA
```

rather than relying exclusively on semantic similarity.

---

# Current Architecture

The implemented architecture currently looks like this:

```text
                         GitHub Repository
                                │
               ┌────────────────┴────────────────┐
               │                                 │
               ▼                                 ▼
        RepositoryLoader                   GitHubClient
               │                                 │
               ▼                                 ▼
      Local Git Repository               GitHub REST API
               │                      ┌──────────┴──────────┐
               ▼                      │                     │
           GitLoader                   ▼                     ▼
               │              GitHubIssueLoader   GitHubPullRequestLoader
               ▼                      │                     │
          GitCommit                    ▼           ┌─────────┼─────────┐
               │                  GitHubIssue      ▼         ▼         ▼
               │                         GitHubPullRequest PR Files Commit Refs
               ▼
         ChangedFile
               │
               ▼
          Diff / Patch
```

RepoRecall currently has two intentionally separate ingestion paths:

### Local Git

Used for repository-native information such as:

```text
commits
parent SHAs
changed files
diffs
authors
commit timestamps
```

### GitHub REST API

Used for GitHub-specific engineering context such as:

```text
issues
pull requests
pull request files
pull request commit references
labels
authors
states
GitHub URLs
```

Future capabilities will add:

```text
comments
reviews
relationships
```

These two paths remain separate because Git and GitHub provide different types of information.

---

# Current Data Flow

The implemented ingestion flow currently produces normalized records from local Git and GitHub API sources:

```text
Git Repository
      │
      ├───────────────────────────────┐
      │                               │
      ▼                               ▼
Local Git                         GitHub API
      │                               │
      ▼                               ▼
Commit History                    Issues
      │                           Pull Requests
      ▼                           PR Files
Changed Files                    Commit References
      │                           Labels
      ▼                           Authors
Diffs / Patches                  Timestamps
      │                           URLs
      └───────────────┬───────────────┘
                      │
                      ▼
             RepoRecall Domain Models
```

No embeddings, vector database, or LLM generation are required at this stage.

That is intentional.

---

# Current Capabilities

## Repository Management

RepoRecall can clone a Git repository into its local repository directory.

```text
Repository URL
      ↓
RepositoryLoader
      ↓
data/repositories/
```

Already cloned repositories are reused rather than cloned again.

Optional update behavior fetches remote refs without performing destructive resets, merges, or checkouts.

---

## Local Commit History

`GitLoader` reads commit history from an existing repository.

Conceptually:

```python
loader = GitLoader(repo_path)

commits = loader.get_commits(limit=10)
```

Each commit is normalized into a strongly typed:

```text
GitCommit
```

containing information such as:

```text
SHA
message
author
author email
authored timestamp
committed timestamp
parent SHAs
changed files
```

---

## Changed Files

Each commit contains structured `ChangedFile` records.

RepoRecall currently recognizes:

```text
added
modified
deleted
renamed
```

Changed-file information includes:

```text
path
old path where applicable
change type
additions
deletions
patch
```

---

## Root Commit Handling

Root commits have no parent.

RepoRecall compares root commits against Git's empty tree so files introduced by the initial commit are correctly represented as:

```text
added
```

rather than producing an empty diff.

---

## Merge Commit Handling

For the current implementation, merge commits are compared against their:

```text
first parent
```

This produces deterministic behavior while avoiding premature complex merge analysis.

More advanced merge reasoning can be added later if it proves useful for retrieval quality.

---

## Repository Cloning

`RepositoryLoader` handles:

```text
clone
reuse
fetch
```

without mixing repository lifecycle responsibilities into `GitLoader`.

Architecture:

```text
RepositoryLoader
      ↓
local Path
      ↓
GitLoader
```

---

# GitHub API Client

RepoRecall includes a reusable GitHub REST API layer.

```text
GitHubRepository
       ↓
GitHubClient
       ↓
HTTPX
       ↓
GitHub REST API
```

The client currently provides:

* standard GitHub request headers
* optional authentication
* generic GET requests
* repository lookup
* GitHub pagination
* rate-limit metadata
* structured GitHub exceptions
* HTTP/network error handling
* JSON response validation
* context-manager lifecycle support

Example:

```python
from reporecall.github import GitHubClient, GitHubRepository

repository = GitHubRepository.parse("fastapi/fastapi")

with GitHubClient() as client:
    data = client.get_repository(repository)

print(data["full_name"])
```

---

# Repository Identifier Parsing

RepoRecall can normalize common GitHub repository identifiers.

Examples:

```text
fastapi/fastapi
```

```text
https://github.com/fastapi/fastapi
```

```text
https://github.com/fastapi/fastapi.git
```

Simple GitHub SSH syntax is also supported where implemented:

```text
git@github.com:fastapi/fastapi.git
```

These become:

```python
GitHubRepository(
    owner="fastapi",
    name="fastapi",
)
```

Repository identity is preserved explicitly because RepoRecall will eventually search across multiple repositories.

---

# GitHub Pagination

GitHub API endpoints commonly split large result sets across pages.

RepoRecall handles pagination centrally inside `GitHubClient`.

```text
Page 1
  ↓
Link: next
  ↓
Page 2
  ↓
Link: next
  ↓
Page 3
  ↓
No next link
  ↓
Complete Result
```

Pagination follows GitHub `Link` headers instead of forcing future ingestion modules to implement their own pagination loops.

This is important because repositories may contain thousands of:

```text
issues
pull requests
comments
reviews
```

---

# GitHub Rate Limits

RepoRecall captures rate-limit information returned by GitHub.

This allows callers to inspect information such as:

```text
limit
remaining
used
reset time
```

The current implementation does not automatically sleep or retry when rate limited.

Instead, RepoRecall raises a clear rate-limit exception and exposes available metadata.

Automatic retry/backoff behavior may be introduced later if needed.

---

# GitHub Error Handling

The GitHub layer exposes focused exceptions for situations such as:

```text
authentication failure
resource not found
rate limiting
network failure
invalid response
unexpected API failure
```

HTTP implementation details remain inside the GitHub layer.

Future ingestion modules therefore do not need to repeatedly implement:

```text
HTTP status handling
JSON validation
authentication
pagination
network exception conversion
```

---

# GitHub Issue Ingestion

RepoRecall includes structured GitHub issue ingestion.

Current flow:

```text
GitHub Repository
       ↓
GitHubIssueLoader
       ↓
GitHubClient
       ↓
/repos/{owner}/{repo}/issues
       ↓
Raw GitHub JSON
       ↓
Validation / Normalization
       ↓
GitHubIssue
```

Each issue preserves structured metadata useful for future retrieval.

Examples include:

```text
repository
issue number
title
body
state
author
labels
created timestamp
updated timestamp
closed timestamp
comment count
locked state
source URL
```

---

# Issues vs Pull Requests

GitHub's repository issues endpoint may return both:

```text
issues
and
pull requests
```

RepoRecall explicitly distinguishes them using GitHub's response structure.

Pull-request entries are not incorrectly normalized into `GitHubIssue` objects.

This distinction becomes particularly important later when RepoRecall constructs relationships such as:

```text
Issue
   ↓
Pull Request
   ↓
Commit
```

---

# GitHub Pull Request Ingestion

RepoRecall includes structured GitHub pull request ingestion.

Current flow:

```text
GitHub Repository
       ↓
GitHubPullRequestLoader
       ↓
GitHubClient
       ↓
/repos/{owner}/{repo}/pulls
       ↓
Raw GitHub JSON
       ↓
Validation / Normalization
       ↓
GitHubPullRequest
```

Each pull request preserves metadata useful for later relationship linking and retrieval.

Examples include:

```text
repository
pull request number
title
body
state
draft flag
merge timestamp
merge commit SHA
author
labels
head branch
base branch
change counts
source URL
```

Changed files and commit references are retrieved explicitly:

```text
/repos/{owner}/{repo}/pulls/{number}/files
/repos/{owner}/{repo}/pulls/{number}/commits
```

Listing pull requests does not automatically fetch files or commits for every PR. That avoids N+1 API behavior and lets later ingestion stages decide when to enrich records.

---

# Domain Models

RepoRecall uses Pydantic models at external-data boundaries.

Current important models include:

## `GitCommit`

Represents normalized local Git commit data.

## `ChangedFile`

Represents a file affected by a commit.

## `FileChangeType`

Represents:

```text
added
modified
deleted
renamed
```

## `GitHubRepository`

Represents:

```text
owner/repository
```

identity.

## `GitHubUser`

Represents minimal GitHub user information needed by RepoRecall.

## `GitHubIssueLabel`

Preserves structured GitHub labels.

## `GitHubIssue`

Represents normalized GitHub issue information.

## `IssueState`

Represents:

```text
open
closed
```

issue state.

## `GitHubPullRequest`

Represents normalized GitHub pull request metadata.

## `GitHubBranchReference`

Represents source and target branch refs for a pull request.

Fork pull requests can preserve a different head repository from the base repository.

## `GitHubPullRequestFile`

Represents a file changed by a pull request.

## `GitHubPullRequestFileStatus`

Represents GitHub PR-file statuses such as:

```text
added
modified
deleted
renamed
copied
changed
unchanged
```

GitHub's `removed` status is normalized to `deleted`.

## `GitHubCommitReference`

Represents a lightweight commit reference returned by GitHub's pull request commits endpoint.

This is intentionally separate from `GitCommit`, which represents richer local Git commit data with changed files and patches.

## `PullRequestState`

Represents:

```text
open
closed
```

pull request listing state. Merged pull requests are represented as closed PRs with merge metadata such as `merged_at`.

Domain models intentionally do not contain:

```text
embedding
similarity score
LLM summary
root cause
vector ID
```

Those concepts belong to later layers.

---

# Why Structured Metadata Matters

RepoRecall is intentionally designed around searchable metadata.

A basic RAG system might store:

```text
Fix database connection leak
```

RepoRecall should eventually understand:

```json
{
  "repository": "example/backend",
  "type": "bug_fix",
  "language": "python",
  "component": "database",
  "files_changed": [
    "src/database/session.py",
    "tests/test_session.py"
  ],
  "tests_changed": true,
  "issue_number": 801,
  "pull_request": 842,
  "commit_sha": "a81fc93"
}
```

This allows queries such as:

```text
Find database-related Python bugs fixed in the last
two years that modified a test file.
```

That query requires more than embeddings.

It combines:

```text
semantic retrieval
+
metadata filtering
+
keyword retrieval
```

This is a core design principle of RepoRecall.

---

# Long-Term Architecture

The intended mature architecture is:

```text
                        GitHub Repository
                               │
            ┌──────────────────┴──────────────────┐
            │                                     │
            ▼                                     ▼
          Git                                  GitHub API
            │                                     │
            ▼                                     ▼
         Commits                              Issues / PRs
         Diffs                                Comments
         Files                                Reviews
         Tests                                Labels
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
                         Normalization
                               │
                               ▼
                     Relationship Linking
                               │
                               ▼
                      Engineering Events
                               │
                               ▼
                      Metadata Extraction
                               │
                               ▼
                  Retrieval Document Builder
                               │
                               ▼
                      Domain-Aware Chunking
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
      Dense Embeddings                    Keyword Index
              │                                 │
              ▼                                 ▼
       Vector Retrieval                    BM25 Search
              │                                 │
              └────────────────┬────────────────┘
                               ▼
                       Metadata Filtering
                               │
                               ▼
                         Rank Fusion
                               │
                               ▼
                           Reranker
                               │
                               ▼
                      Context Expansion
                               │
                               ▼
                           OpenAI
                               │
                               ▼
                    Grounded Final Answer
                               │
                               ▼
                Historical Fixes + Citations
```

---

# Engineering Event Model

One of RepoRecall's most important future concepts is the:

```text
EngineeringEvent
```

Instead of treating historical artifacts as unrelated documents:

```text
Issue

PR

Commit

Diff
```

RepoRecall will connect them:

```text
Issue #801
    │
    ▼
PR #842
    │
    ▼
Commit a81fc93
    │
    ▼
src/database/session.py
    │
    ▼
tests/test_session.py
```

This represents one historical engineering event:

```text
Problem
   ↓
Discussion
   ↓
Solution
   ↓
Implementation
   ↓
Verification
```

That connected history is much more useful than isolated vector chunks.

---

# Planned Retrieval Architecture

RepoRecall will eventually support multiple retrieval strategies.

## Dense Retrieval

Semantic embeddings will help retrieve conceptually similar problems.

Example:

```text
connection pool exhausted
```

may retrieve:

```text
database sessions remain open after failed workers
```

even when the wording differs.

---

## Keyword Retrieval

Software contains many exact identifiers where semantic search alone is insufficient.

Examples:

```text
QueuePool
NullPointerException
session.py
get_database_session
HTTP 401
RedisConnectionError
```

RepoRecall will therefore introduce BM25-style keyword retrieval.

---

## Metadata Filtering

Structured metadata will allow constraints such as:

```text
language = python

repository = backend-api

tests_changed = true

date >= 2025-01-01

component = authentication
```

---

## Hybrid Retrieval

The intended architecture is:

```text
Query
  ↓
Query Understanding
  ↓
┌─────────────────┬─────────────────┐
│ Dense Retrieval │ Keyword Search  │
└────────┬────────┴────────┬────────┘
         │                 │
         └────────┬────────┘
                  ↓
           Metadata Filtering
                  ↓
              Rank Fusion
                  ↓
               Reranker
                  ↓
          Historical Evidence
```

---

# Reranking

Initial retrieval may produce many candidates.

Example:

```text
Vector Search + BM25
          ↓
     50 candidates
          ↓
       Reranker
          ↓
    Best 5–10 results
```

This allows RepoRecall to optimize initial retrieval for recall while using a more precise model to determine final ranking.

---

# Context Expansion

RepoRecall should not simply concatenate retrieved chunks.

If retrieval finds:

```text
Commit a81fc93
```

the system should eventually follow known relationships and recover:

```text
associated issue
associated pull request
relevant diff
changed files
tests
discussion
```

Result:

```text
Retrieved Commit
       ↓
Relationship Lookup
       ↓
Issue + PR + Diff + Tests
       ↓
Complete Historical Context
```

This richer context will then be provided to the language model.

---

# Grounded Generation

The LLM should not act as the knowledge source.

The retrieval pipeline provides evidence first.

```text
Repository History
       ↓
Retrieval
       ↓
Evidence
       ↓
LLM
       ↓
Explanation
```

If there is insufficient evidence, RepoRecall should be able to respond accordingly rather than fabricate a historical fix.

A future answer should contain direct provenance such as:

```text
Issue #801
PR #842
Commit a81fc93
src/database/session.py
```

---

# Planned Stack Trace Search

A later capability will allow developers to paste errors directly.

Example:

```text
sqlalchemy.exc.TimeoutError:
QueuePool limit of size 5 overflow 10 reached
```

RepoRecall will extract useful identifiers:

```text
SQLAlchemy
TimeoutError
QueuePool
```

and combine:

```text
semantic retrieval
+
keyword retrieval
+
historical metadata
```

to locate related fixes.

---

# Planned Code-Aware Parsing

Fixed character chunks are often poor representations of source-code changes.

RepoRecall will eventually support structure-aware parsing.

For Python:

```text
AST
```

For multiple languages:

```text
Tree-sitter
```

Potential chunk boundaries:

```text
function
method
class
file
test
diff hunk
```

Example metadata:

```json
{
  "file": "src/database/session.py",
  "function": "get_session",
  "language": "python",
  "change_type": "modified"
}
```

---

# Evaluation

Retrieval quality will be measured rather than assumed.

A future evaluation dataset will contain questions and known relevant historical artifacts.

Example:

```json
{
  "query": "Find previous database connection leak fixes",
  "relevant_prs": [842, 900]
}
```

RepoRecall will evaluate metrics such as:

```text
Recall@K
Precision@K
MRR
NDCG
```

It will also track operational metrics such as:

```text
retrieval latency
reranking latency
generation latency
total latency
token usage
query cost
```

---

# Retrieval Experiments

The project will compare retrieval configurations such as:

| Retrieval Strategy           | Recall@5 | MRR | NDCG | Latency |
| ---------------------------- | -------: | --: | ---: | ------: |
| Vector only                  |      TBD | TBD |  TBD |     TBD |
| BM25 only                    |      TBD | TBD |  TBD |     TBD |
| Hybrid                       |      TBD | TBD |  TBD |     TBD |
| Hybrid + metadata            |      TBD | TBD |  TBD |     TBD |
| Hybrid + metadata + reranker |      TBD | TBD |  TBD |     TBD |

Results will be populated once evaluation capabilities are implemented.

The purpose is to determine which architecture actually improves retrieval rather than assuming additional complexity is automatically better.

---

# Tech Stack

## Current

```text
Python 3.12
uv
Pydantic
Pydantic Settings
GitPython
HTTPX
pytest
pytest-cov
Ruff
mypy
```

## Planned

```text
GitHub REST API
Sentence Transformers / embedding model
FAISS
BM25
cross-encoder reranking
OpenAI
PostgreSQL
pgvector
FastAPI
Next.js
Docker
Kubernetes
```

Technologies are introduced only when the implemented architecture requires them.

---

# Project Structure

The repository currently grows incrementally toward:

```text
reporecall/
│
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── requirements.txt
├── .python-version
├── .env.example
├── .gitignore
│
├── src/
│   └── reporecall/
│       ├── __init__.py
│       ├── config.py
│       │
│       ├── github/
│       │   ├── __init__.py
│       │   ├── client.py
│       │   ├── exceptions.py
│       │   └── models.py
│       │
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── git_loader.py
│       │   ├── repository_loader.py
│       │   ├── github_issue_loader.py
│       │   └── github_pull_request_loader.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   └── records.py
│       │
│       └── utils/
│
├── tests/
│   ├── ingestion/
│   └── github/
│
├── data/
│   ├── repositories/
│   └── cache/
│
├── docs/
│
└── scripts/
```

Future directories will only be added when their corresponding capabilities are implemented.

Potential future modules include:

```text
processing/
documents/
embeddings/
storage/
retrieval/
generation/
evaluation/
api/
```

---

# Installation

## Requirements

* Python 3.12+
* Git
* `uv`

Install `uv` if necessary using the official installation instructions for your platform.

Clone RepoRecall:

```bash
git clone <reporecall-repository-url>
cd reporecall
```

Create/synchronize the environment:

```bash
uv sync
```

---

# Environment Configuration

Create your local environment file:

```bash
cp .env.example .env
```

Example:

```env
GITHUB_TOKEN=
GITHUB_API_URL=https://api.github.com

REPO_DATA_DIR=data/repositories
CACHE_DIR=data/cache
```

Never commit `.env`.

For public repositories, GitHub API access can work without authentication, subject to lower rate limits.

Authenticated access is recommended for larger ingestion workloads.

---

# Dependency Management

RepoRecall uses:

```text
pyproject.toml
+
uv.lock
```

as its primary dependency system.

Add production dependencies with:

```bash
uv add <package>
```

Add development dependencies with:

```bash
uv add --dev <package>
```

`requirements.txt` is maintained for deployment compatibility.

After production dependencies change:

```bash
uv export --format requirements.txt --output-file requirements.txt
```

Do not manually maintain dependency versions in `requirements.txt`.

---

# Basic Usage

## Clone a Repository

```python
from reporecall.ingestion import RepositoryLoader

loader = RepositoryLoader()

repo_path = loader.clone_or_get(
    "https://github.com/fastapi/fastapi.git"
)

print(repo_path)
```

---

## Inspect Commit History

```python
from reporecall.ingestion import GitLoader

loader = GitLoader(repo_path)

commits = loader.get_commits(limit=5)

for commit in commits:
    print(commit.sha)
    print(commit.message)

    for changed_file in commit.changed_files:
        print(
            changed_file.change_type,
            changed_file.path,
            changed_file.additions,
            changed_file.deletions,
        )
```

---

## Query GitHub Repository Metadata

```python
from reporecall.github import GitHubClient, GitHubRepository

repository = GitHubRepository.parse(
    "https://github.com/fastapi/fastapi"
)

with GitHubClient() as client:
    data = client.get_repository(repository)

print(data["full_name"])
print(client.rate_limit)
```

---

## Retrieve GitHub Issues

```python
from reporecall.github import GitHubClient, GitHubRepository
from reporecall.ingestion import GitHubIssueLoader
from reporecall.models import IssueState

repository = GitHubRepository.parse("fastapi/fastapi")

with GitHubClient() as client:
    loader = GitHubIssueLoader(client, repository)

    issues = loader.get_issues(
        state=IssueState.OPEN,
        limit=10,
    )

for issue in issues:
    print(issue.number, issue.title)
    print([label.name for label in issue.labels])
```

---

## Retrieve GitHub Pull Requests

```python
from reporecall.github import GitHubClient, GitHubRepository
from reporecall.ingestion import GitHubPullRequestLoader
from reporecall.models import PullRequestState

repository = GitHubRepository.parse("fastapi/fastapi")

with GitHubClient() as client:
    loader = GitHubPullRequestLoader(client, repository)

    pull_requests = loader.get_pull_requests(
        state=PullRequestState.CLOSED,
        limit=10,
    )

    pull_request = loader.get_pull_request(123)
    files = loader.get_pull_request_files(123)
    commits = loader.get_pull_request_commits(123)

print(pull_request.number, pull_request.title, pull_request.is_merged)
print([file.filename for file in files])
print([commit.sha for commit in commits])
```

---

# Testing

Run the complete test suite:

```bash
uv run pytest
```

Run with coverage:

```bash
uv run pytest --cov=reporecall
```

Run linting:

```bash
uv run ruff check .
```

Run static type checking:

```bash
uv run mypy src
```

Current project validation is expected to keep:

```text
pytest     passing
Ruff       passing
mypy       passing
coverage   high
```

Exact test and coverage counts may change as new capabilities are implemented.

---

# Testing Philosophy

Automated tests should not depend on external GitHub availability.

Local Git behavior is tested using temporary repositories created during pytest execution.

GitHub HTTP behavior is tested with mocked HTTPX transports.

This allows the core test suite to remain:

```text
deterministic
fast
network-independent
repeatable
```

Manual smoke tests against real public repositories may be performed separately.

---

# Data and Security

Do not commit:

```text
.env
API keys
GitHub tokens
cloned repositories
cache data
generated indexes
local databases
private repository content
```

Cloned repositories are stored under:

```text
data/repositories/
```

Cache data is stored under:

```text
data/cache/
```

Both are intended to remain outside version control.

---

# Roadmap

RepoRecall is being built incrementally. Implemented capabilities include:

* project setup with Python, `uv`, linting, typing, and tests
* environment-backed configuration
* local Git repository cloning and reuse
* local commit-history ingestion
* changed-file and patch extraction
* GitHub REST API access
* GitHub pagination and rate-limit metadata
* normalized GitHub issue ingestion
* normalized GitHub pull request ingestion
* explicit pull request changed-file and commit-reference retrieval

Planned capabilities include:

* issue, pull request, review, and comment ingestion
* relationship linking between issues, pull requests, commits, files, and tests
* normalized engineering events
* deterministic metadata extraction
* retrieval document generation
* domain-aware chunking
* embeddings and vector retrieval
* keyword retrieval
* hybrid retrieval and rank fusion
* reranking
* relationship-aware context expansion
* grounded generation with citations
* retrieval and RAG evaluation
* persistent storage
* API and frontend surfaces
* deployment and observability

---

# Design Principles

RepoRecall follows several core principles.

## 1. Retrieval Before Generation

The LLM should explain retrieved evidence, not replace repository search.

## 2. Metadata Is First-Class

Repository metadata should be searchable alongside text.

## 3. Preserve Relationships

Issue, PR, commit, diff, and test relationships carry valuable engineering meaning.

## 4. Preserve Provenance

Answers should eventually show exactly where evidence came from.

## 5. Measure Retrieval Quality

Complex retrieval architectures must demonstrate measurable improvements.

## 6. Prefer Explicit Architecture

Important retrieval and ingestion logic should remain understandable instead of being hidden behind large convenience frameworks.

## 7. Build Incrementally

Each milestone should be independently understandable, testable, and explainable.

---

# What RepoRecall Is Not

RepoRecall is not intended to be merely:

```text
GitHub data
    ↓
chunk text
    ↓
FAISS
    ↓
GPT
```

The larger goal is:

```text
Raw Engineering History
        ↓
Data Modeling
        ↓
Relationship Extraction
        ↓
Metadata
        ↓
Domain-Aware Parsing
        ↓
Hybrid Retrieval
        ↓
Ranking
        ↓
Context Engineering
        ↓
Grounded Generation
        ↓
Evaluation
```

---

# Project Goal

By the end of the project, a developer should be able to connect a repository and ask:

```text
Have we seen this problem before?
```

RepoRecall should then search the engineering history, identify the strongest historical matches, reconstruct how those problems were solved, and return an evidence-backed answer.

The finished system should demonstrate practical experience with:

```text
Python backend engineering
Git internals
GitHub APIs
data ingestion
data modeling
RAG
information retrieval
embeddings
vector search
BM25
hybrid retrieval
reranking
metadata filtering
LLM integration
evaluation
FastAPI
PostgreSQL
pgvector
Docker
Kubernetes
observability
```

while solving a problem that real software-engineering teams face:

> **Important engineering knowledge already exists — but finding it again is hard.**
