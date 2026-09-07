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
      ├────────────── Local Git
      │                   │
      │                   ▼
      │                Commits
      │                   │
      │                   ▼
      │              Changed Files
      │                   │
      │                   ▼
      │                 Diffs
      │
      └────────────── GitHub API
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
      Issues / PRs / Reviews     Relationship Evidence
             │                         │
             ▼                         ▼
      Normalized Records        Normalized Evidence
             │                         │
             ├── ReferenceParser       │
             ▼                         │
      RelationshipLinker               │
             │                         │
             └────────────┬────────────┘
                          ▼
                RelationshipEnricher
                          ▼
              Engineering Relationships
                          │
            Normalized Records + Relationships
                          ▼
             EngineeringEventAssembler
                          ▼
                Engineering Events
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
issue comments
pull request conversation comments
pull request reviews
pull request review comments
issue and pull request timeline relationship evidence
commit-associated pull request evidence
labels
authors
states
GitHub URLs
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
Commit History                    Issues / Pull Requests
      │                           Discussions / Reviews
      ▼                           PR Files / Commit References
Changed Files                    Timeline Evidence
      │                           Commit / PR Associations
      ▼                                  │
Diffs / Patches                         ▼
      │                         RepoRecall Domain Models
      └───────────────┬──────────────────┘
                      ▼
       RelationshipLinker + RelationshipEnricher
                      ▼
              Engineering Relationships
                      ▼
          EngineeringEventAssembler
                      ▼
               Engineering Events
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

# GitHub Discussion and Review Ingestion

RepoRecall can now retrieve discussion context around issues and pull requests as independent normalized records.

```text
GitHub API
   │
   ├── Issues
   │    └── Comments
   │
   └── Pull Requests
        ├── Files
        ├── Commit References
        ├── Conversation Comments
        ├── Reviews
        └── Review Comments
```

`GitHubCommentLoader` retrieves issue-style comments:

```text
/repos/{owner}/{repo}/issues/{number}/comments
```

GitHub exposes regular pull request conversation comments through the same issue-comment endpoint, so RepoRecall normalizes both issue comments and PR conversation comments into `GitHubIssueComment`.

`GitHubReviewLoader` retrieves pull request review records and line-level review comments:

```text
/repos/{owner}/{repo}/pulls/{number}/reviews
/repos/{owner}/{repo}/pulls/{number}/comments
```

These records preserve identifiers and provenance for later linking:

```text
repository
issue number
pull request number
review ID
commit SHA
original commit SHA
file path
line metadata where GitHub provides it
GitHub URL
```

Discussion and review loading is explicit. Listing issues does not fetch comments, and listing pull requests does not fetch conversation comments, reviews, or review comments.

---

# GitHub Relationship Evidence

RepoRecall can explicitly retrieve relationship evidence that GitHub records independently of local text parsing.

```text
GitHub API
    │
    ├── /issues/{number}/timeline
    │       ├── cross-referenced
    │       ├── referenced
    │       └── closed with a commit ID
    │
    └── /commits/{sha}/pulls
            └── associated pull requests
```

`GitHubRelationshipEvidenceLoader` converts these responses into normalized `GitHubTimelineRelationshipEvidence` and `GitHubCommitPullRequestAssociation` records. Unsupported timeline events are ignored, while malformed supported relationship events fail with a structured GitHub response error.

Evidence loading is always explicit. Listing issues, pull requests, or commits does not automatically request timelines or commit-associated pull requests, avoiding N+1 API behavior.

---

# Deterministic Relationship Linking

RepoRecall can connect normalized engineering-history records using deterministic evidence.

The linker operates only on records supplied by the caller:

```text
Normalized Records
       ↓
ReferenceParser
       ↓
RelationshipLinker
       ↓
EngineeringRelationship[]
```

It does not call GitHub, Git, databases, external services, embeddings, or LLMs.

GitHub-authoritative evidence follows a separate path and is combined offline:

```text
GitHubClient
      ↓
GitHubRelationshipEvidenceLoader
      ↓
Normalized GitHub Evidence

Normalized Records + Existing Relationships + GitHub Evidence
      ↓
RelationshipEnricher
      ↓
EngineeringRelationship[]
```

`RelationshipEnricher` performs no network or Git operations. It adds relationships only when their source and target artifacts exist in the supplied normalized records, and it uses exact repository and commit SHA identity.

Implemented direct relationships include:

```text
Issue → Issue Comment
Pull Request → Conversation Comment
Pull Request → Review
Review → Review Comment
Pull Request → Review Comment
Pull Request → GitHub Commit Reference
GitHub Commit Reference → Local Git Commit
Pull Request → PR Changed File
Local Git Commit → Local Changed File
Pull Request → Referenced Issue
Pull Request → Closed Issue
Local Git Commit → Referenced Issue
Local Git Commit → Closed Issue
```

GitHub evidence can directly add or confirm:

```text
Pull Request → Referenced Issue
Commit → Referenced Issue
Commit → Closed Issue
Pull Request → Associated Commit
```

Every emitted relationship includes evidence explaining why it exists, such as a parent identifier, PR commit-list membership, exact SHA match, PR file-list membership, local commit file membership, a closing keyword, or a supported text reference.

Plain references and closing references remain distinct. For example, `See #123` can produce a reference relationship, while `Fixes #123` can produce a closing relationship. Missing issue targets are skipped rather than fabricated. Cross-repository references are only linked when the referenced repository's issue record is present in the supplied input.

Multiple evidence sources are preserved rather than reduced to a generated confidence score. A PR body containing `Fixes #10` remains closing-keyword evidence, while a GitHub timeline cross-reference can independently preserve authoritative evidence that the PR referenced issue `#10`. Relationship identity includes the evidence type, so distinct provenance can coexist while duplicate evidence of the same type is collapsed deterministically.

Only direct relationships are emitted. For example, a commit closing an issue and the same commit being associated with a PR does not cause RepoRecall to infer that the PR closes the issue.

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

## `GitHubIssueComment`

Represents an issue-style GitHub comment. This model is also used for regular pull request conversation comments because GitHub exposes those through the issue-comments API.

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

## `GitHubTimelineRelationshipEvidence`

Represents normalized GitHub timeline evidence for supported `cross-referenced`, `referenced`, and commit-backed `closed` events. It preserves the target, timestamp, actor where available, structural PR detection for cross-reference sources, and exact commit identity for commit events.

## `GitHubCommitPullRequestAssociation`

Represents GitHub's authoritative association between one exact commit SHA and a pull request without duplicating the full pull request record.

## `GitHubPullRequestReview`

Represents a pull request review, including review state, reviewer identity where available, submission timestamp, commit SHA, body, and source URL.

## `GitHubPullRequestReviewComment`

Represents a line-level pull request review comment, including review ID, file path, commit identifiers, line metadata where GitHub provides it, timestamps, author, body, and source URL.

## `PullRequestState`

Represents:

```text
open
closed
```

pull request listing state. Merged pull requests are represented as closed PRs with merge metadata such as `merged_at`.

## `ReviewState`

Represents normalized pull request review states:

```text
approved
changes_requested
commented
dismissed
pending
```

## `ArtifactReference`

Represents a lightweight, stable reference to an artifact participating in relationships. It stores:

```text
artifact type
repository
deterministic identifier
```

Examples include issue number `123`, pull request number `456`, commit SHA `abc123`, PR file identifier `pr:456:file:src/app.py`, and local changed-file identifier `abc123:src/app.py`.

## `EngineeringRelationship`

Represents a direct deterministic relationship between two artifact references.

Each relationship preserves:

```text
source artifact
target artifact
relationship type
evidence type
concise evidence text where useful
source field where useful
```

Relationship direction is consistent. Examples:

```text
Issue → Issue Comment
Pull Request → Conversation Comment
Pull Request → Review
Review → Review Comment
Pull Request → GitHub Commit Reference
GitHub Commit Reference → Local Git Commit
Pull Request → PR Changed File
Local Git Commit → Local Changed File
Pull Request → Referenced or Closed Issue
Local Git Commit → Referenced or Closed Issue
```

Relationship deduplication uses source artifact, relationship type, target artifact, and evidence type as the relationship identity. Duplicate text matches that produce the same direct relationship are collapsed deterministically.

## `EngineeringEvent`

Represents one repository-local structural episode assembled from normalized records and direct relationships. It preserves its deterministic ID and anchor, issues, discussions, pull requests, reviews, changed files, GitHub and local commit representations, internal relationships, and contextual relationships.

Convenience properties expose deterministic issue numbers, pull request numbers, commit SHAs, and changed paths without adding semantic classification.

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

RepoRecall assembles normalized records into coherent structural units:

```text
EngineeringEvent
```

The assembler receives normalized records and the direct relationships already produced by the linking and enrichment layers:

```text
Normalized Records + EngineeringRelationship[]
                         ↓
             EngineeringEventAssembler
                         ↓
                EngineeringEvent[]
```

Event membership is computed as repository-local connected components over an explicit set of event-forming relationships:

```text
PULL_REQUEST_CLOSES_ISSUE
COMMIT_CLOSES_ISSUE
PULL_REQUEST_HAS_COMMIT
GITHUB_COMMIT_MATCHES_LOCAL_COMMIT
ISSUE_HAS_COMMENT
PULL_REQUEST_HAS_COMMENT
PULL_REQUEST_HAS_REVIEW
REVIEW_HAS_COMMENT
PULL_REQUEST_HAS_REVIEW_COMMENT
PULL_REQUEST_CHANGES_FILE
LOCAL_COMMIT_CHANGES_FILE
```

These directed relationships remain unchanged, but the assembler may traverse them in either direction for component membership. It does not create transitive relationships.

Weak references are contextual and do not merge components:

```text
"Fixes #123"
→ may structurally join a PR or commit with Issue #123

"See #123"
→ remains contextual and leaves independent events separate
```

Each event can contain:

```text
EngineeringEvent
├── Issues
│   └── Comments
├── Pull Requests
│   ├── Conversation Comments
│   ├── Reviews and Review Comments
│   ├── Changed Files
│   └── GitHub Commit References
├── Local Commits
│   └── Changed Files and Patches
└── Internal and Contextual Relationships
```

Core artifacts such as issues, pull requests, local commits, and GitHub commit references can form standalone events. Secondary records with no connected core artifact are skipped rather than becoming low-context standalone events. Missing relationship endpoints are ignored for membership and no records are fabricated.

Anchors use the deterministic priority pull request, issue, local commit, then GitHub commit reference. Numbered anchors choose the lowest number, and commit anchors choose the lexicographically smallest SHA. Event IDs use the repository plus anchor, for example `github.com__fastapi__fastapi__pull_request__487`.

Artifacts are ordered deterministically: issues and PRs by number; comments by timestamp and ID; reviews by submission timestamp and ID; PR files by path and stable relationship identifier; GitHub commits by SHA; local commits by committed timestamp and SHA. Relationships and final events are also sorted deterministically, and duplicate records are resolved by artifact identity rather than matching titles or messages.

An event may contain multiple issues or pull requests when strong direct relationships connect them. This is structural grouping, not a semantic claim that the event is a bug fix, feature, refactor, or any other inferred category.

Event identity currently depends on the selected anchor. If later ingestion discovers a previously absent higher-priority anchor, the deterministic event ID can change until a future persistence layer introduces durable identity.

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
│       │   ├── github_comment_loader.py
│       │   ├── github_issue_loader.py
│       │   ├── github_pull_request_loader.py
│       │   ├── github_relationship_evidence_loader.py
│       │   └── github_review_loader.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── events.py
│       │   ├── github_evidence.py
│       │   ├── records.py
│       │   └── relationships.py
│       │
│       ├── processing/
│       │   ├── __init__.py
│       │   ├── event_assembler.py
│       │   ├── reference_parser.py
│       │   ├── relationship_enricher.py
│       │   └── relationship_linker.py
│       │
│       └── utils/
│
├── tests/
│   ├── github/
│   ├── ingestion/
│   ├── models/
│   └── processing/
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

## Retrieve GitHub Comments and Reviews

```python
from reporecall.github import GitHubClient, GitHubRepository
from reporecall.ingestion import GitHubCommentLoader, GitHubReviewLoader

repository = GitHubRepository.parse("fastapi/fastapi")

with GitHubClient() as client:
    comments = GitHubCommentLoader(client, repository)
    reviews = GitHubReviewLoader(client, repository)

    issue_comments = comments.get_issue_comments(123)
    pr_comments = comments.get_pull_request_comments(456)
    pr_reviews = reviews.get_pull_request_reviews(456)
    review_comments = reviews.get_pull_request_review_comments(456)

print([comment.id for comment in issue_comments])
print([comment.id for comment in pr_comments])
print([review.state for review in pr_reviews])
print([comment.path for comment in review_comments])
```

---

## Link Normalized Records

```python
from reporecall.processing import RelationshipInput, RelationshipLinker

records = RelationshipInput(
    repository=repository,
    issues=issues,
    pull_requests=pull_requests,
    pull_request_commits={123: commits},
    local_commits=local_commits,
)

relationships = RelationshipLinker().link(records)

for relationship in relationships:
    print(
        relationship.source,
        relationship.relationship_type,
        relationship.target,
        relationship.evidence_type,
    )
```

---

## Enrich Relationships with GitHub Evidence

```python
from reporecall.github import GitHubClient
from reporecall.ingestion import GitHubRelationshipEvidenceLoader
from reporecall.processing import RelationshipEnricher

with GitHubClient() as client:
    evidence_loader = GitHubRelationshipEvidenceLoader(client, repository)
    timeline_evidence = evidence_loader.get_timeline_relationship_evidence(10)
    commit_pr_associations = evidence_loader.get_pull_requests_for_commit("abc123")

enriched_relationships = RelationshipEnricher().enrich(
    relationships,
    records=records,
    timeline_evidence=timeline_evidence,
    commit_pr_associations=commit_pr_associations,
)
```

The caller chooses which issue, pull request, or commit evidence to load. This keeps API cost explicit and allows the same normalized evidence to be processed deterministically offline.

---

## Assemble Engineering Events

```python
from reporecall.processing import EngineeringEventAssembler, EngineeringEventInput

event_input = EngineeringEventInput(
    **records.model_dump(),
    relationships=enriched_relationships,
)

events = EngineeringEventAssembler().assemble(event_input)

for event in events:
    print(event.event_id)
    print(event.pull_request_numbers)
    print(event.issue_numbers)
    print(event.commit_shas)
```

Event assembly is offline and deterministic. It organizes existing evidence without producing summaries, metadata inference, retrieval documents, or chunks.

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
* explicit issue and pull request conversation comment ingestion
* explicit pull request review and review-comment ingestion
* deterministic relationship linking over normalized records
* explicit GitHub timeline and commit-associated pull request evidence ingestion
* offline relationship enrichment with preserved evidence provenance
* deterministic repository-local engineering event assembly

Planned capabilities include:

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
