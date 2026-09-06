# RepoRecall

RepoRecall is an engineering-history intelligence system for finding and
understanding past fixes in repository history. The project is being built one
phase at a time, with the early phases focused on reliable ingestion and
strongly typed engineering records before any retrieval or LLM behavior is
introduced.

## Current Status - Phase 2A

RepoRecall can currently:

- clone or reuse local Git repository copies
- optionally fetch updates from an existing clone's `origin`
- inspect local commit history
- retrieve commit metadata, including authorship, timestamps, and parent SHAs
- extract changed files from commits
- classify added, modified, deleted, and renamed files
- extract textual patches when Git provides them
- communicate with the GitHub REST API
- authenticate GitHub API requests with `GITHUB_TOKEN`
- follow GitHub REST pagination links
- detect common GitHub API failures
- expose the latest observed GitHub rate-limit metadata

The local Git ingestion flow is:

```text
RepositoryLoader
      ↓
local repository
      ↓
GitLoader
      ↓
GitCommit
      ↓
ChangedFile
```

The GitHub API client flow is:

```text
GitHubRepository
      ↓
GitHubClient
      ↓
HTTPX
      ↓
GitHub REST API
      ↓
JSON
```

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

Copy `.env.example` to `.env` for local overrides if needed. `GITHUB_TOKEN` is
optional for public GitHub API requests, but authenticated requests receive
higher GitHub rate limits and will be needed for private repositories.

## Phase 1 Usage

```python
from reporecall.ingestion import GitLoader, RepositoryLoader


repository_loader = RepositoryLoader()
repo_path = repository_loader.clone_or_get("https://github.com/fastapi/fastapi.git")

git_loader = GitLoader(repo_path)
commits = git_loader.get_commits(limit=3)

for commit in commits:
    print(commit.sha)
    print(commit.message)
    print(commit.changed_files)
```

## Phase 2A GitHub API Usage

```python
from reporecall.github import GitHubClient, GitHubRepository


repository = GitHubRepository.parse("fastapi/fastapi")

with GitHubClient() as client:
    data = client.get_repository(repository)

print(data["full_name"])
print(client.rate_limit)
```

## Planned Future Work

Later phases will add GitHub issue and pull request ingestion, comments,
reviews, relationship linking, metadata extraction, retrieval documents,
chunking, embeddings, keyword retrieval, hybrid retrieval, grounded generation,
citations, and evaluation. Those capabilities are not part of Phase 2A.
