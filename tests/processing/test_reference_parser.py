from reporecall.github import GitHubRepository
from reporecall.processing import IssueReferenceKind, ReferenceParser


def test_parser_extracts_closing_keywords_case_insensitively():
    parser = ReferenceParser()
    repository = GitHubRepository(owner="owner", name="repo")

    references = parser.parse_issue_references(
        "Fixes #123, fix #124, FIXED #125. Closes #126, closed #127. Resolves #128, resolved #129.",
        default_repository=repository,
    )

    assert [reference.issue_number for reference in references] == [123, 124, 125, 126, 127, 128, 129]
    assert {reference.kind for reference in references} == {IssueReferenceKind.CLOSES}
    assert references[0].evidence == "Fixes #123"


def test_parser_extracts_plain_reference_phrases_without_closing_meaning():
    parser = ReferenceParser()
    repository = GitHubRepository(owner="owner", name="repo")

    references = parser.parse_issue_references(
        "See #123. Related to #124. Follow-up to #125.",
        default_repository=repository,
    )

    assert [reference.issue_number for reference in references] == [123, 124, 125]
    assert {reference.kind for reference in references} == {IssueReferenceKind.REFERENCES}


def test_parser_preserves_cross_repository_references():
    parser = ReferenceParser()
    repository = GitHubRepository(owner="owner", name="repo")

    references = parser.parse_issue_references(
        "Fixes other/project#123.",
        default_repository=repository,
    )

    assert references[0].repository == GitHubRepository(owner="other", name="project")
    assert references[0].issue_number == 123


def test_parser_avoids_unsupported_hash_number_contexts():
    parser = ReferenceParser()
    repository = GitHubRepository(owner="owner", name="repo")

    references = parser.parse_issue_references(
        "HTTP #500, version #12, and array index #5 are not issue references.",
        default_repository=repository,
    )

    assert references == []
