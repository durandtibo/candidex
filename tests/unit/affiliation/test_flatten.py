r"""Tests for candidex.affiliation.flatten."""

from __future__ import annotations

import pytest

from candidex.affiliation.flatten import flatten_authors
from candidex.schemas import AuthorAffiliation, PaperAffiliations


def make_author(name: str, affiliations: list[str]) -> AuthorAffiliation:
    return AuthorAffiliation(author=name, affiliations=affiliations, email=None)


def make_paper(authors: list[AuthorAffiliation]) -> PaperAffiliations:
    return PaperAffiliations(authors=authors)


#####################################
#     Tests for flatten_authors     #
#####################################


@pytest.mark.parametrize(
    ("papers", "expected_names"),
    [
        pytest.param([], [], id="empty_sequence"),
        pytest.param([make_paper([])], [], id="single_paper_no_authors"),
        pytest.param(
            [make_paper([make_author("Jane Smith", ["MIT"])])],
            ["Jane Smith"],
            id="single_paper_single_author",
        ),
        pytest.param(
            [
                make_paper(
                    [make_author("Jane Smith", ["MIT"]), make_author("John Doe", ["Stanford"])]
                )
            ],
            ["Jane Smith", "John Doe"],
            id="single_paper_multiple_authors",
        ),
        pytest.param(
            [
                make_paper([make_author("Jane Smith", ["MIT"])]),
                make_paper([make_author("John Doe", ["Stanford"])]),
            ],
            ["Jane Smith", "John Doe"],
            id="multiple_papers",
        ),
        pytest.param(
            [
                make_paper(
                    [make_author("Jane Smith", ["MIT"]), make_author("John Doe", ["Stanford"])]
                ),
                make_paper([make_author("Alice Brown", ["CMU"])]),
            ],
            ["Jane Smith", "John Doe", "Alice Brown"],
            id="multiple_papers_multiple_authors",
        ),
        pytest.param(
            [make_paper([]), make_paper([make_author("Jane Smith", ["MIT"])])],
            ["Jane Smith"],
            id="skips_empty_papers",
        ),
    ],
)
def test_flatten_authors(
    papers: list[PaperAffiliations],
    expected_names: list[str],
) -> None:
    result = flatten_authors(papers)
    assert [a.author for a in result] == expected_names


def test_flatten_authors_preserves_author_objects() -> None:
    author = make_author("Jane Smith", ["MIT"])
    result = flatten_authors([make_paper([author])])
    assert result == [author]
