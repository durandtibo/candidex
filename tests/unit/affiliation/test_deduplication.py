from __future__ import annotations

import pytest

from candidex.affiliation import deduplicate_authors
from candidex.schemas import AuthorAffiliation


def make_author(name: str, affiliations: list[str]) -> AuthorAffiliation:
    return AuthorAffiliation(author=name, affiliations=affiliations, email=None)


#########################################
#     Tests for deduplicate_authors     #
#########################################


@pytest.mark.parametrize(
    ("authors", "expected"),
    [
        pytest.param([], [], id="empty_list"),
        pytest.param(
            [make_author("Jane Smith", ["MIT"])],
            [make_author("Jane Smith", ["MIT"])],
            id="single_author",
        ),
        pytest.param(
            [make_author("Jane Smith", ["MIT"]), make_author("John Doe", ["Stanford"])],
            [make_author("Jane Smith", ["MIT"]), make_author("John Doe", ["Stanford"])],
            id="no_duplicates",
        ),
        pytest.param(
            [make_author("Jane Smith", ["MIT"]), make_author("Jane Smith", ["MIT"])],
            [make_author("Jane Smith", ["MIT"])],
            id="exact_duplicate_removed",
        ),
        pytest.param(
            [make_author("Jane Smith", ["MIT"]), make_author("Jane Smith", ["MIT CSAIL"])],
            [make_author("Jane Smith", ["MIT"]), make_author("Jane Smith", ["MIT CSAIL"])],
            id="same_name_different_affiliation_kept",
        ),
        pytest.param(
            [
                AuthorAffiliation(author="Jane Smith", affiliations=["MIT"], email="jane@mit.edu"),
                AuthorAffiliation(author="Jane Smith", affiliations=["MIT"], email=None),
            ],
            [
                AuthorAffiliation(author="Jane Smith", affiliations=["MIT"], email="jane@mit.edu"),
                AuthorAffiliation(author="Jane Smith", affiliations=["MIT"], email=None),
            ],
            id="same_name_and_affiliation_different_email_kept",
        ),
        pytest.param(
            [
                AuthorAffiliation(author="Jane Smith", affiliations=["MIT"], email="jane@mit.edu"),
                AuthorAffiliation(author="Jane Smith", affiliations=["MIT"], email="jane@mit.edu"),
            ],
            [
                AuthorAffiliation(author="Jane Smith", affiliations=["MIT"], email="jane@mit.edu"),
            ],
            id="exact_duplicate_with_email_removed",
        ),
    ],
)
def test_deduplicate_authors(
    authors: list[AuthorAffiliation],
    expected: list[AuthorAffiliation],
) -> None:
    assert deduplicate_authors(authors) == expected


def test_deduplicate_authors_preserves_first_occurrence() -> None:
    first = make_author("Jane Smith", ["MIT"])
    second = make_author("Jane Smith", ["MIT"])
    result = deduplicate_authors([first, second])
    assert result == [first]
    assert result[0] is first


def test_deduplicate_authors_preserves_order() -> None:
    authors = [
        make_author("Charlie", ["MIT"]),
        make_author("Alice", ["Stanford"]),
        make_author("Bob", ["CMU"]),
    ]
    assert deduplicate_authors(authors) == authors
