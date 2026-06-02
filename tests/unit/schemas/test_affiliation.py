from __future__ import annotations

import pytest

from candidex.schemas import AuthorAffiliation

#######################################
#     Tests for AuthorAffiliation     #
#######################################


@pytest.fixture
def author() -> AuthorAffiliation:
    return AuthorAffiliation(
        author="Jane Smith",
        affiliations=["MIT CSAIL", "Stanford University"],
        email=None,
    )


@pytest.mark.parametrize(
    (
        "author",
        "affiliations",
        "email",
        "expected_author",
        "expected_affiliations",
        "expected_email",
    ),
    [
        pytest.param(
            "  Jane Smith  ",
            ["  MIT CSAIL  ", "  Stanford University  "],
            "  jane@mit.edu  ",
            "Jane Smith",
            ["MIT CSAIL", "Stanford University"],
            "jane@mit.edu",
            id="leading_trailing_spaces",
        ),
        pytest.param(
            "\tJane Smith\n",
            ["\tMIT CSAIL\n", "\tStanford University\n"],
            "\tjane@mit.edu\n",
            "Jane Smith",
            ["MIT CSAIL", "Stanford University"],
            "jane@mit.edu",
            id="tabs_and_newlines",
        ),
        pytest.param(
            "Jane Smith",
            ["MIT CSAIL", "Stanford University"],
            "jane@mit.edu",
            "Jane Smith",
            ["MIT CSAIL", "Stanford University"],
            "jane@mit.edu",
            id="no_whitespace_unchanged",
        ),
        pytest.param(
            "  Jane Smith  ",
            ["  MIT CSAIL  "],
            None,
            "Jane Smith",
            ["MIT CSAIL"],
            None,
            id="none_email_unchanged",
        ),
    ],
)
def test_author_affiliation_strips_strings(
    author: str,
    affiliations: list[str],
    email: str | None,
    expected_author: str,
    expected_affiliations: list[str],
    expected_email: str | None,
) -> None:
    assert AuthorAffiliation(
        author=author, affiliations=affiliations, email=email
    ) == AuthorAffiliation(
        author=expected_author, affiliations=expected_affiliations, email=expected_email
    )


@pytest.mark.parametrize(
    ("separator", "expected"),
    [
        pytest.param("; ", "MIT CSAIL; Stanford University", id="default_separator"),
        pytest.param(" | ", "MIT CSAIL | Stanford University", id="pipe_separator"),
        pytest.param(", ", "MIT CSAIL, Stanford University", id="comma_separator"),
        pytest.param("", "MIT CSAILStanford University", id="empty_separator"),
    ],
)
def test_format_affiliations_author_affiliation(
    separator: str,
    expected: str,
    author: AuthorAffiliation,
) -> None:
    assert author.format_affiliations(separator=separator) == expected


def test_format_affiliations_author_affiliation_empty_affiliations() -> None:
    author = AuthorAffiliation(author="Jane Smith", affiliations=[], email=None)
    assert author.format_affiliations() == ""


def test_format_affiliations_author_affiliation_single_affiliation() -> None:
    author = AuthorAffiliation(author="Jane Smith", affiliations=["MIT CSAIL"], email=None)
    assert author.format_affiliations() == "MIT CSAIL"
