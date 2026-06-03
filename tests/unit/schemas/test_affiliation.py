from __future__ import annotations

import pytest

from candidex.author import Author
from candidex.schemas import AuthorAffiliation, PaperAffiliations

#######################################
#     Tests for AuthorAffiliation     #
#######################################


@pytest.fixture
def author() -> AuthorAffiliation:
    return AuthorAffiliation(
        author="Jane Smith",
        affiliations=["MIT CSAIL", "Stanford University"],
        email="jane@mit.edu",
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


# --- to_author ---


def test_author_affiliation_to_author_returns_author_instance(
    author: AuthorAffiliation,
) -> None:
    assert isinstance(author.to_author(), Author)


def test_author_affiliation_to_author_name(author: AuthorAffiliation) -> None:
    assert author.to_author().name == "Jane Smith"


def test_author_affiliation_to_author_affiliations(author: AuthorAffiliation) -> None:
    assert author.to_author().affiliations == ("MIT CSAIL", "Stanford University")


def test_author_affiliation_to_author_email(author: AuthorAffiliation) -> None:
    assert author.to_author().email == "jane@mit.edu"


def test_author_affiliation_to_author_none_email() -> None:
    affiliation = AuthorAffiliation(author="Jane Smith", affiliations=["MIT"], email=None)
    assert affiliation.to_author().email is None


def test_author_affiliation_to_author_empty_affiliations() -> None:
    affiliation = AuthorAffiliation(author="Jane Smith", affiliations=[], email=None)
    assert affiliation.to_author().affiliations == ()


def test_author_affiliation_to_author_normalizes_unicode() -> None:
    affiliation = AuthorAffiliation(
        author="Université de Montréal",
        affiliations=["École Polytechnique"],
        email=None,
    )
    author = affiliation.to_author()
    assert author.name == "Universite de Montreal"
    assert author.affiliations == ("Ecole Polytechnique",)


def test_author_affiliation_to_author_strips_whitespace() -> None:
    affiliation = AuthorAffiliation(
        author="  Jane Smith  ",
        affiliations=["  MIT  "],
        email="  jane@mit.edu  ",
    )
    author = affiliation.to_author()
    assert author.name == "Jane Smith"
    assert author.affiliations == ("MIT",)
    assert author.email == "jane@mit.edu"


def test_author_affiliation_to_author_consistent_with_from_raw(
    author: AuthorAffiliation,
) -> None:
    expected = Author.from_raw(
        name=author.author,
        affiliations=author.affiliations,
        email=author.email,
    )
    assert author.to_author() == expected


#######################################
#     Tests for PaperAffiliations     #
#######################################


# --- Fixtures ---


@pytest.fixture
def author_a() -> AuthorAffiliation:
    return AuthorAffiliation(
        author="Jane Smith",
        affiliations=["MIT CSAIL"],
        email="jane@mit.edu",
    )


@pytest.fixture
def author_b() -> AuthorAffiliation:
    return AuthorAffiliation(
        author="John Doe",
        affiliations=["Stanford University"],
        email=None,
    )


# --- Construction ---


def test_paper_affiliations_empty_authors() -> None:
    paper = PaperAffiliations(authors=[])
    assert paper.authors == []


def test_paper_affiliations_single_author(author_a: AuthorAffiliation) -> None:
    paper = PaperAffiliations(authors=[author_a])
    assert len(paper.authors) == 1
    assert paper.authors[0] == author_a


def test_paper_affiliations_multiple_authors(
    author_a: AuthorAffiliation, author_b: AuthorAffiliation
) -> None:
    paper = PaperAffiliations(authors=[author_a, author_b])
    assert len(paper.authors) == 2


# --- Order preservation ---


def test_paper_affiliations_preserves_author_order(
    author_a: AuthorAffiliation, author_b: AuthorAffiliation
) -> None:
    paper = PaperAffiliations(authors=[author_a, author_b])
    assert paper.authors[0] == author_a
    assert paper.authors[1] == author_b


def test_paper_affiliations_preserves_author_order_reversed(
    author_a: AuthorAffiliation, author_b: AuthorAffiliation
) -> None:
    paper = PaperAffiliations(authors=[author_b, author_a])
    assert paper.authors[0] == author_b
    assert paper.authors[1] == author_a


# --- Author fields are accessible ---


def test_paper_affiliations_author_fields_accessible(author_a: AuthorAffiliation) -> None:
    paper = PaperAffiliations(authors=[author_a])
    assert paper.authors[0].author == "Jane Smith"
    assert paper.authors[0].affiliations == ["MIT CSAIL"]
    assert paper.authors[0].email == "jane@mit.edu"


# --- Serialisation round-trip ---


def test_paper_affiliations_serialises_to_dict(
    author_a: AuthorAffiliation, author_b: AuthorAffiliation
) -> None:
    paper = PaperAffiliations(authors=[author_a, author_b])
    data = paper.model_dump()
    assert len(data["authors"]) == 2
    assert data["authors"][0]["author"] == "Jane Smith"
    assert data["authors"][1]["author"] == "John Doe"


def test_paper_affiliations_round_trip(
    author_a: AuthorAffiliation, author_b: AuthorAffiliation
) -> None:
    paper = PaperAffiliations(authors=[author_a, author_b])
    restored = PaperAffiliations.model_validate(paper.model_dump())
    assert restored == paper
