from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from candidex.author import Author

############################
#     Tests for Author     #
############################


# --- Construction ---


def test_author_from_raw_basic() -> None:
    author = Author.from_raw("Jane Smith", ["MIT"], "jane@mit.edu")
    assert author.name == "Jane Smith"
    assert author.affiliations == ("MIT",)
    assert author.email == "jane@mit.edu"


def test_author_from_raw_none_affiliations() -> None:
    author = Author.from_raw("Jane Smith", None, None)
    assert author.affiliations is None
    assert author.email is None


def test_author_from_raw_empty_affiliations() -> None:
    author = Author.from_raw("Jane Smith", [], None)
    assert author.affiliations == ()


def test_author_from_raw_default_arguments() -> None:
    author = Author.from_raw("Jane Smith")
    assert author.affiliations is None
    assert author.email is None


def test_author_from_raw_raises_on_empty_name() -> None:
    with pytest.raises(ValueError, match="Author name cannot be empty"):
        Author.from_raw("   ")


# --- Normalization ---


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        pytest.param("  Jane Smith  ", "Jane Smith", id="strips_whitespace"),
        pytest.param("Stéphane", "Stephane", id="normalizes_unicode"),
        pytest.param("  Stéphane Geörge  ", "Stephane George", id="strips_and_normalizes"),
    ],
)
def test_author_from_raw_normalizes_name(name: str, expected: str) -> None:
    assert Author.from_raw(name).name == expected


@pytest.mark.parametrize(
    ("affiliations", "expected"),
    [
        pytest.param(["  MIT  "], ("MIT",), id="strips_whitespace"),
        pytest.param(
            ["Université de Montréal"], ("Universite de Montreal",), id="normalizes_unicode"
        ),
        pytest.param(["MIT", "Stanford"], ("MIT", "Stanford"), id="multiple_affiliations"),
    ],
)
def test_author_from_raw_normalizes_affiliations(
    affiliations: list[str],
    expected: tuple[str, ...],
) -> None:
    assert Author.from_raw("Jane Smith", affiliations).affiliations == expected


@pytest.mark.parametrize(
    ("email", "expected"),
    [
        pytest.param("  jane@mit.edu  ", "jane@mit.edu", id="strips_whitespace"),
        pytest.param("jane@mit.edu", "jane@mit.edu", id="unchanged"),
        pytest.param(None, None, id="none_unchanged"),
    ],
)
def test_author_from_raw_normalizes_email(
    email: str | None,
    expected: str | None,
) -> None:
    assert Author.from_raw("Jane Smith", email=email).email == expected


# --- Hashability and equality ---


def test_author_is_hashable() -> None:
    author = Author.from_raw("Jane Smith", ["MIT"])
    assert hash(author) is not None


def test_author_can_be_used_as_dict_key() -> None:
    author = Author.from_raw("Jane Smith", ["MIT"])
    d = {author: "value"}
    assert d[author] == "value"


def test_author_can_be_used_in_set() -> None:
    author_a = Author.from_raw("Jane Smith", ["MIT"])
    author_b = Author.from_raw("Jane Smith", ["MIT"])
    assert len({author_a, author_b}) == 1


def test_author_equality_same_fields() -> None:
    a = Author.from_raw("Jane Smith", ["MIT"], "jane@mit.edu")
    b = Author.from_raw("Jane Smith", ["MIT"], "jane@mit.edu")
    assert a == b


def test_author_inequality_different_name() -> None:
    assert Author.from_raw("Jane Smith") != Author.from_raw("John Doe")


def test_author_inequality_different_affiliations() -> None:
    assert Author.from_raw("Jane Smith", ["MIT"]) != Author.from_raw("Jane Smith", ["Stanford"])


def test_author_inequality_different_email() -> None:
    assert Author.from_raw("Jane Smith", email="jane@mit.edu") != Author.from_raw(
        "Jane Smith", email="jane@stanford.edu"
    )


# --- Immutability ---


def test_author_is_frozen() -> None:
    author = Author.from_raw("Jane Smith", ["MIT"])
    with pytest.raises(FrozenInstanceError, match=r"cannot assign to field 'name'"):
        author.name = "John Doe"  # type: ignore[misc]


# --- format_affiliations ---


@pytest.mark.parametrize(
    ("affiliations", "separator", "expected"),
    [
        pytest.param(["MIT", "Stanford"], "; ", "MIT; Stanford", id="default_separator"),
        pytest.param(["MIT", "Stanford"], " | ", "MIT | Stanford", id="custom_separator"),
        pytest.param(["MIT"], "; ", "MIT", id="single_affiliation"),
        pytest.param([], "; ", "", id="empty_affiliations"),
        pytest.param(None, "; ", "", id="none_affiliations"),
    ],
)
def test_author_format_affiliations(
    affiliations: list[str] | None,
    separator: str,
    expected: str,
) -> None:
    author = Author.from_raw("Jane Smith", affiliations)
    assert author.format_affiliations(separator=separator) == expected
