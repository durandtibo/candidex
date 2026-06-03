from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from candidex.author import Author, authors_to_dataframe

# --- Helpers ---


def make_author(
    name: str,
    affiliations: list[str] | None = None,
    email: str | None = None,
) -> Author:
    return Author.from_raw(name, affiliations, email)


##########################################
#     Tests for authors_to_dataframe     #
##########################################


def test_authors_to_dataframe_empty_sequence() -> None:
    expected = pl.DataFrame(
        schema={
            "name": pl.String,
            "affiliations": pl.List(pl.String),
            "email": pl.String,
        }
    )
    assert_frame_equal(authors_to_dataframe([]), expected)


def test_authors_to_dataframe_single_author_with_all_fields() -> None:
    authors = [make_author("Jane Smith", ["MIT"], "jane@mit.edu")]
    expected = pl.DataFrame(
        {"name": ["Jane Smith"], "affiliations": [["MIT"]], "email": ["jane@mit.edu"]},
        schema={"name": pl.String, "affiliations": pl.List(pl.String), "email": pl.String},
    )
    assert_frame_equal(authors_to_dataframe(authors), expected)


def test_authors_to_dataframe_none_email() -> None:
    authors = [make_author("Jane Smith", ["MIT"], None)]
    expected = pl.DataFrame(
        {"name": ["Jane Smith"], "affiliations": [["MIT"]], "email": [None]},
        schema={"name": pl.String, "affiliations": pl.List(pl.String), "email": pl.String},
    )
    assert_frame_equal(authors_to_dataframe(authors), expected)


def test_authors_to_dataframe_none_affiliations() -> None:
    authors = [make_author("Jane Smith", None)]
    expected = pl.DataFrame(
        {"name": ["Jane Smith"], "affiliations": [None], "email": [None]},
        schema={"name": pl.String, "affiliations": pl.List(pl.String), "email": pl.String},
    )
    assert_frame_equal(authors_to_dataframe(authors), expected)


def test_authors_to_dataframe_empty_affiliations() -> None:
    authors = [make_author("Jane Smith", [])]
    expected = pl.DataFrame(
        {"name": ["Jane Smith"], "affiliations": [[]], "email": [None]},
        schema={"name": pl.String, "affiliations": pl.List(pl.String), "email": pl.String},
    )
    assert_frame_equal(authors_to_dataframe(authors), expected)


def test_authors_to_dataframe_multiple_affiliations() -> None:
    authors = [make_author("Jane Smith", ["MIT", "Stanford", "CMU"])]
    expected = pl.DataFrame(
        {"name": ["Jane Smith"], "affiliations": [["MIT", "Stanford", "CMU"]], "email": [None]},
        schema={"name": pl.String, "affiliations": pl.List(pl.String), "email": pl.String},
    )
    assert_frame_equal(authors_to_dataframe(authors), expected)


def test_authors_to_dataframe_multiple_authors() -> None:
    authors = [
        make_author("Jane Smith", ["MIT", "Stanford"], "jane@mit.edu"),
        make_author("John Doe", None, None),
        make_author("Alice Brown", [], "alice@cmu.edu"),
    ]
    expected = pl.DataFrame(
        {
            "name": ["Jane Smith", "John Doe", "Alice Brown"],
            "affiliations": [["MIT", "Stanford"], None, []],
            "email": ["jane@mit.edu", None, "alice@cmu.edu"],
        },
        schema={"name": pl.String, "affiliations": pl.List(pl.String), "email": pl.String},
    )
    assert_frame_equal(authors_to_dataframe(authors), expected)


def test_authors_to_dataframe_preserves_order() -> None:
    authors = [
        make_author("Charlie", ["CMU"]),
        make_author("Alice", ["Stanford"]),
        make_author("Bob", ["MIT"]),
    ]
    expected = pl.DataFrame(
        {
            "name": ["Charlie", "Alice", "Bob"],
            "affiliations": [["CMU"], ["Stanford"], ["MIT"]],
            "email": [None, None, None],
        },
        schema={"name": pl.String, "affiliations": pl.List(pl.String), "email": pl.String},
    )
    assert_frame_equal(authors_to_dataframe(authors), expected)
