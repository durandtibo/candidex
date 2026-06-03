from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from candidex.author import Author, authors_to_dataframe
from candidex.columns import AUTHOR_AFFILIATION, AUTHOR_EMAIL, AUTHOR_ID, AUTHOR_NAME

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


# --- Without ID ---


def test_authors_to_dataframe_empty_sequence() -> None:
    assert_frame_equal(
        authors_to_dataframe([]),
        pl.DataFrame(
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
            }
        ),
    )


def test_authors_to_dataframe_single_author_with_all_fields() -> None:
    assert_frame_equal(
        authors_to_dataframe([make_author("Jane Smith", ["MIT"], "jane@mit.edu")]),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith"],
                AUTHOR_AFFILIATION: [["MIT"]],
                AUTHOR_EMAIL: ["jane@mit.edu"],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
            },
        ),
    )


def test_authors_to_dataframe_none_email() -> None:
    assert_frame_equal(
        authors_to_dataframe([make_author("Jane Smith", ["MIT"], None)]),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith"],
                AUTHOR_AFFILIATION: [["MIT"]],
                AUTHOR_EMAIL: [None],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
            },
        ),
    )


def test_authors_to_dataframe_none_affiliations() -> None:
    assert_frame_equal(
        authors_to_dataframe([make_author("Jane Smith", None)]),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith"],
                AUTHOR_AFFILIATION: [None],
                AUTHOR_EMAIL: [None],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
            },
        ),
    )


def test_authors_to_dataframe_empty_affiliations() -> None:
    assert_frame_equal(
        authors_to_dataframe([make_author("Jane Smith", [])]),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith"],
                AUTHOR_AFFILIATION: [[]],
                AUTHOR_EMAIL: [None],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
            },
        ),
    )


def test_authors_to_dataframe_multiple_authors() -> None:
    authors = [
        make_author("Jane Smith", ["MIT", "Stanford"], "jane@mit.edu"),
        make_author("John Doe", None, None),
        make_author("Alice Brown", [], "alice@cmu.edu"),
    ]
    assert_frame_equal(
        authors_to_dataframe(authors),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith", "John Doe", "Alice Brown"],
                AUTHOR_AFFILIATION: [["MIT", "Stanford"], None, []],
                AUTHOR_EMAIL: ["jane@mit.edu", None, "alice@cmu.edu"],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
            },
        ),
    )


def test_authors_to_dataframe_preserves_order() -> None:
    authors = [
        make_author("Charlie", ["CMU"]),
        make_author("Alice", ["Stanford"]),
        make_author("Bob", ["MIT"]),
    ]
    assert_frame_equal(
        authors_to_dataframe(authors),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Charlie", "Alice", "Bob"],
                AUTHOR_AFFILIATION: [["CMU"], ["Stanford"], ["MIT"]],
                AUTHOR_EMAIL: [None, None, None],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
            },
        ),
    )


# --- With ID ---


def test_authors_to_dataframe_exclude_id_by_default() -> None:
    assert AUTHOR_ID not in authors_to_dataframe([make_author("Jane Smith", ["MIT"])]).columns


def test_authors_to_dataframe_include_id_full_output() -> None:
    author_a = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    author_b = make_author("John Doe", ["Stanford"], None)
    assert_frame_equal(
        authors_to_dataframe([author_a, author_b], include_id=True),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith", "John Doe"],
                AUTHOR_AFFILIATION: [["MIT"], ["Stanford"]],
                AUTHOR_EMAIL: ["jane@mit.edu", None],
                AUTHOR_ID: [author_a.hash(), author_b.hash()],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
            },
        ),
    )
