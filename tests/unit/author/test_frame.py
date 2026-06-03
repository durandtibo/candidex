from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from candidex.author import (
    Author,
    add_openreview_profile_ids_to_dataframe,
    authors_to_dataframe,
)
from candidex.columns import (
    AUTHOR_AFFILIATION,
    AUTHOR_EMAIL,
    AUTHOR_ID,
    AUTHOR_NAME,
    AUTHOR_OPENREVIEW_PROFILE_ID,
)


def make_author(
    name: str,
    affiliations: list[str] | None = None,
    email: str | None = None,
) -> Author:
    return Author.from_raw(name, affiliations, email)


def make_frame(authors: list[Author]) -> pl.DataFrame:
    return authors_to_dataframe(authors, include_id=True)


#############################################################
#     Tests for add_openreview_profile_ids_to_dataframe     #
#############################################################


def test_add_openreview_profile_ids_to_dataframe_single_profile_id() -> None:
    author = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    assert_frame_equal(
        add_openreview_profile_ids_to_dataframe(make_frame([author]), {author: ["~Jane_Smith1"]}),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith"],
                AUTHOR_AFFILIATION: [["MIT"]],
                AUTHOR_EMAIL: ["jane@mit.edu"],
                AUTHOR_ID: [author.hash()],
                AUTHOR_OPENREVIEW_PROFILE_ID: [["~Jane_Smith1"]],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                AUTHOR_OPENREVIEW_PROFILE_ID: pl.List(pl.String),
            },
        ),
    )


def test_add_openreview_profile_ids_to_dataframe_multiple_profile_ids() -> None:
    author = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    assert_frame_equal(
        add_openreview_profile_ids_to_dataframe(
            make_frame([author]), {author: ["~Jane_Smith1", "~Jane_Smith2"]}
        ),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith"],
                AUTHOR_AFFILIATION: [["MIT"]],
                AUTHOR_EMAIL: ["jane@mit.edu"],
                AUTHOR_ID: [author.hash()],
                AUTHOR_OPENREVIEW_PROFILE_ID: [["~Jane_Smith1", "~Jane_Smith2"]],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                AUTHOR_OPENREVIEW_PROFILE_ID: pl.List(pl.String),
            },
        ),
    )


def test_add_openreview_profile_ids_to_dataframe_empty_profile_ids() -> None:
    author = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    assert_frame_equal(
        add_openreview_profile_ids_to_dataframe(make_frame([author]), {author: []}),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith"],
                AUTHOR_AFFILIATION: [["MIT"]],
                AUTHOR_EMAIL: ["jane@mit.edu"],
                AUTHOR_ID: [author.hash()],
                AUTHOR_OPENREVIEW_PROFILE_ID: [[]],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                AUTHOR_OPENREVIEW_PROFILE_ID: pl.List(pl.String),
            },
        ),
    )


def test_add_openreview_profile_ids_to_dataframe_none_profile_ids_is_null() -> None:
    author = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    assert_frame_equal(
        add_openreview_profile_ids_to_dataframe(make_frame([author]), {author: None}),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith"],
                AUTHOR_AFFILIATION: [["MIT"]],
                AUTHOR_EMAIL: ["jane@mit.edu"],
                AUTHOR_ID: [author.hash()],
                AUTHOR_OPENREVIEW_PROFILE_ID: [None],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                AUTHOR_OPENREVIEW_PROFILE_ID: pl.List(pl.String),
            },
        ),
    )


def test_add_openreview_profile_ids_to_dataframe_multiple_authors() -> None:
    author_a = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    author_b = make_author("John Doe", ["Stanford"], None)
    assert_frame_equal(
        add_openreview_profile_ids_to_dataframe(
            make_frame([author_a, author_b]),
            {author_a: ["~Jane_Smith1"], author_b: ["~John_Doe1"]},
        ),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith", "John Doe"],
                AUTHOR_AFFILIATION: [["MIT"], ["Stanford"]],
                AUTHOR_EMAIL: ["jane@mit.edu", None],
                AUTHOR_ID: [author_a.hash(), author_b.hash()],
                AUTHOR_OPENREVIEW_PROFILE_ID: [["~Jane_Smith1"], ["~John_Doe1"]],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                AUTHOR_OPENREVIEW_PROFILE_ID: pl.List(pl.String),
            },
        ),
    )


def test_add_openreview_profile_ids_to_dataframe_preserves_row_order() -> None:
    author_a = make_author("Charlie", ["CMU"])
    author_b = make_author("Alice", ["Stanford"])
    author_c = make_author("Bob", ["MIT"])
    assert_frame_equal(
        add_openreview_profile_ids_to_dataframe(
            make_frame([author_a, author_b, author_c]),
            {author_a: ["~Charlie1"], author_b: ["~Alice1"], author_c: ["~Bob1"]},
        ),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Charlie", "Alice", "Bob"],
                AUTHOR_AFFILIATION: [["CMU"], ["Stanford"], ["MIT"]],
                AUTHOR_EMAIL: [None, None, None],
                AUTHOR_ID: [author_a.hash(), author_b.hash(), author_c.hash()],
                AUTHOR_OPENREVIEW_PROFILE_ID: [["~Charlie1"], ["~Alice1"], ["~Bob1"]],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                AUTHOR_OPENREVIEW_PROFILE_ID: pl.List(pl.String),
            },
        ),
    )


def test_add_openreview_profile_ids_to_dataframe_author_not_in_mapping_is_null() -> None:
    author_a = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    author_b = make_author("John Doe", ["Stanford"], None)
    assert_frame_equal(
        add_openreview_profile_ids_to_dataframe(
            make_frame([author_a, author_b]),
            {author_a: ["~Jane_Smith1"]},
        ),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith", "John Doe"],
                AUTHOR_AFFILIATION: [["MIT"], ["Stanford"]],
                AUTHOR_EMAIL: ["jane@mit.edu", None],
                AUTHOR_ID: [author_a.hash(), author_b.hash()],
                AUTHOR_OPENREVIEW_PROFILE_ID: [["~Jane_Smith1"], None],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                AUTHOR_OPENREVIEW_PROFILE_ID: pl.List(pl.String),
            },
        ),
    )


def test_add_openreview_profile_ids_to_dataframe_empty_mapping() -> None:
    author = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    assert_frame_equal(
        add_openreview_profile_ids_to_dataframe(make_frame([author]), {}),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith"],
                AUTHOR_AFFILIATION: [["MIT"]],
                AUTHOR_EMAIL: ["jane@mit.edu"],
                AUTHOR_ID: [author.hash()],
                AUTHOR_OPENREVIEW_PROFILE_ID: [None],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                AUTHOR_OPENREVIEW_PROFILE_ID: pl.List(pl.String),
            },
        ),
    )
