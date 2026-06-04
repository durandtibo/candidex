from __future__ import annotations

import polars as pl
from openreview import Profile
from polars.testing import assert_frame_equal

from candidex.author import (
    Author,
    add_author_column,
    add_openreview_profile_ids_to_dataframe,
    add_openreview_profiles_to_dataframe,
    authors_to_dataframe,
)
from candidex.columns import (
    AUTHOR_AFFILIATION,
    AUTHOR_EMAIL,
    AUTHOR_ID,
    AUTHOR_NAME,
    AUTHOR_OPENREVIEW_PROFILE,
    AUTHOR_OPENREVIEW_PROFILE_ID,
)
from candidex.openreview import serialize_profiles


def make_author(
    name: str,
    affiliations: list[str] | None = None,
    email: str | None = None,
) -> Author:
    return Author.from_raw(name, affiliations, email)


def make_frame(authors: list[Author]) -> pl.DataFrame:
    return authors_to_dataframe(authors, include_id=True)


def make_profile(profile_id: str, fullname: str, position: str, institution: str) -> Profile:
    return Profile(
        id=profile_id,
        content={
            "names": [{"fullname": fullname}],
            "history": [{"position": position, "institution": {"name": institution}}],
        },
    )


#######################################
#     Tests for add_author_column     #
#######################################


def test_add_author_column_single_string_value() -> None:
    author = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    assert_frame_equal(
        add_author_column(make_frame([author]), {author: "value_a"}, "my_column", pl.String),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith"],
                AUTHOR_AFFILIATION: [["MIT"]],
                AUTHOR_EMAIL: ["jane@mit.edu"],
                AUTHOR_ID: [author.hash()],
                "my_column": ["value_a"],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                "my_column": pl.String,
            },
        ),
    )


def test_add_author_column_list_value() -> None:
    author = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    assert_frame_equal(
        add_author_column(
            make_frame([author]), {author: ["a", "b"]}, "my_column", pl.List(pl.String)
        ),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith"],
                AUTHOR_AFFILIATION: [["MIT"]],
                AUTHOR_EMAIL: ["jane@mit.edu"],
                AUTHOR_ID: [author.hash()],
                "my_column": [["a", "b"]],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                "my_column": pl.List(pl.String),
            },
        ),
    )


def test_add_author_column_none_value_is_null() -> None:
    author = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    assert_frame_equal(
        add_author_column(make_frame([author]), {author: None}, "my_column", pl.List(pl.String)),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith"],
                AUTHOR_AFFILIATION: [["MIT"]],
                AUTHOR_EMAIL: ["jane@mit.edu"],
                AUTHOR_ID: [author.hash()],
                "my_column": [None],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                "my_column": pl.List(pl.String),
            },
        ),
    )


def test_add_author_column_author_not_in_data_is_null() -> None:
    author_a = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    author_b = make_author("John Doe", ["Stanford"], None)
    assert_frame_equal(
        add_author_column(
            make_frame([author_a, author_b]),
            {author_a: ["x"]},
            "my_column",
            pl.List(pl.String),
        ),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith", "John Doe"],
                AUTHOR_AFFILIATION: [["MIT"], ["Stanford"]],
                AUTHOR_EMAIL: ["jane@mit.edu", None],
                AUTHOR_ID: [author_a.hash(), author_b.hash()],
                "my_column": [["x"], None],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                "my_column": pl.List(pl.String),
            },
        ),
    )


def test_add_author_column_empty_data_all_null() -> None:
    author = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    assert_frame_equal(
        add_author_column(make_frame([author]), {}, "my_column", pl.List(pl.String)),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith"],
                AUTHOR_AFFILIATION: [["MIT"]],
                AUTHOR_EMAIL: ["jane@mit.edu"],
                AUTHOR_ID: [author.hash()],
                "my_column": [None],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                "my_column": pl.List(pl.String),
            },
        ),
    )


def test_add_author_column_multiple_authors() -> None:
    author_a = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    author_b = make_author("John Doe", ["Stanford"], None)
    assert_frame_equal(
        add_author_column(
            make_frame([author_a, author_b]),
            {author_a: ["x"], author_b: ["y"]},
            "my_column",
            pl.List(pl.String),
        ),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith", "John Doe"],
                AUTHOR_AFFILIATION: [["MIT"], ["Stanford"]],
                AUTHOR_EMAIL: ["jane@mit.edu", None],
                AUTHOR_ID: [author_a.hash(), author_b.hash()],
                "my_column": [["x"], ["y"]],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                "my_column": pl.List(pl.String),
            },
        ),
    )


def test_add_author_column_preserves_row_order() -> None:
    author_a = make_author("Charlie", ["CMU"])
    author_b = make_author("Alice", ["Stanford"])
    author_c = make_author("Bob", ["MIT"])
    assert_frame_equal(
        add_author_column(
            make_frame([author_a, author_b, author_c]),
            {author_a: "x", author_b: "y", author_c: "z"},
            "my_column",
            pl.String,
        ),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Charlie", "Alice", "Bob"],
                AUTHOR_AFFILIATION: [["CMU"], ["Stanford"], ["MIT"]],
                AUTHOR_EMAIL: [None, None, None],
                AUTHOR_ID: [author_a.hash(), author_b.hash(), author_c.hash()],
                "my_column": ["x", "y", "z"],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                "my_column": pl.String,
            },
        ),
    )


#############################################################
#     Tests for add_openreview_profile_ids_to_dataframe     #
#############################################################


def test_add_openreview_profile_ids_single_author() -> None:
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


def test_add_openreview_profile_ids_multiple_ids() -> None:
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


def test_add_openreview_profile_ids_none_is_null() -> None:
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


def test_add_openreview_profile_ids_missing_author_is_null() -> None:
    author_a = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    author_b = make_author("John Doe", ["Stanford"], None)
    assert_frame_equal(
        add_openreview_profile_ids_to_dataframe(
            make_frame([author_a, author_b]), {author_a: ["~Jane_Smith1"]}
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


def test_add_openreview_profile_ids_empty_mapping() -> None:
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


##########################################################
#     Tests for add_openreview_profiles_to_dataframe     #
##########################################################


def test_add_openreview_profiles_none_is_null() -> None:
    author = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    assert_frame_equal(
        add_openreview_profiles_to_dataframe(make_frame([author]), {author: None}),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith"],
                AUTHOR_AFFILIATION: [["MIT"]],
                AUTHOR_EMAIL: ["jane@mit.edu"],
                AUTHOR_ID: [author.hash()],
                AUTHOR_OPENREVIEW_PROFILE: [None],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                AUTHOR_OPENREVIEW_PROFILE: pl.List(pl.String),
            },
        ),
    )


def test_add_openreview_profiles_empty_mapping_is_null() -> None:
    author = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    assert_frame_equal(
        add_openreview_profiles_to_dataframe(make_frame([author]), {}),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith"],
                AUTHOR_AFFILIATION: [["MIT"]],
                AUTHOR_EMAIL: ["jane@mit.edu"],
                AUTHOR_ID: [author.hash()],
                AUTHOR_OPENREVIEW_PROFILE: [None],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                AUTHOR_OPENREVIEW_PROFILE: pl.List(pl.String),
            },
        ),
    )


def test_add_openreview_profiles_single_profile() -> None:
    author = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    profile = make_profile("~Jane_Smith1", "Jane Smith", "PhD Student", "MIT")
    assert_frame_equal(
        add_openreview_profiles_to_dataframe(make_frame([author]), {author: [profile]}),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith"],
                AUTHOR_AFFILIATION: [["MIT"]],
                AUTHOR_EMAIL: ["jane@mit.edu"],
                AUTHOR_ID: [author.hash()],
                AUTHOR_OPENREVIEW_PROFILE: [serialize_profiles([profile])],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                AUTHOR_OPENREVIEW_PROFILE: pl.List(pl.String),
            },
        ),
    )


def test_add_openreview_profiles_multiple_profiles_per_author() -> None:
    author = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    profile_a = make_profile("~Jane_Smith1", "Jane Smith", "PhD Student", "MIT")
    profile_b = make_profile("~Jane_Smith2", "Jane Smith", "Postdoc", "Stanford")
    assert_frame_equal(
        add_openreview_profiles_to_dataframe(
            make_frame([author]), {author: [profile_a, profile_b]}
        ),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith"],
                AUTHOR_AFFILIATION: [["MIT"]],
                AUTHOR_EMAIL: ["jane@mit.edu"],
                AUTHOR_ID: [author.hash()],
                AUTHOR_OPENREVIEW_PROFILE: [serialize_profiles([profile_a, profile_b])],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                AUTHOR_OPENREVIEW_PROFILE: pl.List(pl.String),
            },
        ),
    )


def test_add_openreview_profiles_multiple_authors() -> None:
    author_a = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    author_b = make_author("John Doe", ["Stanford"], None)
    profile_a = make_profile("~Jane_Smith1", "Jane Smith", "PhD Student", "MIT")
    profile_b = make_profile("~John_Doe1", "John Doe", "Professor", "Stanford")
    assert_frame_equal(
        add_openreview_profiles_to_dataframe(
            make_frame([author_a, author_b]),
            {author_a: [profile_a], author_b: [profile_b]},
        ),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith", "John Doe"],
                AUTHOR_AFFILIATION: [["MIT"], ["Stanford"]],
                AUTHOR_EMAIL: ["jane@mit.edu", None],
                AUTHOR_ID: [author_a.hash(), author_b.hash()],
                AUTHOR_OPENREVIEW_PROFILE: [
                    serialize_profiles([profile_a]),
                    serialize_profiles([profile_b]),
                ],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                AUTHOR_OPENREVIEW_PROFILE: pl.List(pl.String),
            },
        ),
    )


def test_add_openreview_profiles_mixed_none_and_profiles() -> None:
    author_a = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    author_b = make_author("John Doe", ["Stanford"], None)
    profile_a = make_profile("~Jane_Smith1", "Jane Smith", "PhD Student", "MIT")
    assert_frame_equal(
        add_openreview_profiles_to_dataframe(
            make_frame([author_a, author_b]),
            {author_a: [profile_a], author_b: None},
        ),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Jane Smith", "John Doe"],
                AUTHOR_AFFILIATION: [["MIT"], ["Stanford"]],
                AUTHOR_EMAIL: ["jane@mit.edu", None],
                AUTHOR_ID: [author_a.hash(), author_b.hash()],
                AUTHOR_OPENREVIEW_PROFILE: [serialize_profiles([profile_a]), None],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                AUTHOR_OPENREVIEW_PROFILE: pl.List(pl.String),
            },
        ),
    )


def test_add_openreview_profiles_preserves_row_order() -> None:
    author_a = make_author("Charlie", ["CMU"])
    author_b = make_author("Alice", ["Stanford"])
    author_c = make_author("Bob", ["MIT"])
    profile_a = make_profile("~Charlie1", "Charlie", "PhD Student", "CMU")
    profile_b = make_profile("~Alice1", "Alice", "Professor", "Stanford")
    profile_c = make_profile("~Bob1", "Bob", "Postdoc", "MIT")
    assert_frame_equal(
        add_openreview_profiles_to_dataframe(
            make_frame([author_a, author_b, author_c]),
            {author_a: [profile_a], author_b: [profile_b], author_c: [profile_c]},
        ),
        pl.DataFrame(
            {
                AUTHOR_NAME: ["Charlie", "Alice", "Bob"],
                AUTHOR_AFFILIATION: [["CMU"], ["Stanford"], ["MIT"]],
                AUTHOR_EMAIL: [None, None, None],
                AUTHOR_ID: [author_a.hash(), author_b.hash(), author_c.hash()],
                AUTHOR_OPENREVIEW_PROFILE: [
                    serialize_profiles([profile_a]),
                    serialize_profiles([profile_b]),
                    serialize_profiles([profile_c]),
                ],
            },
            schema={
                AUTHOR_NAME: pl.String,
                AUTHOR_AFFILIATION: pl.List(pl.String),
                AUTHOR_EMAIL: pl.String,
                AUTHOR_ID: pl.String,
                AUTHOR_OPENREVIEW_PROFILE: pl.List(pl.String),
            },
        ),
    )
