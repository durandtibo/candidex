r"""Contain DataFrame utilities."""

from __future__ import annotations

__all__ = [
    "add_author_column",
    "add_openreview_profile_ids_to_dataframe",
    "add_openreview_profiles_to_dataframe",
]

from typing import TYPE_CHECKING, Any

import polars as pl

from candidex.columns import (
    AUTHOR_ID,
    AUTHOR_OPENREVIEW_PROFILE,
    AUTHOR_OPENREVIEW_PROFILE_ID,
)
from candidex.openreview import serialize_profiles

if TYPE_CHECKING:
    from openreview import Profile

    from candidex.author import Author


def add_author_column(
    frame: pl.DataFrame,
    data: dict[Author, Any],
    column_name: str,
    dtype: pl.DataType,
) -> pl.DataFrame:
    """Add a column to a DataFrame by left joining on the author ID.

    Builds a two-column DataFrame mapping author hashes to the provided data,
    then left joins it onto the input frame using the `AUTHOR_ID` column.
    Authors not present in `data` are mapped to null.

    Args:
        frame:       A Polars DataFrame containing an `AUTHOR_ID` column as
                     returned by `authors_to_dataframe` with `include_id=True`.
        data:        A dictionary mapping each `Author` to the value to add.
                     Authors not present in the dictionary are mapped to null.
        column_name: Name of the new column to add.
        dtype:       Polars data type of the new column (e.g. `pl.List(pl.String)`).

    Returns:
        The input DataFrame with an additional column named `column_name`
        containing the values from `data` keyed by author hash. Null for
        authors not present in `data`.

    Example:
        ```pycon
        >>> from candidex.author import Author, authors_to_dataframe, add_author_column
        >>> import polars as pl
        >>> authors = [Author.from_raw("Jane Smith", ["MIT"], "jane@mit.edu")]
        >>> frame = authors_to_dataframe(authors, include_id=True)
        >>> data = {authors[0]: ["~Jane_Smith1"]}
        >>> result = add_author_column(frame, data, "profile_ids", pl.List(pl.String))
        >>> result["profile_ids"]
        shape: (1,)
        Series: 'profile_ids' [list[str]]
        [
            ["~Jane_Smith1"]
        ]

        ```
    """
    lookup_frame = pl.DataFrame(
        {
            AUTHOR_ID: [author.hash() for author in data],
            column_name: list(data.values()),
        },
        schema={
            AUTHOR_ID: pl.String,
            column_name: dtype,
        },
    )
    return frame.join(lookup_frame, on=AUTHOR_ID, how="left")


def add_openreview_profile_ids_to_dataframe(
    frame: pl.DataFrame,
    profile_ids_by_author: dict[Author, list[str] | None],
) -> pl.DataFrame:
    """Add OpenReview profile IDs to a DataFrame by joining on author
    ID.

    Convenience wrapper around `add_author_column` that adds an
    `AUTHOR_OPENREVIEW_PROFILE_ID` column of type `List[String]`.

    Args:
        frame:                 A Polars DataFrame containing an `AUTHOR_ID`
                               column as returned by `authors_to_dataframe`
                               with `include_id=True`.
        profile_ids_by_author: A dictionary mapping each `Author` to their
                               list of OpenReview profile ID strings, or None
                               if the lookup failed. As returned by
                               `extract_profile_ids_by_author`.

    Returns:
        The input DataFrame with an additional `AUTHOR_OPENREVIEW_PROFILE_ID`
        column of type `List[String]`. Null for authors not present in
        `profile_ids_by_author` or whose lookup failed.

    Example:
        ```pycon
        >>> from candidex.author import Author, authors_to_dataframe
        >>> from candidex.author import add_openreview_profile_ids_to_dataframe
        >>> authors = [Author.from_raw("Jane Smith", ["MIT"], "jane@mit.edu")]
        >>> frame = authors_to_dataframe(authors, include_id=True)
        >>> result = add_openreview_profile_ids_to_dataframe(frame, {authors[0]: ["~Jane_Smith1"]})
        >>> result[AUTHOR_OPENREVIEW_PROFILE_ID]
        shape: (1,)
        Series: 'author_openreview_profile_id' [list[str]]
        [
            ["~Jane_Smith1"]
        ]

        ```
    """
    return add_author_column(
        frame,
        profile_ids_by_author,
        AUTHOR_OPENREVIEW_PROFILE_ID,
        pl.List(pl.String),
    )


def add_openreview_profiles_to_dataframe(
    frame: pl.DataFrame,
    profiles_by_author: dict[Author, list[Profile] | None],
) -> pl.DataFrame:
    """Add OpenReview profiles to a DataFrame by joining on author ID.

    Convenience wrapper around `add_author_column` that adds an
    `AUTHOR_OPENREVIEW_PROFILE` column of type `Object`.

    Args:
        frame:               A Polars DataFrame containing an `AUTHOR_ID`
                             column as returned by `authors_to_dataframe`
                             with `include_id=True`.
        profiles_by_author:  A dictionary mapping each `Author` to their
                             list of `Profile` objects, or None if the
                             lookup failed. As returned by
                             `extract_profiles_by_author`.

    Returns:
        The input DataFrame with an additional `AUTHOR_OPENREVIEW_PROFILE`
        column of type `Object`. Null for authors not present in
        `profiles_by_author` or whose lookup failed.

    Example:
        ```pycon
        >>> from candidex.author import Author, authors_to_dataframe
        >>> from candidex.author import add_openreview_profiles_to_dataframe
        >>> authors = [Author.from_raw("Jane Smith", ["MIT"], "jane@mit.edu")]
        >>> frame = authors_to_dataframe(authors, include_id=True)
        >>> result = add_openreview_profiles_to_dataframe(frame, {authors[0]: None})
        >>> result[AUTHOR_OPENREVIEW_PROFILE]
        shape: (1,)
        Series: 'author_openreview_profile' [list[str]]
        [
            null
        ]

        ```
    """
    serialized = {
        author: serialize_profiles(profiles) if profiles is not None else None
        for author, profiles in profiles_by_author.items()
    }
    return add_author_column(
        frame,
        serialized,
        AUTHOR_OPENREVIEW_PROFILE,
        pl.List(pl.String),
    )
