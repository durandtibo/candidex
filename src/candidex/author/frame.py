r"""Contain DataFrame utilities."""

from __future__ import annotations

__all__ = ["add_openreview_profile_ids_to_dataframe"]


from typing import TYPE_CHECKING

import polars as pl

from candidex.columns import (
    AUTHOR_ID,
    AUTHOR_OPENREVIEW_PROFILE_ID,
)

if TYPE_CHECKING:
    from candidex.author import Author


def add_openreview_profile_ids_to_dataframe(
    frame: pl.DataFrame,
    profile_ids_by_author: dict[Author, list[str] | None],
) -> pl.DataFrame:
    """Add OpenReview profile IDs to a DataFrame by joining on author
    ID.

    Maps each row in the DataFrame to its OpenReview profile IDs using the
    `AUTHOR_ID` column and the provided dictionary. The author hash is used
    as the join key. Authors with no profile IDs are mapped to an empty list.
    Authors with a failed lookup (None) are mapped to null.

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
            column of type `List[String]`, containing the OpenReview profile IDs
            for each author. Null if the lookup failed, empty list if no profiles
            were found.

    Example:
        ```pycon
        >>> from candidex.author import (
        ...     Author,
        ...     authors_to_dataframe,
        ...     add_openreview_profile_ids_to_dataframe,
        ... )
        >>> from candidex.columns import AUTHOR_OPENREVIEW_PROFILE_ID
        >>> authors = [Author.from_raw("Jane Smith", ["MIT"], "jane@mit.edu")]
        >>> frame = authors_to_dataframe(authors, include_id=True)
        >>> profile_ids_by_author = {authors[0]: ["~Jane_Smith1"]}
        >>> result = add_openreview_profile_ids_to_dataframe(frame, profile_ids_by_author)
        >>> result
        shape: (1, 5)
        ┌─────────────┬────────────────────┬──────────────┬────────────────────────┬───────────────────────┐
        │ author_name ┆ author_affiliation ┆ author_email ┆ author_id              ┆ author_openreview_pro │
        │ ---         ┆ ---                ┆ ---          ┆ ---                    ┆ file_id               │
        │ str         ┆ list[str]          ┆ str          ┆ str                    ┆ ---                   │
        │             ┆                    ┆              ┆                        ┆ list[str]             │
        ╞═════════════╪════════════════════╪══════════════╪════════════════════════╪═══════════════════════╡
        │ Jane Smith  ┆ ["MIT"]            ┆ jane@mit.edu ┆ 7f98ad3b48a68a5ead8a53 ┆ ["~Jane_Smith1"]      │
        │             ┆                    ┆              ┆ 10e333c6…              ┆                       │
        └─────────────┴────────────────────┴──────────────┴────────────────────────┴───────────────────────┘

        ```
    """
    lookup: dict[str, list[str] | None] = {
        author.hash(): profile_ids for author, profile_ids in profile_ids_by_author.items()
    }

    profile_ids_frame = pl.DataFrame(
        {
            AUTHOR_ID: list(lookup.keys()),
            AUTHOR_OPENREVIEW_PROFILE_ID: list(lookup.values()),
        },
        schema={
            AUTHOR_ID: pl.String,
            AUTHOR_OPENREVIEW_PROFILE_ID: pl.List(pl.String),
        },
    )
    return frame.join(profile_ids_frame, on=AUTHOR_ID, how="left")
