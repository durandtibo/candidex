r"""Contain conversion utilities."""

from __future__ import annotations

__all__ = ["authors_to_dataframe"]


from typing import TYPE_CHECKING

import polars as pl

from candidex.columns import AUTHOR_AFFILIATION, AUTHOR_EMAIL, AUTHOR_ID, AUTHOR_NAME

if TYPE_CHECKING:
    from collections.abc import Sequence

    from candidex.author import Author


def authors_to_dataframe(
    authors: Sequence[Author],
    *,
    include_id: bool = False,
) -> pl.DataFrame:
    """Convert a sequence of authors to a Polars DataFrame.

    Each `Author` is represented as a row with columns for name, affiliations,
    and email. The affiliations tuple is converted to a list for Polars
    compatibility. Optionally includes a column with the author's hash-based ID.

    Args:
        authors: A sequence of `Author` objects to convert.
        include_id: If True, includes an `author_id` column containing the
                    SHA-256 hash of each author as returned by `Author.hash()`.
                    Defaults to False.

    Returns:
        A Polars DataFrame with columns:
            - author_name         (String):       Full name of the author.
            - author_affiliation  (List[String]): List of institutional affiliations.
                                                  None if affiliations is None.
            - author_email        (String):       Email address, or null if None.
            - author_id           (String):       Hash-based author ID, only
                                                  present if `include_id=True`.

    Example:
        >>> from candidex.author import Author
        >>> from candidex.author import authors_to_dataframe
        >>> authors = [
        ...     Author.from_raw("Jane Smith", ["MIT"], "jane@mit.edu"),
        ...     Author.from_raw("John Doe", ["Stanford"], None),
        ... ]
        >>> authors_to_dataframe(authors)
        shape: (2, 3)
        ┌─────────────┬────────────────────┬──────────────┐
        │ author_name ┆ author_affiliation ┆ author_email │
        │ ---         ┆ ---                ┆ ---          │
        │ str         ┆ list[str]          ┆ str          │
        ╞═════════════╪════════════════════╪══════════════╡
        │ Jane Smith  ┆ ["MIT"]            ┆ jane@mit.edu │
        │ John Doe    ┆ ["Stanford"]       ┆ null         │
        └─────────────┴────────────────────┴──────────────┘

        ...
    """
    data = {
        AUTHOR_NAME: [a.name for a in authors],
        AUTHOR_AFFILIATION: [
            list(a.affiliations) if a.affiliations is not None else None for a in authors
        ],
        AUTHOR_EMAIL: [a.email for a in authors],
    }
    schema = {
        AUTHOR_NAME: pl.String,
        AUTHOR_AFFILIATION: pl.List(pl.String),
        AUTHOR_EMAIL: pl.String,
    }

    if include_id:
        data[AUTHOR_ID] = [a.hash() for a in authors]
        schema[AUTHOR_ID] = pl.String

    return pl.DataFrame(data, schema=schema)
