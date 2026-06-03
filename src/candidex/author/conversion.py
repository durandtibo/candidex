r"""Contain conversion utilities."""

from __future__ import annotations

__all__ = ["authors_to_dataframe"]


from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from candidex.author import Author


def authors_to_dataframe(authors: Sequence[Author]) -> pl.DataFrame:
    """Convert a sequence of authors to a Polars DataFrame.

    Each `Author` is represented as a row with columns for name, affiliations,
    and email. The affiliations tuple is converted to a list for Polars
    compatibility.

    Args:
        authors: A sequence of `Author` objects to convert.

    Returns:
        A Polars DataFrame with columns:
            - name          (String):       Full name of the author.
            - affiliations  (List[String]): List of institutional affiliations.
                                            Empty list if None.
            - email         (String):       Email address, or null if None.

    Example:
        >>> from candidex.author import Author
        >>> from candidex.author import authors_to_dataframe
        >>> authors = [
        ...     Author.from_raw("Jane Smith", ["MIT"], "jane@mit.edu"),
        ...     Author.from_raw("John Doe", ["Stanford"], None),
        ... ]
        >>> authors_to_dataframe(authors)
        shape: (2, 3)
        ┌────────────┬──────────────┬──────────────┐
        │ name       ┆ affiliations ┆ email        │
        │ ---        ┆ ---          ┆ ---          │
        │ str        ┆ list[str]    ┆ str          │
        ╞════════════╪══════════════╪══════════════╡
        │ Jane Smith ┆ ["MIT"]      ┆ jane@mit.edu │
        │ John Doe   ┆ ["Stanford"] ┆ null         │
        └────────────┴──────────────┴──────────────┘

        ...
    """
    return pl.DataFrame(
        {
            "name": [a.name for a in authors],
            "affiliations": [
                list(a.affiliations) if a.affiliations is not None else None for a in authors
            ],
            "email": [a.email for a in authors],
        },
        schema={
            "name": pl.String,
            "affiliations": pl.List(pl.String),
            "email": pl.String,
        },
    )
