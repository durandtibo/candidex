r"""Contain conversion utilities."""

from __future__ import annotations

__all__ = ["papers_to_dataframe"]


from typing import TYPE_CHECKING

import polars as pl

from candidex.columns import (
    PAPER_AUTHORS,
    PAPER_ID,
    PAPER_TITLE,
    PAPER_URL,
    PAPER_VENUE,
    PAPER_YEAR,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from candidex.paper import Paper


def papers_to_dataframe(
    papers: Sequence[Paper],
    *,
    include_id: bool = False,
) -> pl.DataFrame:
    """Convert a sequence of papers to a Polars DataFrame.

    Each `Paper` is represented as a row with columns for title, authors,
    venue, year, and PDF URL. Optionally includes a column with the paper's
    hash-based ID.

    Args:
        papers:     A sequence of `Paper` objects to convert.
        include_id: If True, includes a `paper_id` column containing the
                    BLAKE2b hash of each paper as returned by `Paper.hash()`.
                    Defaults to False.

    Returns:
        A Polars DataFrame with columns:
            - paper_title   (String):       Title of the paper.
            - paper_authors (List[String]): List of author names.
            - paper_venue   (String):       Venue where the paper was published.
            - paper_year    (Int32):        Year the paper was published.
            - paper_url     (String):       URL of the paper's PDF.
            - paper_id      (String):       Hash-based paper ID, only present
                                            if `include_id=True`.

    Example:
        >>> from candidex.schemas.paper import Paper
        >>> from candidex.paper import papers_to_dataframe
        >>> papers = [
        ...     Paper.from_raw(
        ...         title="Attention Is All You Need",
        ...         authors=["Ashish Vaswani", "Noam Shazeer"],
        ...         venue="NeurIPS",
        ...         year=2017,
        ...         pdf_url="https://arxiv.org/pdf/1706.03762",
        ...     ),
        ... ]
        >>> df_papers = papers_to_dataframe(papers)
        >>> df_papers.columns
        shape: (1, 5)
        >>> df_papers
        shape: (1, 5)

        ...
    """
    data = {
        PAPER_TITLE: [p.title for p in papers],
        PAPER_AUTHORS: [list(p.authors) for p in papers],
        PAPER_VENUE: [p.venue for p in papers],
        PAPER_YEAR: [p.year for p in papers],
        PAPER_URL: [p.pdf_url for p in papers],
    }
    schema = {
        PAPER_TITLE: pl.String,
        PAPER_AUTHORS: pl.List(pl.String),
        PAPER_VENUE: pl.String,
        PAPER_YEAR: pl.Int32,
        PAPER_URL: pl.String,
    }

    if include_id:
        data[PAPER_ID] = [p.hash() for p in papers]
        schema[PAPER_ID] = pl.String

    return pl.DataFrame(data, schema=schema)
