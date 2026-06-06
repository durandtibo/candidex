r"""Define the base class for scraping papers from a venue."""

from __future__ import annotations

__all__ = ["BasePaperScraper"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl


class BasePaperScraper(ABC):
    """Base class for scraping papers from a venue.

    Defines the interface that all venue-specific paper scrapers must
    implement. Subclasses are responsible for fetching and parsing papers
    from a specific venue (e.g. CVF, OpenReview) and returning them as a
    Polars DataFrame with a consistent schema.

    Subclasses must implement `scrape`, which should return a DataFrame
    with at least the following columns as defined in `candidex.columns`:
        - `PAPER_TITLE`  (String): Title of the paper.
        - `PAPER_URL`    (String): URL of the paper's page.
        - `PAPER_PDF_URL`(String): URL of the paper's PDF.
        - `AUTHORS`      (List[String]): List of author names.
    """

    @abstractmethod
    def scrape(self) -> pl.DataFrame:
        """Scrape papers from the venue URL.

        Fetches and parses the paper listing page at `self.url` and returns
        all papers as a Polars DataFrame. Implementations should handle
        pagination, retries, and parsing errors internally.

        Returns:
            A Polars DataFrame with at least the following columns:
                - `PAPER_TITLE`   (String):       Title of the paper.
                - `PAPER_URL`     (String):       URL of the paper's page.
                - `PAPER_PDF_URL` (String):       URL of the paper's PDF.
                - `AUTHORS`       (List[String]): List of author names.
        """
