r"""Define the class for scraping CVF papers."""

from __future__ import annotations

__all__ = ["CVFPaperScraper"]

import logging
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup, Tag

from candidex.scraper.base import BasePaperScraper

if TYPE_CHECKING:
    import polars as pl

logger: logging.Logger = logging.getLogger(__name__)

PAPER_ENTRY_CLASS = "ptitle"
HTML_PARSER = "html.parser"


class CVFPaperScraper(BasePaperScraper):
    """Scrape papers from the CVF open access website (CVPR, ICCV,
    WACV)."""

    def __init__(self, venue: str, year: int) -> None:
        self._venue = venue.strip()
        self._year = year

    def scrape(self) -> pl.DataFrame: ...


def parse_paper_entries(html: str) -> list[Tag]:
    """Parse all paper entry tags from a CVPR OpenAccess listing page.

    Each paper on the listing page is represented by a `<dt class='ptitle'>`
    tag, which acts as the anchor for the title, authors, and PDF link.

    Args:
        html: Raw HTML string of the CVPR listing page.

    Returns:
        A list of `<dt>` Tag objects, one per paper. Returns an empty list
        if no paper entries are found, which may indicate a malformed page
        or a change in the page structure.
    """
    soup = BeautifulSoup(html, HTML_PARSER)
    entries = soup.find_all("dt", class_=PAPER_ENTRY_CLASS)

    if not entries:
        logger.warning(
            "No paper entries found in HTML — the page may be malformed or "
            "the structure of the CVF listing page may have changed."
        )
    else:
        logger.debug("Found %d paper entries in HTML.", len(entries))

    return entries
