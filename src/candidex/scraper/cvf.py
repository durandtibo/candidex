r"""Define the class for scraping CVF papers."""

from __future__ import annotations

__all__ = ["CVFPaperScraper", "build_listing_url", "parse_paper_entries"]

import logging
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup, Tag

from candidex.scraper.base import BasePaperScraper

if TYPE_CHECKING:
    import polars as pl

logger: logging.Logger = logging.getLogger(__name__)

BASE_URL = "https://openaccess.thecvf.com"

PAPER_ENTRY_CLASS = "ptitle"
HTML_PARSER = "html.parser"


class CVFPaperScraper(BasePaperScraper):
    """Scrape papers from the CVF open access website (CVPR, ICCV,
    WACV)."""

    def __init__(self, venue: str, year: int) -> None:
        self._venue = venue.strip()
        self._year = year

    def scrape(self) -> pl.DataFrame: ...


def build_listing_url(venue: str, year: int) -> str:
    """Build the CVF OpenAccess listing URL for a given venue and year.

    Constructs the URL of the page that lists all papers for a specific
    venue and year on the CVF OpenAccess website.

    Args:
        venue: The venue name as it appears in the CVF URL
               (e.g. 'CVPR', 'ICCV', 'ECCV').
        year:  The year of the venue (e.g. 2024).

    Returns:
        The full URL of the CVF listing page for the given venue and year.

    Example:
        ```pycon
        >>> from candidex.scraper.cvf import build_listing_url
        >>> build_listing_url("CVPR", 2024)
        'https://openaccess.thecvf.com/CVPR2024?day=all'
        >>> build_listing_url("ICCV", 2023)
        'https://openaccess.thecvf.com/ICCV2023?day=all'

        ```
    """
    return f"{BASE_URL}/{venue}{year}?day=all"


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
