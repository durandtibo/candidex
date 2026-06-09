r"""Define the class for scraping CVF papers."""

from __future__ import annotations

__all__ = [
    "CVFPaperScraper",
    "build_listing_url",
    "load_or_scrape_papers",
    "parse_paper",
    "parse_paper_entries",
    "resolve_url",
    "scrape_papers",
]

import logging
import re
from typing import TYPE_CHECKING

import polars as pl
from bs4 import BeautifulSoup, Tag
from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

from candidex.columns import PAPER_AUTHORS, PAPER_URL
from candidex.paper import papers_to_dataframe
from candidex.paper.paper import Paper
from candidex.scraper.base import BasePaperScraper
from candidex.utils.http import fetch_html
from candidex.utils.progressbar import make_progressbar

if TYPE_CHECKING:
    from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)

BASE_URL = "https://openaccess.thecvf.com"

PAPER_ENTRY_CLASS = "ptitle"
HTML_PARSER = "html.parser"
PDF_HREF_PATTERN = re.compile(r"\.pdf$", re.IGNORECASE)


class CVFPaperScraper(BasePaperScraper):
    """Scrape papers from the CVF OpenAccess website.

    Supports all venues published on `openaccess.thecvf.com`, including
    CVPR, ICCV, ECCV, and WACV. Results are optionally cached to disk as
    a Parquet file to avoid re-scraping on subsequent runs.

    Args:
        venue:     The venue name as it appears in the CVF URL
                   (e.g. 'CVPR', 'ICCV', 'ECCV', 'WACV').
        year:      The year of the venue (e.g. 2024).
        cache_dir: Directory where the Parquet cache file will be read
                   from or written to. If None, caching is disabled and
                   papers are scraped on every call to `scrape`.

    Raises:
        ValueError: If `venue` is empty or whitespace-only.

    Example:
        ```pycon
        >>> from pathlib import Path
        >>> from candidex.scraper.cvf import CVFPaperScraper
        >>> scraper = CVFPaperScraper(venue="CVPR", year=2024)
        >>> df = scraper.scrape()  # doctest: +SKIP

        ```
    """

    def __init__(
        self,
        venue: str,
        year: int,
        cache_dir: Path | None = None,
    ) -> None:
        venue = venue.strip()
        if not venue:
            msg = "Venue cannot be empty."
            raise ValueError(msg)
        self._venue = venue
        self._year = year
        self._cache_dir = cache_dir

    def __repr__(self) -> str:
        args = {"venue": self._venue, "year": self._year, "cache_dir": self._cache_dir}
        return f"{self.__class__.__qualname__}(\n  {repr_indent(repr_mapping(args))}\n)"

    def __str__(self) -> str:
        args = {"venue": self._venue, "year": self._year, "cache_dir": self._cache_dir}
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(args))}\n)"

    def scrape(self) -> pl.DataFrame:
        return load_or_scrape_papers(venue=self._venue, year=self._year, cache_dir=self._cache_dir)


def build_listing_url(venue: str, year: int) -> str:
    """Build the CVF OpenAccess listing URL for a given venue and year.

    Constructs the URL of the page that lists all papers for a specific
    venue and year on the CVF OpenAccess website.

    Args:
        venue: The venue name as it appears in the CVF URL
               (e.g. 'CVPR', 'ICCV', 'WACV').
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


def resolve_url(href: str, base_url: str) -> str:
    """Resolve a potentially relative href to an absolute URL.

    Args:
        href:     The href attribute value, either absolute or relative.
        base_url: The base URL to prepend to relative hrefs.

    Returns:
        An absolute URL string.
    """
    return href if href.startswith("http") else f"{base_url}{href}"


def parse_paper(
    dt: Tag, venue: str | None = None, year: int | None = None, base_url: str = BASE_URL
) -> Paper:
    """Extract all metadata for a single paper from its <dt> tag and
    return a `Paper` object.

    The CVPR listing page uses a definition list structure where each paper
    occupies one <dt> (title) followed by two <dd> tags: the first holds the
    authors as comma-separated text, and the second contains action links
    including the PDF download.

    Args:
        dt:       The <dt class='ptitle'> Tag for a single paper entry.
        venue:    The venue name (e.g. 'CVPR', 'ICCV'), or None if not known.
        year:     The year of the venue (e.g. 2024), or None if not known.
        base_url: Root URL used to resolve relative hrefs. Defaults to BASE_URL.

    Returns:
        A `Paper` object with title, authors, venue, year, and PDF URL.
            Fields that cannot be parsed are set to None.

    Example:
        >>> from bs4 import BeautifulSoup
        >>> from candidex.scraper.cvf import parse_paper
        >>> dt = BeautifulSoup(
        ...     '<dt class="ptitle"><a href="/paper.html">My Paper</a></dt>'
        ...     '<dd>Jane Smith, John Doe</dd>'
        ...     '<dd><a href="/paper.pdf">pdf</a></dd>',
        ...     "html.parser",
        ... ).dt
        >>> paper = parse_paper(dt, venue="CVPR", year=2024)
        >>> paper.title
        'My Paper'
    """
    # Title from the <a> tag inside <dt>
    a_tag = dt.find("a")
    title = a_tag.get_text(strip=True) if a_tag else dt.get_text(strip=True)

    # Authors from the first <dd> sibling, comma-separated
    dd_authors = dt.find_next_sibling("dd")
    authors = (
        [
            re.sub(r"\s+", " ", auth.strip())
            for auth in dd_authors.get_text(strip=True).split(",")
            if auth.strip()
        ]
        if dd_authors
        else None
    )

    if not authors:
        logger.debug("No authors found for paper: %s", title)

    # PDF URL from any subsequent <dd> siblings
    all_dd = dt.find_next_siblings("dd")
    pdf_link = next(
        (
            dd.find("a", href=PDF_HREF_PATTERN)
            for dd in all_dd
            if dd.find("a", href=PDF_HREF_PATTERN)
        ),
        None,
    )
    pdf_url = resolve_url(pdf_link["href"], base_url) if pdf_link else None

    if pdf_url is None:
        logger.debug("No PDF URL found for paper: %s", title)

    return Paper.from_raw(
        title=title,
        authors=authors,
        venue=venue,
        year=year,
        pdf_url=pdf_url,
    )


def scrape_papers(venue: str, year: int) -> pl.DataFrame:
    """Scrape paper metadata from a CVF OpenAccess listing page into a
    DataFrame.

    Builds the listing URL from the venue and year, fetches the page,
    parses each paper entry, and returns structured metadata as a Polars
    DataFrame. On network failure the exception propagates so callers can
    decide how to handle it (retry, log, etc.) rather than silently
    returning an empty result.

    Args:
        venue: The venue name as it appears in the CVF URL
               (e.g. 'CVPR', 'ICCV', 'WACV').
        year:  The year of the venue (e.g. 2024).

    Returns:
        A Polars DataFrame with columns as defined in `candidex.columns`:
            - PAPER_TITLE   (String):       Paper title.
            - PAPER_AUTHORS (List[String]): Author names. Null if not found.
            - PAPER_VENUE   (String):       Venue name.
            - PAPER_YEAR    (Int32):        Year of the venue.
            - PAPER_URL     (String):       Direct URL to the paper's PDF.
                                            Null if not found.
            - PAPER_ID      (String):       BLAKE2b hash of the paper,
                                            derived from all fields.

    Raises:
        requests.exceptions.RequestException: On any network or HTTP error.

    Example:
        ```pycon
        >>> from candidex.scraper.cvf import scrape_papers
        >>> df = scrape_papers(venue="CVPR", year=2024)  # doctest: +SKIP

        ```
    """
    url = build_listing_url(venue, year)
    logger.info("Scraping %s %d from %s...", venue, year, url)

    html = fetch_html(url)
    entries = parse_paper_entries(html)
    logger.info("Found %d paper entries. Parsing...", len(entries))

    papers: list[Paper] = []
    with make_progressbar() as progress:
        task = progress.add_task("Parsing papers", total=len(entries))
        for dt in entries:
            papers.append(parse_paper(dt, venue=venue, year=year))
            progress.advance(task)

    frame = papers_to_dataframe(papers, include_id=True)

    missing_pdf = frame[PAPER_URL].null_count()
    missing_authors = frame[PAPER_AUTHORS].null_count()
    logger.info(
        "Scraping complete: %d papers extracted, %d missing PDF URLs, %d missing authors.",
        len(frame),
        missing_pdf,
        missing_authors,
    )
    return frame


def load_or_scrape_papers(
    venue: str,
    year: int,
    cache_dir: Path | None = None,
) -> pl.DataFrame:
    """Load cached CVF papers from disk or scrape them if not yet
    cached.

    On the first call for a given venue and year, scrapes the CVF listing
    page, saves the results as a Parquet file under `cache_dir`, and returns
    the DataFrame. On subsequent calls, loads directly from the cached Parquet
    file, making repeated runs fast and network-independent.

    If `cache_dir` is None, scrapes on every call without caching.

    The cache file is named `{venue}_{year}.parquet` (e.g. `CVPR_2024.parquet`).

    Args:
        venue:     The venue name as it appears in the CVF URL
                   (e.g. 'CVPR', 'ICCV', 'ECCV').
        year:      The year of the venue (e.g. 2024).
        cache_dir: Directory where the Parquet cache file will be read from
                   or written to. Created automatically if it does not exist.
                   If None, caching is disabled and papers are scraped on
                   every call.

    Returns:
        A Polars DataFrame with columns as returned by `scrape_cvf_papers`:
            - PAPER_TITLE   (String):       Paper title.
            - PAPER_AUTHORS (List[String]): Author names. Null if not found.
            - PAPER_VENUE   (String):       Venue name.
            - PAPER_YEAR    (Int32):        Year of the venue.
            - PAPER_URL     (String):       Direct URL to the paper's PDF.
                                            Null if not found.
            - PAPER_ID      (String):       BLAKE2b hash of the paper.

    Raises:
        requests.exceptions.RequestException: If the scrape fails due to a
            network or HTTP error and no cached file exists.

    Example:
        ```pycon
        >>> from pathlib import Path
        >>> from candidex.scraper.cvf import load_or_scrape_papers
        >>> df = load_or_scrape_papers(
        ...     venue="CVPR", year=2024, cache_dir=Path("data/papers")
        ... )  # doctest: +SKIP
        >>> df = load_or_scrape_papers(venue="CVPR", year=2024)  # doctest: +SKIP

        ```
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"papers_{venue}_{year}.parquet"

        if path.is_file():
            logger.info("Loading cached %s %d papers from %s.", venue, year, path)
            return pl.read_parquet(path)

        logger.info("No cache found at %s. Scraping %s %d papers...", path, venue, year)
    else:
        logger.info("No cache directory provided. Scraping %s %d papers...", venue, year)

    frame = scrape_papers(venue=venue, year=year)

    if cache_dir is not None:
        frame.write_parquet(path)
        logger.info("Saved %d papers to %s.", len(frame), path)

    return frame
