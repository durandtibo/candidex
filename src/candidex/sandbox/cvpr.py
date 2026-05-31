r"""Contain functionalities to find the CVPR papers from the website."""

from __future__ import annotations

__all__ = ["find_and_save_papers"]


import logging
import re
from pathlib import Path
from typing import Any

import polars as pl
import requests
from bs4 import BeautifulSoup, Tag
from coola.utils.timing import timeblock
from requests.adapters import HTTPAdapter
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from urllib3.util.retry import Retry

from candidex.columns import (
    AUTHORS,
    PAPER_PDF_URL,
    PAPER_STEM,
    PAPER_TITLE,
    PAPER_URL,
)

logger: logging.Logger = logging.getLogger(__name__)


BASE_URL = "https://openaccess.thecvf.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

PAPER_SCHEMA: dict[str, Any] = {
    PAPER_TITLE: pl.String,
    PAPER_URL: pl.String,
    PAPER_PDF_URL: pl.String,
    PAPER_STEM: pl.String,
    AUTHORS: pl.List(pl.String),
}


def fetch_page(url: str, timeout: int = 30, max_retries: int = 3) -> str:
    """Fetch the raw HTML content of a webpage with automatic retries.

    Mounts a retry adapter with exponential backoff on the session to handle
    transient network failures, connection timeouts, and 5xx server errors.
    Each retry waits progressively longer before attempting again:
    1s, 2s, 4s, ... up to `max_retries` attempts.

    Args:
        url:         The full URL to fetch.
        timeout:     Request timeout in seconds per attempt. Defaults to 30.
        max_retries: Maximum number of retry attempts on failure. Defaults to 3.
                     Set to 0 to disable retries.

    Returns:
        The raw HTML string of the page.

    Raises:
        requests.exceptions.ConnectTimeout:   If all retry attempts exceed `timeout`.
        requests.exceptions.HTTPError:        On 4xx/5xx responses that are not retried.
        requests.exceptions.ConnectionError:  If the host is unreachable after all retries.
        requests.exceptions.RequestException: For any other unrecoverable network failure.

    Example:
        >>> html = fetch_page("https://openaccess.thecvf.com/CVPR2024?day=all")
    """
    logger.info("Fetching %s...", url)

    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)

    with requests.Session() as session:
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        with timeblock("Time for fetching the page: {time}"):
            response = session.get(url, headers=HEADERS, timeout=timeout)

    response.raise_for_status()
    logger.debug(
        "Response received: HTTP %d (%d bytes)", response.status_code, len(response.content)
    )
    return response.text


def parse_paper_entries(html: str, limit: int | None = None) -> list[Tag]:
    """Parse all paper entry tags from a CVPR OpenAccess listing page.

    Each paper on the listing page is represented by a <dt class='ptitle'> tag,
    which acts as the anchor for the title, authors, and PDF link.

    Args:
        html:  Raw HTML string of the CVPR listing page.
        limit: Maximum number of entries to return. If None, returns all found.
               Useful for quick tests or partial scrapes without fetching extra pages.

    Returns:
        A list of <dt> Tag objects, one per paper, up to `limit`.
    """
    soup = BeautifulSoup(html, "html.parser")
    entries = soup.find_all("dt", class_="ptitle")
    logger.debug("Found %d paper entries in HTML.", len(entries))

    if limit is not None and len(entries) > limit:
        logger.info("Limiting to %d of %d papers found.", limit, len(entries))
        return entries[:limit]

    return entries


def parse_paper(dt: Tag, base_url: str = BASE_URL) -> dict[str, Any]:
    """Extract all metadata for a single paper from its <dt> tag.

    The CVPR listing page uses a definition list structure where each paper
    occupies one <dt> (title) followed by two <dd> tags: the first holds the
    authors as comma-separated text, and the second contains action links
    including the PDF download.

    Args:
        dt:       The <dt class='ptitle'> Tag for a single paper entry.
        base_url: Root URL used to resolve relative hrefs. Defaults to BASE_URL.

    Returns:
        A dict with keys:
            - title      (str):       Paper title.
            - paper_url  (str):       Absolute URL to the paper's HTML page.
                                      Empty string if no link is found.
            - pdf_url    (str):       Absolute URL to the PDF file.
                                      Empty string if no PDF link is found.
            - authors    (list[str]): Author names parsed from the first <dd>.
                                      Empty list if the <dd> is missing.
    """
    # Title and paper URL from the <a> tag inside <dt>
    a_tag = dt.find("a")
    title = a_tag.get_text(strip=True) if a_tag else dt.get_text(strip=True)
    relative_url = a_tag["href"] if a_tag and a_tag.get("href") else ""
    paper_url = f"{base_url}{relative_url}" if relative_url else ""

    if not paper_url:
        logger.warning("No paper URL found for entry: %s", title)

    # Authors from the first <dd> sibling
    dd_authors = dt.find_next_sibling("dd")
    authors = (
        [
            re.sub(r"\s+", " ", auth.strip())
            for auth in dd_authors.get_text(strip=True).split(",")
            if auth.strip()
        ]
        if dd_authors
        else []
    )

    if not authors:
        logger.warning("No authors found for paper: %s", title)

    # PDF URL from subsequent <dd> siblings
    pdf_url = ""
    dd = dd_authors.find_next_sibling("dd") if dd_authors else dt.find_next_sibling("dd")
    while dd:
        a = dd.find("a", href=re.compile(r"\.pdf$", re.IGNORECASE))
        if a:
            href = a["href"]
            pdf_url = href if href.startswith("http") else f"{base_url}{href}"
            break
        dd = dd.find_next_sibling("dd")

    if not pdf_url:
        logger.warning("No PDF URL found for paper: %s", title)

    return {
        PAPER_TITLE: title,
        PAPER_URL: paper_url,
        PAPER_PDF_URL: pdf_url,
        AUTHORS: authors,
        PAPER_STEM: Path(pdf_url.split("/")[-1]).stem,
    }


def scrape_cvpr_papers(
    url: str,
    base_url: str = BASE_URL,
    limit: int | None = None,
) -> pl.DataFrame:
    """Scrape paper metadata from a CVPR OpenAccess listing page into a
    DataFrame.

    Fetches the listing page, parses each paper entry, and returns structured
    metadata as a typed Polars DataFrame. On network failure the exception
    propagates so callers can decide how to handle it (retry, log, etc.)
    rather than silently returning an empty result.

    Args:
        url:      Full URL of the CVPR listing page, e.g.
                  'https://openaccess.thecvf.com/CVPR2024?day=all'.
        base_url: Root URL for resolving relative hrefs. Defaults to BASE_URL.
        limit:    Maximum number of papers to scrape. Defaults to 100.
                  Pass None to scrape the full listing (typically 2000+ papers).

    Returns:
        A Polars DataFrame with columns:
            - title      (String):       Paper title.
            - paper_url  (String):       URL to the paper's HTML page on CVF.
            - pdf_url    (String):       Direct URL to the paper's PDF.
            - authors    (List[String]): Author names.

    Raises:
        requests.exceptions.RequestException: On any network or HTTP error.

    Example:
        >>> df = scrape_cvpr_papers("https://openaccess.thecvf.com/CVPR2024?day=all")
        >>> df.filter(pl.col(AUTHORS).list.len() > 5)
    """
    html = fetch_page(url)
    entries = parse_paper_entries(html, limit=limit)

    logger.info("Extracting data for %d papers...", len(entries))
    records = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Parsing papers", total=len(entries))
        for dt in entries:
            records.append(parse_paper(dt, base_url))
            progress.advance(task)

    df = pl.DataFrame(records, schema=PAPER_SCHEMA)
    logger.info(
        "Scraping complete. %d papers extracted, %d missing PDF URLs.",
        len(df),
        df[PAPER_PDF_URL].is_in([""]).sum(),
    )
    return df


def find_and_save_papers(
    url: str,
    filepath: Path,
    limit: int | None = None,
) -> pl.DataFrame:
    """Scrape CVPR papers from a listing page and cache the results to
    disk.

    On the first call, scrapes the CVPR listing page at `url`, saves the
    results as a Parquet file at `filepath`, and returns the DataFrame.
    On subsequent calls, skips the scrape entirely and loads directly from
    the cached Parquet file, making repeated runs fast and
    network-independent.

    Args:
        url:      Full URL of the CVPR listing page to scrape, e.g.
                  'https://openaccess.thecvf.com/CVPR2024?day=all'.
        filepath: Path where the Parquet file will be read from or written
                  to. Parent directory must already exist.
        limit:    Maximum number of papers to scrape. Defaults to 100.
                  Pass None to scrape the full listing (typically 2000+ papers).

    Returns:
        A Polars DataFrame with columns as returned by `scrape_cvpr_papers`:
            - title      (String):       Paper title.
            - paper_url  (String):       URL to the paper's HTML page on CVF.
            - pdf_url    (String):       Direct URL to the paper's PDF.
            - authors    (List[String]): Author names.

    Raises:
        requests.exceptions.RequestException: If the scrape fails due to a
            network or HTTP error and no cached file exists.

    Example:
        >>> df = find_and_save_papers(
        ...     url="https://openaccess.thecvf.com/CVPR2024?day=all",
        ...     filepath=Path("data/cvpr2024.parquet"),
        ... )
    """
    logger.info("Finding CVPR papers...")
    if not filepath.is_file():
        logger.info(f"{filepath} not found. Generating the list of papers from {url}...")
        papers = scrape_cvpr_papers(url, limit=limit)
        papers.write_parquet(filepath)

    logger.info(f"Reading papers from file {filepath}...")
    return pl.read_parquet(filepath)
