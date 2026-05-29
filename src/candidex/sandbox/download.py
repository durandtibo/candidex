r"""Contain utilities to download papers."""

from __future__ import annotations

__all__ = ["download_papers"]

import logging
from typing import TYPE_CHECKING

import polars as pl
import requests
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

if TYPE_CHECKING:
    from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


def download_paper(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download a single file from a URL to a destination path.

    Streams the response to avoid loading large PDFs fully into memory.
    Skips the download if the file already exists at `dest`, making the
    function safe to call repeatedly without re-downloading completed files.

    Args:
        url:     Direct URL to the file to download.
        dest:    Full destination path including filename, e.g.
                 Path("papers/attention_is_all_you_need.pdf").
                 Parent directories must already exist.
        timeout: Request timeout in seconds. Defaults to 30.

    Returns:
        True if the file was downloaded successfully, False if the download
        failed due to a network or HTTP error. Skipped files (already exist)
        also return True since the file is present on disk.

    Raises:
        OSError: If the destination path is not writable.
    """
    if dest.exists():
        logger.debug("Skipping %s, already exists at %s.", url, dest)
        return True

    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning("Failed to download %s: %s", url, e)
        return False

    with dest.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.debug("Downloaded %s -> %s", url, dest)
    return True


def download_papers(
    urls: list[str],
    output_path: Path,
    timeout: int = 30,
) -> pl.DataFrame:
    """Download a list of files from URLs into a directory, with a
    progress bar.

    Iterates over each URL, derives a filename from the URL's last path
    segment, and streams the file to disk. Skips files that already exist,
    making repeated calls safe for resuming interrupted downloads. Failures
    on individual URLs are logged as warnings and recorded in the returned
    summary rather than raising, so a single bad URL does not abort the
    entire batch.

    Args:
        urls:       List of direct file URLs to download. Empty strings and
                    duplicates are silently skipped.
        output_path: Directory to save downloaded files into. Created
                    automatically if it does not exist.
        timeout:    Per-request timeout in seconds. Defaults to 30.

    Returns:
        A Polars DataFrame summarising the result of each download attempt,
        with columns:
            - url     (String):  The original URL.
            - dest    (String):  Absolute path to the file on disk. Empty
                                 string if the download failed.
            - success (Boolean): True if the file is present on disk after
                                 the attempt (downloaded or already existed).

    Example:
        >>> df = scrape_cvpr_papers("https://openaccess.thecvf.com/CVPR2024?day=all")
        >>> results = download_papers(df["pdf_url"].to_list(), Path("papers/cvpr2024"))
        >>> results.filter(~pl.col("success"))  # inspect failures
    """
    output_path.mkdir(parents=True, exist_ok=True)

    valid_urls = list(dict.fromkeys(url for url in urls if url))
    skipped = len(urls) - len(valid_urls)
    if skipped:
        logger.warning("Skipping %d empty or duplicate URLs.", skipped)

    logger.info("Downloading %d files to %s...", len(valid_urls), output_path)

    records = []
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )
    with progress:
        task = progress.add_task("Downloading papers", total=len(valid_urls))
        for _i, url in enumerate(valid_urls, start=1):
            filename = url.split("/")[-1]
            dest = output_path.joinpath(filename)
            success = download_paper(url, dest, timeout=timeout)
            records.append(
                {
                    "url": url,
                    "dest": str(dest.resolve()) if success else "",
                    "success": success,
                }
            )
            progress.advance(task)

    df = pl.DataFrame(records, schema={"url": pl.String, "dest": pl.String, "success": pl.Boolean})

    succeeded = df["success"].sum()
    failed = len(df) - succeeded
    logger.info("Download complete. %d succeeded, %d failed.", succeeded, failed)

    return df
