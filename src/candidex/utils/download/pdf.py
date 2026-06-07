r"""Contain utilities for downloading PDFs."""

from __future__ import annotations

__all__ = ["download_pdf"]


import logging
from typing import TYPE_CHECKING

import requests

from candidex.utils.http import HEADERS, create_session

if TYPE_CHECKING:
    from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


def download_pdf(
    url: str,
    pdf_path: Path,
    timeout: int = 30,
    chunk_size: int = 8192,
    max_retries: int = 3,
    session: requests.Session | None = None,
) -> bool:
    """Download a single PDF from a URL to a destination path.

    Streams the response to avoid loading large PDFs fully into memory.
    Skips the download if the file already exists at `pdf_path`, making the
    function safe to call repeatedly without re-downloading completed files.

    Uses an atomic write via a temporary file to avoid leaving partial files
    on disk if the download is interrupted. If a `session` is provided, it
    is used directly without creating a new one, allowing callers to reuse
    a single session across multiple calls for connection pooling.

    Args:
        url:         Direct URL to the PDF to download.
        pdf_path:    Full destination path including filename, e.g.
                     Path("papers/attention_is_all_you_need.pdf").
                     Parent directories are created automatically if they
                     do not exist.
        timeout:     Request timeout in seconds. Defaults to 30.
        chunk_size:  Number of bytes per chunk when streaming the response.
                     Larger values use more memory but may be faster on fast
                     connections. Defaults to 8192.
        max_retries: Maximum number of retry attempts on transient failures
                     (429, 500, 502, 503, 504) with exponential backoff.
                     Defaults to 3. Ignored if `session` is provided.
        session:     An optional `requests.Session` instance to reuse. If
                     not provided, a new session is created via
                     `create_session` and closed after the request completes.

    Returns:
        True if the file was downloaded successfully or already exists.
            False if the download failed due to a network or HTTP error.

    Raises:
        OSError: If the destination path is not writable.

    Example:
        >>> from pathlib import Path
        >>> from candidex.utils.download import download_pdf
        >>> download_pdf(
        ...     url="https://arxiv.org/pdf/1706.03762",
        ...     pdf_path=Path("papers/attention.pdf"),
        ... )  # doctest: +SKIP
        True
    """
    if pdf_path.exists():
        logger.debug("Skipping %s — already exists at %s.", url, pdf_path)
        return True

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pdf_path.with_suffix(".tmp")

    own_session = session is None
    if own_session:
        session = create_session(max_retries=max_retries)

    try:
        with session.get(url, headers=HEADERS, timeout=timeout, stream=True) as response:
            response.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
        tmp.rename(pdf_path)
    except requests.exceptions.RequestException as e:
        logger.warning("Failed to download %s: %s", url, e)
        tmp.unlink(missing_ok=True)
        return False
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        if own_session:
            session.close()

    size_kb = pdf_path.stat().st_size / 1024
    logger.debug("Downloaded %s -> %s (%.1f KB)", url, pdf_path, size_kb)
    return True
