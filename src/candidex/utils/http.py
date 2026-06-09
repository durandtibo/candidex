r"""Contain utilities for fetching data from web pages."""

from __future__ import annotations

__all__ = ["fetch_html"]


import logging
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger: logging.Logger = logging.getLogger(__name__)


HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def create_session(max_retries: int = 3) -> requests.Session:
    """Create a `requests.Session` with a retry adapter mounted.

    Configures exponential backoff retries for transient failures on both
    `https://` and `http://` connections. Useful for sharing a single
    session across multiple requests for connection pooling.

    Args:
        max_retries: Maximum number of retry attempts on transient failures
            (429, 500, 502, 503, 504) with exponential backoff.
            Defaults to 3.

    Returns:
        A configured `requests.Session` instance with retry adapters mounted.

    Example:
        >>> from candidex.utils.http import create_session
        >>> session = create_session(max_retries=5)
        >>> session.get("https://example.com")  # doctest: +SKIP
    """
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_html(
    url: str,
    timeout: int = 30,
    max_retries: int = 3,
    headers: dict[str, str] | None = None,
    session: requests.Session | None = None,
) -> str:
    """Fetch the raw HTML content of a webpage with automatic retries.

    Uses exponential backoff retries to handle transient network failures,
    connection timeouts, and 5xx server errors. Each retry waits progressively
    longer before attempting again: 1s, 2s, 4s, ... up to `max_retries`
    attempts.

    If a `session` is provided, it is used directly without creating a new
    one. This allows callers to reuse a single session across multiple calls,
    benefiting from connection pooling.

    Args:
        url: The full URL to fetch.
        timeout: Request timeout in seconds per attempt. Defaults to 30.
        max_retries: Maximum number of retry attempts on failure. Defaults to 3.
            Set to 0 to disable retries. Ignored if `session` is
            provided.
        headers: HTTP headers to send with the request. If not provided,
            defaults to `HEADERS` which mimics a real browser to
            avoid being blocked. Pass an empty dict to send no
            headers.
        session: An optional `requests.Session` instance to reuse. If not
            provided, a new session is created with a retry adapter.

    Returns:
        The raw HTML string of the page.

    Raises:
        requests.exceptions.ConnectTimeout:   If all retry attempts exceed `timeout`.
        requests.exceptions.HTTPError:        On 4xx/5xx responses that are not retried.
        requests.exceptions.ConnectionError:  If the host is unreachable after all retries.
        requests.exceptions.RequestException: For any other unrecoverable network failure.

    Example:
        ```pycon
        >>> from candidex.utils.http import fetch_html
        >>> html = fetch_html("https://openaccess.thecvf.com/CVPR2024?day=all")  # doctest: +SKIP

        ```
    """
    logger.debug("Fetching %s...", url)

    resolved_headers = headers if headers is not None else HEADERS

    own_session = session is None
    if own_session:
        session = create_session(max_retries=max_retries)

    try:
        start = time.perf_counter()
        response = session.get(url, headers=resolved_headers, timeout=timeout)
        elapsed = time.perf_counter() - start
        logger.debug(
            "Response received: HTTP %d (%d bytes) in %.2fs",
            response.status_code,
            len(response.content),
            elapsed,
        )
        response.raise_for_status()
        return response.text
    finally:
        if own_session:
            session.close()
