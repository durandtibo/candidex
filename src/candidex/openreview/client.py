r"""Contain code to create an authenticated OpenReview client."""

from __future__ import annotations

__all__ = ["create_openreview_client"]

import logging
import os

from openreview.api import OpenReviewClient

logger: logging.Logger = logging.getLogger(__name__)

OPENREVIEW_BASE_URL = "https://api2.openreview.net"


def create_openreview_client(
    username: str | None = None,
    password: str | None = None,
) -> OpenReviewClient | None:
    """Create an authenticated OpenReview API client.

    Credentials can be provided directly as arguments or via the
    `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD` environment variables.
    Explicit arguments take priority over environment variables.

    Args:
        username: OpenReview account username. If not provided, falls back
                  to the `OPENREVIEW_USERNAME` environment variable.
        password: OpenReview account password. If not provided, falls back
                  to the `OPENREVIEW_PASSWORD` environment variable.

    Returns:
        An authenticated `OpenReviewClient` instance, or None if credentials
        are missing or authentication fails.

    Example:
        >>> client = create_openreview_client()  # uses env vars
        >>> client = create_openreview_client(username="user@example.com", password="secret")
    """
    username = username or os.getenv("OPENREVIEW_USERNAME")
    password = password or os.getenv("OPENREVIEW_PASSWORD")

    if not username:
        logger.error("No username provided and OPENREVIEW_USERNAME is not set.")
        return None

    if not password:
        logger.error("No password provided and OPENREVIEW_PASSWORD is not set.")
        return None

    logger.info("Authenticating with OpenReview as %s...", username)

    try:
        client = OpenReviewClient(
            baseurl=OPENREVIEW_BASE_URL,
            username=username,
            password=password,
        )
    except Exception:
        logger.exception("Failed to authenticate with OpenReview.")
        return None
    else:
        logger.info("Successfully authenticated with OpenReview as %s.", username)
        return client
