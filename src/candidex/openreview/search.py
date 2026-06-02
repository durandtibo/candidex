r"""Contain search utilities for OpenReview."""

from __future__ import annotations

__all__ = ["search_openreview_profiles"]

import logging
from typing import TYPE_CHECKING

from openreview import OpenReviewException

from candidex.openreview.client import create_openreview_client

if TYPE_CHECKING:
    from openreview import Profile
    from openreview.api import OpenReviewClient


logger: logging.Logger = logging.getLogger(__name__)


def search_openreview_profiles(
    name: str,
    client: OpenReviewClient | None = None,
) -> list[Profile] | None:
    """Search for OpenReview profiles matching a given name.

    Queries the OpenReview API for profiles matching the provided name string.
    If no client is provided, one is created automatically using credentials
    from the environment. Returns an empty list if no profiles are found or
    if the client cannot be created.

    Args:
        name:   The name to search for. Whitespace is stripped before querying.
        client: An authenticated `OpenReviewClient` instance. If not provided,
                one is created via `create_openreview_client()` using the
                `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD` environment
                variables.

    Returns:
        A list of `openreview.Profile` objects matching the search term.
            Returns an empty list if no profiles are found, the client cannot
            be created, or the API call fails.

    Example:
        >>> from candidex.openreview import search_openreview_profiles
        >>> profiles = search_openreview_profiles("Thibaut Durand") # doctest: +SKIP
        >>> for profile in profiles:  # doctest: +SKIP
        ...     print(profile.id)
    """
    client = client or create_openreview_client()
    if client is None:
        logger.warning("No OpenReview client available, cannot search for profiles.")
        return None

    name = name.strip()
    logger.debug("Searching OpenReview profiles for '%s'.", name)

    try:
        profiles = client.search_profiles(term=name)
    except OpenReviewException:
        logger.warning("Failed to search OpenReview profiles for '%s'.", name)
        return None

    logger.debug("Found %d profile(s) for '%s'.", len(profiles), name)
    return profiles
