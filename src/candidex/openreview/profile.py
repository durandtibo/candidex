r"""Contain utilities for finding OpenReview profiles."""

from __future__ import annotations

__all__ = ["find_author_profile_ids"]

import logging
from typing import TYPE_CHECKING

from candidex.openreview.client import create_openreview_client
from candidex.openreview.filtering import (
    filter_profiles_by_affiliation,
    filter_profiles_by_email,
)
from candidex.openreview.search import search_profiles_by_name

if TYPE_CHECKING:
    from openreview.api import OpenReviewClient

logger: logging.Logger = logging.getLogger(__name__)


def find_author_profile_ids(
    name: str,
    affiliation: str,
    email: str | None = None,
    client: OpenReviewClient | None = None,
) -> list[str] | None:
    """Find the OpenReview profile IDs of an author by name and
    affiliation.

    Searches OpenReview for profiles matching the given name, then filters
    by affiliation and optionally by email domain. Returns the IDs of all
    matching profiles sorted alphabetically.

    Args:
        name:        Full name of the author to search for.
        affiliation: Institutional affiliation to filter by (e.g. 'MIT CSAIL').
        email:       Optional email address used as an additional filter.
                     If provided, only profiles whose confirmed emails or
                     institution domains match the email domain are returned.
        client:      An authenticated `OpenReviewClient` instance. If not
                     provided, one is created via `create_openreview_client()`
                     using the `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD`
                     environment variables.

    Returns:
        A sorted list of OpenReview profile ID strings (e.g. ['~Jane_Smith1'])
        for profiles matching the name, affiliation, and optionally email.
        Returns an empty list if no profiles match the filters.
        Returns None if the client cannot be created or the search fails,
        distinguishing infrastructure failures from legitimate empty results.

    Example:
        ```pycon
        >>> from candidex.openreview import find_author_profile_ids
        >>> ids = find_author_profile_ids(
        ...     name="Jane Smith",
        ...     affiliation="MIT CSAIL",
        ...     email="jane@mit.edu",
        ... )  # doctest: +SKIP
        >>> print(ids)
        ['~Jane_Smith1']

        ```
    """
    client = client or create_openreview_client()
    if client is None:
        logger.warning("No OpenReview client available, cannot find profile for %s.", name)
        return None

    profiles = search_profiles_by_name(name, client=client)
    if profiles is None:
        logger.warning("Profile search failed for %s.", name)
        return None
    if not profiles:
        logger.debug("No profiles found for %s.", name)
        return []

    logger.debug("Found %d profile(s) for %s.", len(profiles), name)

    filtered = filter_profiles_by_affiliation(profiles, affiliation=affiliation)

    if email is not None:
        filtered = filter_profiles_by_email(filtered, email=email)

    profile_ids = sorted(p.id for p in filtered)
    logger.debug(
        "Found %d matching profile(s) for %s (%s): %s.",
        len(profile_ids),
        name,
        affiliation,
        profile_ids,
    )
    return profile_ids
