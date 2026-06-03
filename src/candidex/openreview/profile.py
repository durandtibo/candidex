r"""Contain utilities for finding OpenReview profiles."""

from __future__ import annotations

__all__ = ["fetch_profile_by_id", "get_unique_profiles"]

import logging
from typing import TYPE_CHECKING

from openreview import OpenReviewException

from candidex.openreview.client import create_client

if TYPE_CHECKING:
    from openreview import Profile
    from openreview.api import OpenReviewClient


logger: logging.Logger = logging.getLogger(__name__)


def get_unique_profiles(profiles_list: list[list[Profile]]) -> list[Profile]:
    """Return a deduplicated list of profiles from a list of profile
    lists.

    Flattens the input into a single list and deduplicates by profile ID,
    preserving the first occurrence of each profile when duplicates are found.

    Args:
        profiles_list: A list of profile lists, as returned by multiple
                       calls to `search_openreview_profiles`. May contain
                       duplicates across lists.

    Returns:
        A flat list of unique `openreview.Profile` objects, deduplicated
            by `Profile.id`. Order reflects first occurrence across the input
            lists. Returns an empty list if the input is empty or all lists
            are empty.

    Example:
        >>> results = [
        ...     search_openreview_profiles("Jane Smith"),
        ...     search_openreview_profiles("Jane Smith"),
        ... ] # doctest: +SKIP
        >>> unique = get_unique_profiles(results) # doctest: +SKIP
    """
    seen: set[str] = set()
    unique: list[Profile] = []
    for profiles in profiles_list:
        for profile in profiles:
            if profile.id not in seen:
                seen.add(profile.id)
                unique.append(profile)
    return unique


def fetch_profile_by_id(
    profile_id: str,
    client: OpenReviewClient | None = None,
) -> Profile | None:
    """Fetch an OpenReview profile by its profile ID.

    Queries the OpenReview API for the profile with the given ID. If no
    client is provided, one is created automatically using credentials
    from the environment.

    Args:
        profile_id: The OpenReview profile ID to fetch
                    (e.g. '~Thibaut_Durand1').
        client:     An authenticated `OpenReviewClient` instance. If not
                    provided, one is created via `create_client()` using
                    the `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD`
                    environment variables.

    Returns:
        The `Profile` object for the given ID, or None if the profile
        does not exist, the request fails, or the client cannot be created.

    Example:
        ```pycon
        >>> from candidex.openreview import fetch_profile_by_id
        >>> profile = fetch_profile_by_id("~Thibaut_Durand1")  # doctest: +SKIP
        >>> if profile:  # doctest: +SKIP
        ...     print(profile.id)
        ...

        ```
    """
    client = client or create_client()
    if client is None:
        logger.warning("No OpenReview client available, cannot fetch profile for %s.", profile_id)
        return None

    logger.debug("Fetching OpenReview profile for ID '%s'.", profile_id)
    try:
        profile = client.get_profile(profile_id)
    except OpenReviewException:
        logger.warning("OpenReview profile not found: '%s'.", profile_id)
        return None
    else:
        logger.debug("Successfully fetched profile '%s'.", profile_id)
        return profile
