r"""Contain utilities for finding OpenReview profiles."""

from __future__ import annotations

__all__ = ["FilterMode", "find_author_profile_ids", "get_unique_profiles"]

import logging
from enum import StrEnum
from typing import TYPE_CHECKING

from candidex.openreview.client import create_client
from candidex.openreview.filtering import (
    filter_profiles_by_affiliation,
    filter_profiles_by_email,
)
from candidex.openreview.search import search_profiles_by_name

if TYPE_CHECKING:
    from openreview import Profile
    from openreview.api import OpenReviewClient


logger: logging.Logger = logging.getLogger(__name__)


class FilterMode(StrEnum):
    """Filtering mode for combining affiliation and email filters.

    Attributes:
        ALL: AND logic — a profile must match both the affiliation and the
             email domain to be included. More restrictive, reduces false
             positives.
        ANY: OR logic — a profile is included if it matches either the
             affiliation or the email domain. More permissive, reduces false
             negatives.
    """

    ALL = "all"
    ANY = "any"


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


def find_author_profile_ids(
    name: str,
    affiliation: str,
    email: str | None = None,
    mode: FilterMode = FilterMode.ANY,
    client: OpenReviewClient | None = None,
) -> list[str] | None:
    """Find the OpenReview profile IDs of an author by name and
    affiliation.

        Searches OpenReview for profiles matching the given name, then filters
        by affiliation and optionally by email domain. The `mode` parameter
        controls how the two filters are combined:

        - `FilterMode.ANY` (default): OR logic — a profile passes if it matches
          the affiliation OR the email domain. More permissive, reduces false
          negatives when one source of information is missing or inconsistent.
        - `FilterMode.ALL`: AND logic — a profile passes only if it matches
          both the affiliation AND the email domain. More restrictive, reduces
          false positives.

        When no email is provided, both modes behave identically and only
        affiliation filtering is applied.

    Args:
            name:        Full name of the author to search for.
            affiliation: Institutional affiliation to filter by (e.g. 'MIT CSAIL').
            email:       Optional email address used as an additional filter.
            mode:        Filtering mode controlling how affiliation and email
                         filters are combined. Defaults to `FilterMode.ANY`.
            client:      An authenticated `OpenReviewClient` instance. If not
                         provided, one is created via `create_client()` using
                         the `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD`
                         environment variables.

    Returns:
            A sorted list of OpenReview profile ID strings (e.g. ['~Jane_Smith1'])
            for profiles matching the filters. Returns an empty list if no
            profiles match. Returns None if the client cannot be created or
            the search fails, distinguishing infrastructure failures from
            legitimate empty results.

    Example:
    ```pycon
    >>> from candidex.openreview import find_author_profile_ids
    >>> ids = find_author_profile_ids(
    ...     name="Jane Smith",
    ...     affiliation="MIT CSAIL",
    ...     email="jane@mit.edu",
    ... )  # doctest: +SKIP
    >>> print(ids)  # doctest: +SKIP
    ['~Jane_Smith1']

    ```
    """
    client = client or create_client()
    if client is None:
        logger.warning("No OpenReview client available, cannot find profile for %s.", name)
        return None

    profiles = search_profiles_by_name(name, client=client)
    if profiles is None:
        logger.warning("Profile search failed for %s.", name)
        return None

    logger.debug("Found %d profile(s) for %s.", len(profiles), name)

    by_affiliation = set(filter_profiles_by_affiliation(profiles, affiliation=affiliation))

    if email is None:
        filtered = list(by_affiliation)
    elif mode == FilterMode.ALL:
        by_email = set(filter_profiles_by_email(list(by_affiliation), email=email))
        filtered = list(by_email)
    else:
        by_email = set(filter_profiles_by_email(profiles, email=email))
        filtered = get_unique_profiles([list(by_affiliation | by_email)])

    profile_ids = sorted(p.id for p in filtered)
    logger.debug(
        "Found %d matching profile(s) for %s (%s) [mode=%s]: %s.",
        len(profile_ids),
        name,
        affiliation,
        mode,
        profile_ids,
    )
    return profile_ids
