r"""Contain filtering utilities for OpenReview."""

from __future__ import annotations

__all__ = ["filter_profiles_by_affiliation", "filter_profiles_by_email"]

import logging
from typing import TYPE_CHECKING

from candidex.openreview.matching import do_affiliations_match, does_email_match_domain

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openreview import Profile


logger: logging.Logger = logging.getLogger(__name__)


def filter_profiles_by_affiliation(
    profiles: Sequence[Profile],
    affiliation: str,
) -> list[Profile]:
    """Filter OpenReview profiles by affiliation.

    Checks whether any institution in each profile's career history matches
    the target affiliation using `do_affiliations_match`.

    Args:
        profiles:    List of OpenReview profiles to filter.
        affiliation: Target affiliation string to match against each profile's
                     career history (e.g. 'MIT CSAIL', 'Tsinghua University').

    Returns:
        A filtered list of `openreview.Profile` objects whose career history
            contains at least one institution matching the target affiliation.
            Returns an empty list if no profiles pass the filter.

    Example:
        ```pycon
        >>> from candidex.openreview import filter_profiles_by_affiliation
        >>> >>> profiles = [...] # doctest: +SKIP
        >>> matched = filter_profiles_by_affiliation(
        ...     profiles,
        ...     affiliation="MIT CSAIL",
        ... ) # doctest: +SKIP

        ```
    """
    matched = [
        profile
        for profile in profiles
        if any(
            do_affiliations_match(affiliation, entry.get("institution", {}).get("name", ""))
            for entry in profile.content.get("history", [])
        )
    ]
    logger.debug(
        "Filtered %d/%d profiles matching affiliation '%s'.",
        len(matched),
        len(profiles),
        affiliation,
    )
    return matched


def filter_profiles_by_email(profiles: Sequence[Profile], email: str) -> list[Profile]:
    """Filter OpenReview profiles by email domain.

    Checks the email against two sources on each profile:
    - The confirmed email addresses listed on the profile.
    - The institution domains in the career history.

    A profile passes if the email domain matches at least one confirmed
    profile email OR at least one institution domain.

    Args:
        profiles: List of OpenReview profiles to filter.
        email:    Email address whose domain is matched against confirmed
                  profile emails and institution domains in the career history.

    Returns:
        A filtered list of `openreview.Profile` objects whose email domain
            matches at least one confirmed email or institution domain.
            Returns an empty list if no profiles pass the filter.

    Example:
        ```pycon
        >>> from candidex.openreview import filter_profiles_by_email
        >>> profiles = [...] # doctest: +SKIP
        >>> matched = filter_profiles_by_email(
        ...     profiles,
        ...     email="jane@csail.mit.edu",
        ... ) # doctest: +SKIP

        ```
    """

    def _profile_matches_email(profile: Profile) -> bool:
        profile_emails = profile.content.get("emails", [])
        if any(does_email_match_domain(email, profile_email) for profile_email in profile_emails):
            return True
        history = profile.content.get("history", [])
        return any(
            does_email_match_domain(email, entry.get("institution", {}).get("domain", ""))
            for entry in history
        )

    matched = [profile for profile in profiles if _profile_matches_email(profile)]
    logger.debug(
        "Filtered %d/%d profiles matching email domain '%s'.",
        len(matched),
        len(profiles),
        email,
    )
    return matched
