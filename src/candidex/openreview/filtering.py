r"""Contain filtering utilities for OpenReview."""

from __future__ import annotations

__all__ = ["filter_profiles_by_affiliation"]

import logging
from typing import TYPE_CHECKING

from candidex.openreview.matching import do_affiliations_match, does_email_match_domain

if TYPE_CHECKING:
    from openreview import Profile


logger: logging.Logger = logging.getLogger(__name__)


def filter_profiles_by_affiliation(
    profiles: list[Profile],
    affiliation: str,
    email: str | None = None,
) -> list[Profile]:
    """Filter OpenReview profiles by affiliation and optionally by email
    domain.

    For each profile, checks whether any institution in the career history
    matches the target affiliation using `do_affiliations_match`. If an email
    is provided, it is checked against two sources:
    - The confirmed email addresses listed on the profile.
    - The institution domains in the career history.

    A profile passes the email filter if the email matches at least one
    confirmed profile email domain OR at least one institution domain.

    Args:
        profiles:    List of OpenReview profiles to filter, as returned by
                     `search_openreview_profiles`.
        affiliation: Target affiliation string to match against each profile's
                     career history (e.g. 'MIT CSAIL', 'Tsinghua University').
        email:       Optional email address. If provided, the domain is matched
                     against both confirmed profile emails and institution
                     domains in the career history. If None, only affiliation
                     is checked.

    Returns:
        A filtered list of `openreview.Profile` objects whose affiliation
            matches the target, and whose email domain matches if provided.
            Returns an empty list if no profiles pass the filter.

    Example:
        >>> from candidex.openreview import filter_profiles_by_affiliation, search_openreview_profiles
        >>> profiles = search_openreview_profiles("Jane Smith") # doctest: +SKIP
        >>> matched = filter_profiles_by_affiliation( # doctest: +SKIP
        ...     profiles,
        ...     affiliation="MIT CSAIL",
        ...     email="jane@csail.mit.edu",
        ... )
    """
    matched = []
    for profile in profiles:
        history = profile.content.get("history", [])

        affiliation_match = any(
            do_affiliations_match(affiliation, entry.get("institution", {}).get("name", None))
            for entry in history
        )
        if not affiliation_match:
            continue

        if email is not None:
            profile_emails = profile.content.get("emails", [])
            email_match = any(
                does_email_match_domain(email, profile_email) for profile_email in profile_emails
            )
            institution_domain_match = any(
                does_email_match_domain(email, entry.get("institution", {}).get("domain", None))
                for entry in history
            )
            if not email_match and not institution_domain_match:
                continue

        matched.append(profile)

    logger.debug(
        "Filtered %d/%d profiles matching affiliation '%s'.",
        len(matched),
        len(profiles),
        affiliation,
    )
    return matched
