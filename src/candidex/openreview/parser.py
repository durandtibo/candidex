r"""Contain openreview utilities to process the history profile."""

from __future__ import annotations

__all__ = ["parse_names_and_history_profile"]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openreview import Profile


def parse_names_and_history_profile(profile: Profile) -> dict[str, Any]:
    """Extract key metadata from an OpenReview profile.

    Parses the profile into a simplified dictionary containing all
    associated names and the career history. Useful for downstream
    processing without carrying the full `Profile` object.

    Args:
        profile: An `openreview.Profile` object as returned by
                 `fetch_profile_by_id` or the OpenReview API.

    Returns:
        A dictionary with the following keys:
            - names    (list[dict]): List of name entries from the profile,
                                     each containing fields such as
                                     'fullname', 'first', 'last'.
            - history  (list[dict]): List of career history entries, each
                                     containing fields such as 'position'
                                     and 'institution'.

    Example:
        ```pycon
        >>> import openreview
        >>> from candidex.openreview import parse_names_and_history_profile
        >>> profile = openreview.Profile(
        ...     id="~Jane_Smith1",
        ...     content={
        ...         "names": [{"fullname": "Jane Smith", "first": "Jane", "last": "Smith"}],
        ...         "history": [{"position": "PhD Student", "institution": {"name": "MIT"}}],
        ...     },
        ... )
        >>> parse_names_and_history_profile(profile)
        {'names': [{'fullname': 'Jane Smith', 'first': 'Jane', 'last': 'Smith'}], 'history': [{'position': 'PhD Student', 'institution': {'name': 'MIT'}}]}

        ```
    """
    return {
        "names": profile.content.get("names", []),
        "history": profile.content.get("history", []),
    }
