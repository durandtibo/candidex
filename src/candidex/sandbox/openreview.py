r"""Contain openreview utilities."""

from __future__ import annotations

__all__ = ["fetch_openreview_profile"]

import logging
import urllib
from typing import Any

import httpx

logger: logging.Logger = logging.getLogger(__name__)

OPENREVIEW_API_URL = "https://api2.openreview.net/profiles"
OPENREVIEW_PROFILE_PREFIX = "https://openreview.net/profile?id="


def fetch_openreview_profile(url: str, timeout: int = 30) -> dict[str, Any] | None:
    """Fetch and parse an OpenReview profile from a profile page URL.

    Extracts the OpenReview user ID from the profile URL, queries the
    OpenReview REST API directly rather than scraping the HTML page, and
    returns the raw profile data as a dict. The API response is more
    structured and reliable than HTML parsing, containing explicit fields
    for name, affiliations, career history, and relations.

    Args:
        url: Full OpenReview profile URL, e.g.
                     'https://openreview.net/profile?id=~Thibaut_Durand1'.
                     The `id` query parameter is used to query the API.
        timeout:     Request timeout in seconds. Defaults to 30.

    Returns:
        A dict containing the raw profile data from the OpenReview API,
        or None if the profile could not be fetched or does not exist.
        The dict contains keys such as:
            - id          (str):  The OpenReview user ID (e.g. '~Thibaut_Durand1').
            - content     (dict): Profile content including name, history,
                                  expertise, and relations.

    Raises:
        ValueError: If the URL does not contain a valid `id` query parameter.

    Example:
        >>> profile = fetch_openreview_profile(
        ...     "https://openreview.net/profile?id=~Thibaut_Durand1"
        ... )
        >>> if profile:
        ...     print(profile["content"]["names"])
    """
    if not is_openreview_profile_url(url):
        return None

    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)

    if "id" not in params:
        msg = f"No 'id' parameter found in URL: {url}"
        raise ValueError(msg)

    profile_id = params["id"][0]
    logger.debug("Fetching OpenReview profile for ID: %s.", profile_id)

    try:
        response = httpx.get(
            OPENREVIEW_API_URL,
            params={"id": profile_id},
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.warning("OpenReview profile not found: %s.", profile_id)
            return None
        logger.warning("HTTP error fetching OpenReview profile %s: %s", profile_id, e)
        return None
    except httpx.HTTPError as e:
        logger.warning("Failed to fetch OpenReview profile %s: %s", profile_id, e)
        return None

    data = response.json()
    profiles = data.get("profiles", [])

    if not profiles:
        logger.warning("No profile data returned for ID: %s.", profile_id)
        return None

    logger.debug("Successfully fetched OpenReview profile for %s.", profile_id)
    return profiles[0]


def fetch_openreview_profile_content(url: str, timeout: int = 30) -> dict[str, Any] | None:
    """Fetch and parse an OpenReview profile from a profile page URL.

    Extracts the OpenReview user ID from the profile URL, queries the
    OpenReview REST API directly rather than scraping the HTML page, and
    returns the raw profile data as a dict. The API response is more
    structured and reliable than HTML parsing, containing explicit fields
    for name, affiliations, career history, and relations.

    Args:
        url: Full OpenReview profile URL, e.g.
                     'https://openreview.net/profile?id=~Thibaut_Durand1'.
                     The `id` query parameter is used to query the API.
        timeout:     Request timeout in seconds. Defaults to 30.

    Returns:
        A dict containing the raw profile data from the OpenReview API,
        or None if the profile could not be fetched or does not exist.
        The dict contains keys such as:
            - id          (str):  The OpenReview user ID (e.g. '~Thibaut_Durand1').
            - content     (dict): Profile content including name, history,
                                  expertise, and relations.

    Raises:
        ValueError: If the URL does not contain a valid `id` query parameter.

    Example:
        >>> profile = fetch_openreview_profile(
        ...     "https://openreview.net/profile?id=~Thibaut_Durand1"
        ... )
        >>> if profile:
        ...     print(profile["content"]["names"])
    """
    profile = fetch_openreview_profile(url=url, timeout=timeout)
    if not profile:
        return None
    return (
        {"id": profile["id"]}
        | {"names": profile["content"]["names"]}
        | {"history": profile["content"]["history"]}
    )


def is_openreview_profile_url(url: str) -> bool:
    """Return True if the URL is a valid OpenReview profile URL.

    Checks that the URL starts with the expected OpenReview profile prefix
    'https://openreview.net/profile?id='. Does not validate whether the
    profile actually exists.

    Args:
        url: The URL string to check.

    Returns:
        True if the URL is a valid OpenReview profile URL, False otherwise.

    Example:
        >>> is_openreview_profile_url("https://openreview.net/profile?id=~Thibaut_Durand1")
        True
        >>> is_openreview_profile_url("https://openreview.net/forum?id=abc123")
        False
    """
    return url.startswith(OPENREVIEW_PROFILE_PREFIX)
