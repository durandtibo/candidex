r"""Contain openreview utilities."""

from __future__ import annotations

__all__ = ["fetch_openreview_profile"]

import logging
import time
import urllib
from typing import Any

import httpx
from ddgs import DDGS
from ddgs.exceptions import DDGSException

from candidex.schemas import AuthorAffiliation

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


def find_openreview_profile_url(
    author: AuthorAffiliation,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
) -> str | None:
    """Search for an author's OpenReview profile URL using DuckDuckGo.

    Searches DuckDuckGo for the author's OpenReview profile page using
    their name and affiliation as search terms. Returns the first result
    URL that is a valid OpenReview profile URL as determined by
    `is_openreview_profile_url`.

    Args:
        author:         An `AuthorAffiliation` object containing the author's
                        name and known affiliations.
        max_retries:    Maximum number of retry attempts on connectivity
                        errors. Defaults to 3.
        backoff_factor: Multiplier for wait time between retries.
                        Defaults to 2.0.

    Returns:
        The OpenReview profile URL (e.g.
        'https://openreview.net/profile?id=~Thibaut_Durand1') if found,
        or None if no matching profile could be located after all retries.

    Example:
        >>> author = AuthorAffiliation(
        ...     author="Thibaut Durand",
        ...     affiliations=["concordia university"],
        ... )
        >>> url = find_openreview_profile_url(author)
        >>> if url:
        ...     print(url)
        'https://openreview.net/profile?id=~Thibaut_Durand1'
    """
    affiliation_str = ", ".join(author.affiliations) if author.affiliations else ""
    query = f'openreview profile "{author.author}" "{affiliation_str}" site:openreview.net/profile'
    # query = f'"{author.author}" "{affiliation_str}" site:openreview.net/profile'

    logger.debug("Searching for OpenReview profile: %s (%s).", author.author, affiliation_str)

    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=5)

            if results:
                for r in results:
                    url = r["href"]
                    if is_openreview_profile_url(url):
                        logger.debug(
                            "Found OpenReview profile for %s: %s.",
                            author.author,
                            url,
                        )
                        return url

            logger.debug("No OpenReview profile found for %s.", author.author)
            return None

        except DDGSException as e:
            wait = backoff_factor**attempt
            if attempt < max_retries - 1:
                logger.debug(
                    "Search failed (attempt %d/%d): %s. Retrying in %.0fs...",
                    attempt + 1,
                    max_retries,
                    e,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.warning(
                    "OpenReview profile search failed after %d attempts for %s: %s.",
                    max_retries,
                    author.author,
                    e,
                )

    return None


if __name__ == "__main__":
    from rich.pretty import pprint

    # pprint(fetch_openreview_profile_content("https://openreview.net/profile?id=~Thibaut_Durand1"))
    # pprint(
    #     fetch_openreview_profile_content(
    #         "https://openreview.net/profile?id=~Sepidehsadat_Hosseini2"
    #     )
    # )
    # pprint(
    #     fetch_openreview_profile_content("https://openreview.net/profile?id=~Yasutaka_Furukawa1")
    # )

    pprint(
        find_openreview_profile_url(
            AuthorAffiliation(author="Thibaut Durand", affiliations=["SFU"], email=None)
        )
    )
