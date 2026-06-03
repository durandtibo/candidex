r"""Contain utilities for finding OpenReview profiles."""

from __future__ import annotations

__all__ = [
    "FilterMode",
    "extract_profile_ids",
    "find_author_profile_ids",
    "load_or_fetch_profile_ids",
    "log_profile_ids_stats",
]

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import TYPE_CHECKING

from iden.io import load_json, save_json

from candidex.openreview.client import create_client
from candidex.openreview.filtering import (
    filter_profiles_by_affiliation,
    filter_profiles_by_email,
)
from candidex.openreview.profile import get_unique_profiles
from candidex.openreview.search import search_profiles_by_name
from candidex.utils.progressbar import make_progressbar

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from openreview.api import OpenReviewClient

    from candidex.author import Author


logger: logging.Logger = logging.getLogger(__name__)


class FilterMode(str, Enum):
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


def load_or_fetch_profile_ids(
    author: Author,
    profile_ids_dir: Path,
    client: OpenReviewClient | None = None,
) -> tuple[Author, list[str] | None]:
    """Load cached profile IDs or fetch them from OpenReview for a
    single author.

    Checks if a cache file exists for the author. If so, loads and returns
    the cached result without querying the API. Otherwise queries OpenReview
    and saves the result to disk for future calls.

    If no client is provided, one is created automatically using credentials
    from the environment. Returns `(author, None)` if the client cannot be
    created or the lookup fails.

    Args:
        author:           The `Author` to look up.
        profile_ids_dir: Directory where profile ID JSON files are stored.
                          Must already exist.
        client:           An authenticated `OpenReviewClient` instance. If
                          not provided, one is created via `create_client()`
                          using the `OPENREVIEW_USERNAME` and
                          `OPENREVIEW_PASSWORD` environment variables.

    Returns:
        A tuple of (author, profile_ids) where profile_ids is a list of
        profile ID strings, an empty list if none were found, or None if
        the client cannot be created or the lookup failed.

    Example:
        >>> from candidex.author import Author
        >>> author = Author.from_raw("Jane Smith", ["MIT"])
        >>> result = load_or_fetch_profile_ids(author, Path("data/profile_ids"))  # doctest: +SKIP
        >>> result[1] # doctest: +SKIP
        ['~Jane_Smith1']
    """
    path = profile_ids_dir / f"{author.hash()}.json"

    if path.is_file():
        logger.debug("Loading cached profile IDs for %s.", author)
        return author, load_json(path)

    resolved_client = client or create_client()
    if resolved_client is None:
        logger.warning("No OpenReview client available, cannot find profile IDs for %s.", author)
        return author, None

    profile_ids = find_author_profile_ids(
        name=author.name,
        affiliation=author.format_affiliations(),
        email=author.email,
        mode=FilterMode.ANY,
        client=resolved_client,
    )

    if profile_ids is not None:
        save_json(profile_ids, path)
        logger.debug("Saved %d profile ID(s) for %s.", len(profile_ids), author)
    else:
        logger.warning("Profile ID lookup failed for %s.", author)

    return author, profile_ids


def extract_profile_ids(
    authors: Sequence[Author],
    profile_ids_dir: Path,
    client: OpenReviewClient | None = None,
    max_workers: int = 4,
) -> dict[Author, list[str] | None]:
    """Find and save OpenReview profile IDs for a sequence of authors.

    For each author, searches OpenReview for matching profile IDs using
    their name, affiliations, and email. Results are cached to disk as
    individual JSON files named by the author's hash, making the function
    safe to call repeatedly and resilient to interruptions.

    Authors whose cache file already exists are skipped without querying
    the API. Uses OR logic to combine affiliation and email filters,
    maximising recall. Lookups are performed concurrently using a thread
    pool to reduce total wall time.

    Args:
        authors:           Sequence of `Author` objects to look up.
        profile_ids_dir:  Directory where profile ID JSON files will be
                           saved. Created automatically if it does not exist.
                           Each file is named `{author.hash()}.json`.
        client:            An authenticated `OpenReviewClient` instance. If
                           not provided, one is created via `create_client()`
                           using environment variables. The same client is
                           reused across all threads.
        max_workers:       Maximum number of concurrent threads. Defaults to
                           4. Reduce if hitting OpenReview API rate limits.

    Returns:
        A dictionary mapping each `Author` to their list of profile ID
        strings, or None if the lookup failed. Authors loaded from cache
        are included alongside freshly queried ones.

    Example:
        ```pycon
        >>> from candidex.author import Author
        >>> authors = [Author.from_raw("Jane Smith", ["MIT"])]
        >>> results = extract_profile_ids(authors, Path("data/profile_ids"))  # doctest: +SKIP
        >>> results[authors[0]]  # doctest: +SKIP
        ['~Jane_Smith1']

        ```
    """
    logger.info(
        "Extracting OpenReview profile IDs for %d authors with %d threads...",
        len(authors),
        max_workers,
    )
    client = client or create_client()
    if client is None:
        logger.warning("No OpenReview client available, cannot find profile IDs.")
        return {}

    profile_ids_dir.mkdir(parents=True, exist_ok=True)
    results: dict[Author, list[str] | None] = {}

    with make_progressbar() as progress:
        task = progress.add_task("Finding OpenReview profile IDs", total=len(authors))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(load_or_fetch_profile_ids, author, profile_ids_dir, client): author
                for author in authors
            }
            for future in as_completed(futures):
                author, profile_ids = future.result()
                results[author] = profile_ids
                progress.advance(task)

    resolved = sum(1 for ids in results.values() if ids is not None and len(ids) > 0)
    logger.info(
        "Profile ID extraction complete. %d/%d authors resolved.",
        resolved,
        len(authors),
    )
    return results


def log_profile_ids_stats(profile_ids_by_author: dict[Author, list[str] | None]) -> None:
    """Log a summary of profile ID lookup results.

    Breaks down the results into four categories and logs them at INFO level:
    - Single match: exactly one profile ID found.
    - Multiple matches: more than one profile ID found.
    - No match: lookup succeeded but no profiles were found.
    - Failed: lookup failed entirely (None).

    Args:
        profile_ids_by_author: A dictionary mapping each `Author` to their list of profile
                 ID strings, or None if the lookup failed, as returned by
                 `extract_profile_ids`.

    Example:
        ```pycon
        >>> from candidex.author import Author
        >>> from candidex.openreview import log_profile_ids_stats
        >>> results = {
        ...     Author.from_raw("Jane Smith", ["MIT"]): ["~Jane_Smith1"],
        ...     Author.from_raw("John Doe", ["Stanford"]): [],
        ...     Author.from_raw("Alice Brown", ["CMU"]): None,
        ... }
        >>> log_profile_ids_stats(results)

        ```
    """
    total = len(profile_ids_by_author)
    profile_ids_list = profile_ids_by_author.values()

    single = sum(1 for ids in profile_ids_list if ids is not None and len(ids) == 1)
    multiple = sum(1 for ids in profile_ids_list if ids is not None and len(ids) > 1)
    empty = sum(1 for ids in profile_ids_list if ids is not None and len(ids) == 0)
    failed = sum(1 for ids in profile_ids_list if ids is None)

    logger.info(
        "Profile ID lookup complete for %d authors — "
        "%d single match, %d multiple matches, %d no match, %d failed.",
        total,
        single,
        multiple,
        empty,
        failed,
    )
