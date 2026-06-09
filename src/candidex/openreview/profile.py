r"""Contain utilities for finding OpenReview profiles."""

from __future__ import annotations

__all__ = [
    "extract_profiles_by_author",
    "extract_profiles_by_id",
    "fetch_profile_by_id",
    "get_unique_profiles",
    "load_or_fetch_profile_by_author",
    "load_or_fetch_profile_by_id",
    "log_profiles_by_author_stats",
]

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from iden.io import load_json, save_json
from openreview import OpenReviewException, Profile

from candidex.openreview.client import create_client
from candidex.utils.progressbar import make_progressbar

if TYPE_CHECKING:
    from pathlib import Path

    from openreview.api import OpenReviewClient

    from candidex.author import Author


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
        client: An authenticated `OpenReviewClient` instance. If not
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


def load_or_fetch_profile_by_author(
    author: Author,
    profile_id: str,
    profiles_dir: Path,
    client: OpenReviewClient | None = None,
) -> tuple[Author, str, Profile | None]:
    """Load a cached OpenReview profile or fetch it from the API.

    Checks if a cache file exists for the profile ID. If so, loads and
    returns the cached result without querying the API. Otherwise fetches
    the profile from OpenReview and saves it to disk for future calls.

    Args:
        author: The `Author` associated with the profile ID.
        profile_id: The OpenReview profile ID to fetch (e.g. '~Jane_Smith1').
        profiles_dir: Directory where profile JSON files are stored.
                      Must already exist.
        client: An authenticated `OpenReviewClient` instance. If not
                     provided, one is created via `create_client()` using
                     the `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD`
                     environment variables.

    Returns:
        A tuple of (author, profile_id, profile) where profile is the
            `Profile` object, or None if the client cannot be created or the
            fetch failed.

    Example:
        ```pycon
        >>> from candidex.author import Author
        >>> from candidex.openreview import load_or_fetch_profile_by_author
        >>> author = Author.from_raw("Jane Smith", ["MIT"])
        >>> result = load_or_fetch_profile_by_author(
        ...     author, "~Jane_Smith1", Path("data/profiles")
        ... )  # doctest: +SKIP

        ```
    """
    path = profiles_dir / f"{profile_id}.json"

    if path.is_file():
        logger.debug("Loading cached profile for %s.", profile_id)
        return author, profile_id, Profile.from_json(load_json(path))

    resolved_client = client or create_client()
    if resolved_client is None:
        logger.warning("No OpenReview client available, cannot fetch profile %s.", profile_id)
        return author, profile_id, None

    profile = fetch_profile_by_id(profile_id, client=resolved_client)

    if profile is not None:
        save_json(profile.to_json(), path)
        logger.debug("Saved profile for %s.", profile_id)
    else:
        logger.warning("Profile fetch failed for %s.", profile_id)

    return author, profile_id, profile


def extract_profiles_by_author(
    profile_ids_by_author: dict[Author, list[str] | None],
    profiles_dir: Path,
    client: OpenReviewClient | None = None,
    max_workers: int = 4,
) -> dict[Author, list[Profile] | None]:
    """Fetch and save OpenReview profiles for a dictionary of authors
    and profile IDs.

    For each author and their associated profile IDs, fetches the full
    OpenReview profile. Results are cached to disk as individual JSON files
    named by profile ID, making the function safe to call repeatedly and
    resilient to interruptions. Profiles whose cache file already exists
    are skipped without querying the API.

    Authors with no profile IDs (empty list or None) are skipped. Fetches
    are performed concurrently using a thread pool to reduce total wall time.

    Args:
        profile_ids_by_author: A dictionary mapping each `Author` to their
                               list of OpenReview profile ID strings, or None
                               if the lookup previously failed. As returned by
                               `extract_profile_ids_by_author`.
        profiles_dir: Directory where profile JSON files will be saved.
                               Created automatically if it does not exist.
                               Each file is named `{profile_id}.json`.
        client: An authenticated `OpenReviewClient` instance. If
                               not provided, one is created via `create_client()`
                               using environment variables. The same client is
                               reused across all threads.
        max_workers: Maximum number of concurrent threads. Defaults
                               to 4. Reduce if hitting OpenReview API rate limits.

    Returns:
        A dictionary mapping each `Author` to their list of fetched `Profile`
            objects. Authors with no profile IDs or failed fetches are mapped to
            an empty list.

    Example:
        ```pycon
        >>> from candidex.author import Author
        >>> from candidex.openreview import extract_profiles_by_author
        >>> profile_ids_by_author = {Author.from_raw("Jane Smith", ["MIT"]): ["~Jane_Smith1"]}
        >>> profiles = extract_profiles_by_author(
        ...     profile_ids_by_author, Path("data/profiles")
        ... )  # doctest: +SKIP

        ```
    """
    total_ids = sum(len(ids) for ids in profile_ids_by_author.values() if ids is not None)
    logger.info(
        "Fetching %d OpenReview profiles for %d authors with %d threads...",
        total_ids,
        len(profile_ids_by_author),
        max_workers,
    )

    client = client or create_client()
    if client is None:
        logger.warning("No OpenReview client available, cannot fetch profiles.")
        return dict.fromkeys(profile_ids_by_author)

    profiles_dir.mkdir(parents=True, exist_ok=True)
    profiles_by_author: dict[Author, list[Profile]] = {
        author: [] for author in profile_ids_by_author
    }

    with make_progressbar() as progress:
        task = progress.add_task("Fetching OpenReview profiles", total=total_ids)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    load_or_fetch_profile_by_author, author, profile_id, profiles_dir, client
                ): (
                    author,
                    profile_id,
                )
                for author, profile_ids in profile_ids_by_author.items()
                if profile_ids is not None
                for profile_id in profile_ids
            }
            for future in as_completed(futures):
                author, _, profile = future.result()
                if profile is not None:
                    profiles_by_author[author].append(profile)
                progress.advance(task)

    resolved = sum(1 for profiles in profiles_by_author.values() if profiles)
    logger.info(
        "Profile fetch complete. %d/%d authors have at least one profile.",
        resolved,
        len(profile_ids_by_author),
    )
    return profiles_by_author


def load_or_fetch_profile_by_id(
    profile_id: str,
    profiles_dir: Path,
    client: OpenReviewClient | None = None,
) -> tuple[str, Profile | None]:
    """Load a cached OpenReview profile or fetch it from the API by ID.

    Checks if a cache file exists for the profile ID. If so, loads and
    returns the cached result without querying the API. Otherwise fetches
    the profile from OpenReview and saves it to disk for future calls.

    Args:
        profile_id: The OpenReview profile ID to fetch
                      (e.g. '~Jane_Smith1').
        profiles_dir: Directory where profile JSON files are stored.
                      Must already exist.
        client: An authenticated `OpenReviewClient` instance. If not
                      provided, one is created via `create_client()` using
                      the `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD`
                      environment variables.

    Returns:
        A tuple of (profile_id, profile) where profile is the `Profile`
            object, or None if the client cannot be created or the fetch failed.

    Example:
        ```pycon
        >>> result = load_or_fetch_profile_by_id(
        ...     "~Jane_Smith1", Path("data/profiles")
        ... )  # doctest: +SKIP
        >>> result[1].id  # doctest: +SKIP
        '~Jane_Smith1'

        ```
    """
    path = profiles_dir / f"{profile_id}.json"

    if path.is_file():
        logger.debug("Loading cached profile for %s.", profile_id)
        return profile_id, Profile.from_json(load_json(path))

    resolved_client = client or create_client()
    if resolved_client is None:
        logger.warning("No OpenReview client available, cannot fetch profile %s.", profile_id)
        return profile_id, None

    profile = fetch_profile_by_id(profile_id, client=resolved_client)

    if profile is not None:
        save_json(profile.to_json(), path)
        logger.debug("Saved profile for %s.", profile_id)
    else:
        logger.warning("Profile fetch failed for %s.", profile_id)

    return profile_id, profile


def extract_profiles_by_id(
    profile_ids: list[str],
    profiles_dir: Path,
    client: OpenReviewClient | None = None,
    max_workers: int = 4,
) -> dict[str, Profile | None]:
    """Fetch and save OpenReview profiles for a list of profile IDs.

    For each profile ID, fetches the full OpenReview profile and caches
    it to disk. Profiles whose cache file already exists are skipped
    without querying the API, making the function safe to call repeatedly
    and resilient to interruptions. Fetches are performed concurrently
    using a thread pool to reduce total wall time.

    Args:
        profile_ids: List of OpenReview profile ID strings to fetch
                      (e.g. ['~Jane_Smith1', '~John_Doe1']).
        profiles_dir: Directory where profile JSON files will be saved.
                      Created automatically if it does not exist. Each
                      file is named `{profile_id}.json`.
        client: An authenticated `OpenReviewClient` instance. If not
                      provided, one is created via `create_client()` using
                      the `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD`
                      environment variables. The same client is reused
                      across all threads.
        max_workers: Maximum number of concurrent threads. Defaults to 4.
                      Reduce if hitting OpenReview API rate limits.

    Returns:
        A dictionary mapping each profile ID string to its `Profile` object,
            or None if the fetch failed. Duplicate profile IDs are deduplicated
            before fetching.

    Example:
        ```pycon
        >>> profiles = extract_profiles_by_id(
        ...     ["~Jane_Smith1", "~John_Doe1"],
        ...     Path("data/profiles"),
        ... )  # doctest: +SKIP
        >>> profiles["~Jane_Smith1"].id  # doctest: +SKIP
        '~Jane_Smith1'

        ```
    """
    unique_ids = list(dict.fromkeys(profile_ids))
    logger.info(
        "Fetching %d OpenReview profiles with %d threads...",
        len(unique_ids),
        max_workers,
    )

    client = client or create_client()
    if client is None:
        logger.warning("No OpenReview client available, cannot fetch profiles.")
        return dict.fromkeys(unique_ids)

    profiles_dir.mkdir(parents=True, exist_ok=True)
    profiles: dict[str, Profile | None] = {}

    with make_progressbar() as progress:
        task = progress.add_task("Fetching OpenReview profiles", total=len(unique_ids))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    load_or_fetch_profile_by_id, profile_id, profiles_dir, client
                ): profile_id
                for profile_id in unique_ids
            }
            for future in as_completed(futures):
                profile_id, profile = future.result()
                profiles[profile_id] = profile
                progress.advance(task)

    resolved = sum(1 for p in profiles.values() if p is not None)
    logger.info(
        "Profile fetch complete. %d/%d profiles fetched successfully.",
        resolved,
        len(unique_ids),
    )
    return profiles


def log_profiles_by_author_stats(
    profiles_by_author: dict[Author, list[Profile] | None],
) -> None:
    """Log statistics about the distribution of OpenReview profiles per
    author.

    Summarises the profile lookup results at INFO level, breaking down authors
    by how many profiles were found. Useful for assessing the quality of the
    profile matching step before downstream processing.

    Args:
        profiles_by_author: A dictionary mapping each `Author` to their list
                            of `Profile` objects, or None if the lookup failed.

    Example:
        >>> from candidex.openreview.profile import log_profiles_by_author_stats
        >>> log_profiles_by_author_stats({})
    """
    total = len(profiles_by_author)
    if total == 0:
        logger.info("No authors to report profile stats for.")
        return

    missing = sum(1 for profiles in profiles_by_author.values() if profiles is None)
    empty = sum(
        1 for profiles in profiles_by_author.values() if profiles is not None and len(profiles) == 0
    )
    single = sum(
        1 for profiles in profiles_by_author.values() if profiles is not None and len(profiles) == 1
    )
    two = sum(
        1 for profiles in profiles_by_author.values() if profiles is not None and len(profiles) == 2
    )
    three_or_more = sum(
        1 for profiles in profiles_by_author.values() if profiles is not None and len(profiles) >= 3
    )

    counts = [len(profiles) for profiles in profiles_by_author.values() if profiles is not None]
    average = sum(counts) / len(counts) if counts else 0.0

    def pct(n: int) -> float:
        return n / total * 100

    logger.info(
        "Profile stats for %d authors:\n"
        "  - Missing (None):        %d (%.1f%%)\n"
        "  - Empty list:            %d (%.1f%%)\n"
        "  - Single profile:        %d (%.1f%%)\n"
        "  - Two profiles:          %d (%.1f%%)\n"
        "  - Three or more:         %d (%.1f%%)\n"
        "  - Average per author:    %.2f (excluding None)",
        total,
        missing,
        pct(missing),
        empty,
        pct(empty),
        single,
        pct(single),
        two,
        pct(two),
        three_or_more,
        pct(three_or_more),
        average,
    )
