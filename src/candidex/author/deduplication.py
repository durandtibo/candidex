r"""Contain utilities for deduplicating authors."""

from __future__ import annotations

__all__ = ["deduplicate_authors"]

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from candidex.author.author import Author

logger: logging.Logger = logging.getLogger(__name__)


def deduplicate_authors(authors: Sequence[Author]) -> list[Author]:
    """Return a deduplicated list of authors preserving first occurrence
    order.

    Deduplicates by identity across all fields — two `Author` objects are
    considered equal only if their name, affiliations, and email are all
    identical. Authors with the same name but different affiliations or email
    are treated as distinct and both kept.

    Since `Author` is a frozen dataclass, equality and hashing are based on
    all fields automatically.

    Args:
        authors: A list of `Author` objects, potentially containing
            duplicate entries.

    Returns:
        A list of `Author` objects with exact duplicates removed, preserving
            the order of first occurrence. Returns an empty list if the input
            is empty.

    Example:
        ```pycon
        >>> from candidex.author import Author, deduplicate_authors
        >>> authors = [
        ...     Author.from_raw("Jane Smith", ["MIT"]),
        ...     Author.from_raw("Jane Smith", ["MIT"]),
        ...     Author.from_raw("Jane Smith", ["MIT CSAIL"]),
        ... ]
        >>> [a.name for a in deduplicate_authors(authors)]
        ['Jane Smith', 'Jane Smith']

        ```
    """
    seen: set[Author] = set()
    unique: list[Author] = []
    for author in authors:
        if author not in seen:
            seen.add(author)
            unique.append(author)

    logger.debug(
        "Deduplicated authors: %d before, %d after, %d removed.",
        len(authors),
        len(unique),
        len(authors) - len(unique),
    )
    return unique
