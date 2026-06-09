r"""Contain utilities for sorting authors."""

from __future__ import annotations

__all__ = ["sort_authors"]

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from candidex.author.author import Author

logger: logging.Logger = logging.getLogger(__name__)


def sort_authors(authors: Sequence[Author], *, reverse: bool = False) -> list[Author]:
    """Return a sorted list of authors by name, then by affiliations.

    Authors are sorted alphabetically by name. When two authors share the
    same name, they are further sorted by their affiliations string
    (as returned by `format_affiliations`). Sorting is case-insensitive.

    Args:
        authors: A sequence of `Author` objects to sort.
        reverse: If True, sort in descending order. Defaults to False
            (ascending).

    Returns:
        A new sorted list of `Author` objects. The original sequence is
            not modified. Returns an empty list if the input is empty.

    Example:
        ```pycon
        >>> from candidex.author import Author, sort_authors
        >>> authors = [
        ...     Author.from_raw("John Doe", ["Stanford"]),
        ...     Author.from_raw("Jane Smith", ["MIT"]),
        ...     Author.from_raw("Jane Smith", ["CMU"]),
        ... ]
        >>> [a.name for a in sort_authors(authors)]
        ['Jane Smith', 'Jane Smith', 'John Doe']
        >>> [a.name for a in sort_authors(authors, reverse=True)]
        ['John Doe', 'Jane Smith', 'Jane Smith']

        ```
    """
    return sorted(
        authors,
        key=lambda a: (a.name.lower(), a.format_affiliations().lower()),
        reverse=reverse,
    )
