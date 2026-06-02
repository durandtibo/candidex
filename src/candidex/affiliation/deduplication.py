r"""Contain utilities for deduplicating affiliation data structures."""

from __future__ import annotations

__all__ = ["deduplicate_authors"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from candidex.schemas import AuthorAffiliation


def deduplicate_authors(authors: list[AuthorAffiliation]) -> list[AuthorAffiliation]:
    """Return a deduplicated list of authors preserving first occurrence
    order.

    Deduplicates based on all fields in `AuthorAffiliation`: author name,
    affiliations, and email. Two authors are considered duplicates only if
    all three fields are identical. When the same author appears multiple
    times with different affiliations or email, both entries are kept.

    Args:
        authors: A list of `AuthorAffiliation` objects, potentially containing
                 duplicate entries.

    Returns:
        A list of `AuthorAffiliation` objects with duplicates removed,
        preserving the order of first occurrence. Returns an empty list
        if the input is empty.

    Example:
        ```pycon
        >>> from candidex.affiliation import deduplicate_authors
        >>> from candidex.schemas import AuthorAffiliation
        >>> authors = [
        ...     AuthorAffiliation(author="Jane Smith", affiliations=["MIT"], email=None),
        ...     AuthorAffiliation(author="Jane Smith", affiliations=["MIT"], email=None),
        ...     AuthorAffiliation(author="Jane Smith", affiliations=["MIT CSAIL"], email=None),
        ... ]
        >>> [a.author for a in deduplicate_authors(authors)]
        ['Jane Smith', 'Jane Smith']

        ```
    """
    seen: set[tuple] = set()
    unique: list[AuthorAffiliation] = []
    for author in authors:
        key = (author.author, tuple(author.affiliations), author.email)
        if key not in seen:
            seen.add(key)
            unique.append(author)
    return unique
