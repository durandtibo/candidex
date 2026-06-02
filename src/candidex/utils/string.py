r"""Contain utilities for strings."""

from __future__ import annotations

__all__ = ["is_substring_match", "remove_spaces"]

import unicodedata


def is_substring_match(a: str, b: str) -> bool:
    """Return True if either string is a substring of the other.

    Comparison is case-insensitive. Useful for matching affiliations or
    names where one source may contain a more complete string than the other
    (e.g. 'MIT' in 'MIT CSAIL, Cambridge, MA, USA').

    Args:
        a: First string to compare.
        b: Second string to compare.

    Returns:
        True if `a` is a substring of `b` or `b` is a substring of `a`,
            False otherwise.

    Example:
        ```pycon
        >>> from candidex.utils.string import is_substring_match
        >>> is_substring_match("MIT", "MIT CSAIL, Cambridge, MA, USA")
        True
        >>> is_substring_match("Stanford", "MIT CSAIL")
        False

        ```
    """
    a_lower = a.lower()
    b_lower = b.lower()
    return a_lower in b_lower or b_lower in a_lower


def remove_spaces(s: str) -> str:
    """Return the string with all whitespace characters removed.

    Removes all whitespace characters including spaces, tabs, and newlines.
    Useful for normalising names or identifiers before comparison.

    Args:
        s: The string to process.

    Returns:
        The string with all whitespace characters removed.

    Example:
        ```pycon
        >>> from candidex.utils.string import remove_spaces
        >>> remove_spaces("Thibaut Durand")
        'ThibautDurand'
        >>> remove_spaces("  MIT  CSAIL  ")
        'MITCSAIL'

        ```
    """
    return "".join(s.split())


def normalize_unicode(s: str) -> str:
    """Normalize unicode characters to their ASCII equivalents.

    Converts accented characters to their base form (e.g. 'é' -> 'e',
    'ü' -> 'u').
    """
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
