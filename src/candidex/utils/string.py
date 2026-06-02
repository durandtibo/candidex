r"""Contain utilities for strings."""

from __future__ import annotations

__all__ = ["is_substring_match"]


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
        >>> is_substring_match("MIT", "MIT CSAIL, Cambridge, MA, USA")
        True
        >>> is_substring_match("Stanford", "MIT CSAIL")
        False
    """
    a_lower = a.lower()
    b_lower = b.lower()
    return a_lower in b_lower or b_lower in a_lower
