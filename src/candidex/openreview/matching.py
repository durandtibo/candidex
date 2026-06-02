r"""Contain matching functions."""

from __future__ import annotations

__all__ = ["do_affiliations_match"]

from candidex.utils.string import is_substring_match, remove_spaces


def do_affiliations_match(a: str, b: str) -> bool:
    """Return True if two affiliation strings are considered a match.

    Two affiliations match if either is a substring of the other, with or
    without spaces. This handles common formatting differences between
    affiliation sources, for example:

    - Partial matches: 'MIT' matches 'MIT CSAIL, Cambridge, MA, USA'
    - Spacing differences: 'EastChinaNormalUniversity' matches
      'East China Normal University'

    Comparison is case-insensitive in both checks.

    Args:
        a: First affiliation string to compare.
        b: Second affiliation string to compare.

    Returns:
        True if the affiliations are considered a match, False otherwise.

    Example:
        ```pycon
        >>> from candidex.openreview import do_affiliations_match
        >>> do_affiliations_match("MIT", "MIT CSAIL, Cambridge, MA, USA")
        True
        >>> do_affiliations_match("EastChinaNormalUniversity", "East China Normal University")
        True
        >>> do_affiliations_match("Stanford", "MIT CSAIL")
        False

        ```
    """
    a = a.strip().lower()
    b = b.strip().lower()
    return is_substring_match(a, b) or is_substring_match(remove_spaces(a), remove_spaces(b))
