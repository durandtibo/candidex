r"""Contain matching functions."""

from __future__ import annotations

__all__ = ["do_affiliations_match", "does_email_match_domain"]

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


def does_email_match_domain(email: str | None, domain: str | None) -> bool:
    """Return True if the email address belongs to the given domain.

    Checks whether the email's domain part matches the provided domain,
    using case-insensitive substring matching to handle subdomains and
    partial domain specifications. For example, 'mit.edu' matches
    'user@csail.mit.edu'.

    The domain may optionally start with '@' (e.g. '@mit.edu' or 'mit.edu'
    are treated equivalently).

    Args:
        email:  The email address to check (e.g. 'jane.smith@csail.mit.edu').
                Returns False if None.
        domain: The domain to match against (e.g. 'mit.edu' or '@mit.edu').
                Returns False if None.

    Returns:
        True if the email domain contains or is contained by the given
        domain, False otherwise. Returns False if either argument is None
        or if the email does not contain an '@' symbol.

    Example:
        ```pycon
        >>> from candidex.openreview import does_email_match_domain
        >>> does_email_match_domain("jane.smith@csail.mit.edu", "mit.edu")
        True
        >>> does_email_match_domain("jane.smith@csail.mit.edu", "@mit.edu")
        True
        >>> does_email_match_domain("jane.smith@stanford.edu", "mit.edu")
        False

        ```
    """
    if email is None or domain is None:
        return False
    if "@" not in email:
        return False

    email_domain = email.split("@")[-1].lower()
    domain = domain.lstrip("@").lower()

    return is_substring_match(email_domain, domain)
