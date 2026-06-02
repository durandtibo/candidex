r"""Contain schemas for author affiliations."""

from __future__ import annotations

__all__ = ["Author"]

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from candidex.utils.string import normalize_unicode

if TYPE_CHECKING:
    from collections.abc import Sequence

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Author:
    """Represents an author with their affiliations and email.

    Designed to be used as a dictionary key or set member — equality and
    hashing are based on all fields. Use `from_raw` to construct instances
    with automatic normalization of unicode characters and whitespace.

    Attributes:
        name:         Full name of the author, unicode-normalized and stripped.
        affiliations: Tuple of institutional affiliations, each unicode-normalized.
                      None if no affiliation information is available.
        email:        Email address of the author, stripped of whitespace.
                      None if not available.
    """

    name: str
    affiliations: tuple[str, ...] | None = None
    email: str | None = None

    def format_affiliations(self, separator: str = "; ") -> str:
        """Return a string representation of the affiliations.

        Args:
            separator: String used to join affiliations. Defaults to '; '.

        Returns:
            A single string of all affiliations joined by the separator.
            Returns an empty string if affiliations is None or empty.

        Example:
            >>> author = Author.from_raw("Jane Smith", ["MIT", "Stanford"], None)
            >>> author.format_affiliations()
            'MIT; Stanford'
        """
        if not self.affiliations:
            return ""
        return separator.join(self.affiliations)

    @classmethod
    def from_raw(
        cls,
        name: str,
        affiliations: Sequence[str] | None = None,
        email: str | None = None,
    ) -> Author:
        """Construct an `Author` with normalized unicode and stripped
        whitespace.

        Normalizes unicode characters (e.g. 'é' → 'e') and strips leading/
        trailing whitespace from the name, each affiliation, and the email.
        Always use this constructor rather than the dataclass constructor
        directly to ensure consistent normalization.

        Args:
            name:         Full name of the author.
            affiliations: List of institutional affiliations, or None.
            email:        Email address of the author, or None.

        Returns:
            A new `Author` instance with normalized fields.

        Raises:
            ValueError: If `name` is empty or whitespace-only.

        Example:
            >>> Author.from_raw("Paul George", ["MIT CSAIL"], "jane@mit.edu")
            Author(name='Universite', affiliations=('MIT CSAIL',), email='jane@mit.edu')
        """
        name = normalize_unicode(name.strip())
        if not name:
            msg = "Author name cannot be empty."
            raise ValueError(msg)
        return cls(
            name=name,
            affiliations=(
                tuple(normalize_unicode(a.strip()) for a in affiliations)
                if affiliations is not None
                else None
            ),
            email=email.strip() if email is not None else None,
        )
