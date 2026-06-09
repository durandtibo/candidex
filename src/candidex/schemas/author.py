r"""Contain schemas for authors."""

from __future__ import annotations

__all__ = ["AuthorExtraction"]

import logging
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from candidex.author.author import Author

logger: logging.Logger = logging.getLogger(__name__)

StrippedStr = Annotated[str, StringConstraints(strip_whitespace=True)]


class AuthorExtraction(BaseModel):
    """Represent a single author and their institutional affiliations.

    An author may be affiliated with multiple institutions simultaneously,
    for example a university department and an associated research lab.
    Extractions are typically listed on the first page of an academic paper,
    often linked to author names via superscript numbers or symbols.

    Attributes:
        author:       Full name of the author exactly as it appears on the paper
                      (e.g. 'Jane Smith', 'J. Smith'). Do not normalise or infer
                      missing name parts.
        affiliations: List of institutional affiliations for this author. Each
                      entry should be a complete affiliation string as it appears
                      on the paper (e.g. 'MIT CSAIL, Cambridge, MA, USA').
                      Use an empty list if no affiliation can be determined.
        email:        Email address of the author if explicitly stated on the
                      paper. None if not present or not determinable. Do not
                      infer or construct email addresses from names or affiliations.
    """

    author: StrippedStr = Field(
        description="Full name of the author exactly as it appears on the paper."
    )
    affiliations: list[StrippedStr] = Field(
        description=(
            "List of institutional affiliations for this author. Each entry is a complete "
            "affiliation string as it appears on the paper. Empty list if none can be determined."
        )
    )
    email: StrippedStr | None = Field(
        description=(
            "Email address of the author if explicitly stated on the paper "
            "(e.g. 'jane.smith@mit.edu'). Set to None if not present. "
            "Do not infer or construct email addresses from names or affiliations."
        )
    )

    def format_affiliations(self, separator: str = "; ") -> str:
        """Return a string representation of the affiliations.

        Joins all affiliations into a single string using the specified
        separator. Useful for displaying or logging affiliations concisely.

        Args:
            separator: String used to join affiliations. Defaults to '; '.

        Returns:
            A single string of all affiliations joined by the separator.
                Returns an empty string if affiliations is empty.

        Example:
            ```pycon
            >>> from candidex.schemas import AuthorExtraction
            >>> author = AuthorExtraction(
            ...     author="Jane Smith",
            ...     affiliations=["MIT CSAIL", "Stanford University"],
            ...     email="jane@mit.edu",
            ... )
            >>> author.format_affiliations()
            'MIT CSAIL; Stanford University'
            >>> author.format_affiliations(separator=" | ")
            'MIT CSAIL | Stanford University'

            ```
        """
        return separator.join(self.affiliations)

    def to_author(self) -> Author:
        r"""Return the author's affiliations as a `Author` object.

        Returns:
            The extracted data as a fully validated 'Author' domain object.

        Example:
            ```pycon
            >>> from candidex.schemas import AuthorExtraction
            >>> author = AuthorExtraction(
            ...     author="Jane Smith",
            ...     affiliations=["MIT CSAIL", "Stanford University"],
            ...     email="jane@mit.edu",
            ... )
            >>> author.to_author()
            Author(name='Jane Smith', affiliations=('MIT CSAIL', 'Stanford University'), email='jane@mit.edu')

            ```
        """
        return Author.from_raw(name=self.author, affiliations=self.affiliations, email=self.email)
