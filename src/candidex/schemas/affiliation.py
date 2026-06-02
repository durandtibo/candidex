r"""Contain schemas for author affiliations."""

from __future__ import annotations

__all__ = ["AuthorAffiliation"]

import logging

from pydantic import BaseModel, Field, field_validator

logger: logging.Logger = logging.getLogger(__name__)


class AuthorAffiliation(BaseModel):
    """Represent a single author and their institutional affiliations.

    An author may be affiliated with multiple institutions simultaneously,
    for example a university department and an associated research lab.
    Affiliations are typically listed on the first page of an academic paper,
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

    author: str = Field(description="Full name of the author exactly as it appears on the paper.")
    affiliations: list[str] = Field(
        description=(
            "List of institutional affiliations for this author. Each entry is a complete "
            "affiliation string as it appears on the paper. Empty list if none can be determined."
        )
    )
    email: str | None = Field(
        description=(
            "Email address of the author if explicitly stated on the paper "
            "(e.g. 'jane.smith@mit.edu'). Set to None if not present. "
            "Do not infer or construct email addresses from names or affiliations."
        )
    )

    @field_validator("author", mode="before")
    @classmethod
    def strip_author(cls, v: str) -> str:
        return v.strip()

    @field_validator("affiliations", mode="before")
    @classmethod
    def strip_affiliations(cls, v: list[str]) -> list[str]:
        return [a.strip() for a in v]

    @field_validator("email", mode="before")
    @classmethod
    def strip_email(cls, v: str | None) -> str | None:
        return v.strip() if v is not None else None

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
            >>> author = AuthorAffiliation(
            ...     author="Jane Smith",
            ...     affiliations=["MIT CSAIL", "Stanford University"],
            ... )
            >>> author.format_affiliations()
            'MIT CSAIL; Stanford University'
            >>> author.format_affiliations(separator=" | ")
            'MIT CSAIL | Stanford University'
        """
        return separator.join(self.affiliations)
