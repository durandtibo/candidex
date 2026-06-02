r"""Contain utilities for flattening affiliation data structures."""

from __future__ import annotations

__all__ = ["flatten_authors"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from candidex.schemas import AuthorAffiliation, PaperAffiliations


def flatten_authors(papers: Sequence[PaperAffiliations]) -> list[AuthorAffiliation]:
    """Return a flat list of all authors from a sequence of paper
    affiliations.

    Flattens the nested structure of `PaperAffiliations` into a single list
    of `AuthorAffiliation` objects. Useful for processing all authors across
    multiple papers in a single pass.

    Args:
        papers: A sequence of `PaperAffiliations` objects, each containing
                an ordered list of authors and their affiliations.

    Returns:
        A flat list of `AuthorAffiliation` objects from all papers in the
            order they appear. Returns an empty list if the input is empty or
            all papers have no authors.

    Example:
        ```pycon
        >>> from candidex.affiliation.flatten import flatten_authors
        >>> from candidex.schemas import AuthorAffiliation, PaperAffiliations
        >>> paper_a = PaperAffiliations(
        ...     authors=[
        ...         AuthorAffiliation(author="Jane Smith", affiliations=["MIT"], email=None),
        ...         AuthorAffiliation(author="John Doe", affiliations=["Stanford"], email=None),
        ...     ]
        ... )
        >>> paper_b = PaperAffiliations(
        ...     authors=[
        ...         AuthorAffiliation(author="Alice Brown", affiliations=["CMU"], email=None),
        ...     ]
        ... )
        >>> authors = flatten_authors([paper_a, paper_b])
        >>> [a.author for a in authors]
        ['Jane Smith', 'John Doe', 'Alice Brown']

        ```
    """
    return [author for paper in papers for author in paper.authors]
