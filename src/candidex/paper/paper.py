r"""Contain schemas for papers."""

from __future__ import annotations

__all__ = ["Paper"]

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from candidex.utils.string import normalize_unicode

if TYPE_CHECKING:
    from collections.abc import Sequence

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Paper:
    """Represents an academic paper.

    Designed to be used as a dictionary key or set member — equality and
    hashing are based on all fields. Use `from_raw` to construct instances
    with automatic normalization of unicode characters and whitespace.

    Attributes:
        title:   Title of the paper, unicode-normalized and stripped.
        authors: Tuple of author names, each unicode-normalized and stripped.
                 None if authors are not available.
        venue:   Venue where the paper was published (e.g. 'CVPR', 'NeurIPS'),
                 unicode-normalized and stripped. None if not available.
        year:    Year the paper was published. None if not available.
        pdf_url: URL of the paper's PDF, stripped of whitespace.
                 None if not available.
    """

    title: str
    authors: tuple[str, ...] | None = None
    venue: str | None = None
    year: int | None = None
    pdf_url: str | None = None

    def hash(self) -> str:
        """Return a stable BLAKE2b hex digest of the paper.

        Serialises all fields to a canonical JSON string before hashing,
        ensuring stability across Python sessions (unlike the built-in
        `__hash__` which is randomised by default).

        Returns:
            A 128-character lowercase hexadecimal BLAKE2b digest string.

        Example:
            ```pycon
            >>> from candidex.paper import Paper
            >>> paper = Paper.from_raw(
            ...     title="Attention Is All You Need",
            ...     authors=["Ashish Vaswani", "Noam Shazeer"],
            ...     venue="NeurIPS",
            ...     year=2017,
            ...     pdf_url="https://arxiv.org/pdf/1706.03762",
            ... )
            >>> len(paper.hash())
            128

            ```
        """
        canonical = json.dumps(
            {
                "title": self.title,
                "authors": list(self.authors) if self.authors is not None else None,
                "venue": self.venue,
                "year": self.year,
                "pdf_url": self.pdf_url,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.blake2b(canonical.encode()).hexdigest()

    @classmethod
    def from_raw(
        cls,
        title: str,
        authors: Sequence[str] | None = None,
        venue: str | None = None,
        year: int | None = None,
        pdf_url: str | None = None,
    ) -> Paper:
        """Construct a `Paper` with normalized unicode and stripped
        whitespace.

        Normalizes unicode characters (e.g. 'é' → 'e') and strips leading/
        trailing whitespace from the title, venue, PDF URL, and each author
        name. Always use this constructor rather than the dataclass constructor
        directly to ensure consistent normalization.

        Args:
            title:   Title of the paper.
            authors: Sequence of author names, or None if not available.
            venue:   Venue where the paper was published (e.g. 'CVPR'),
                     or None if not available.
            year:    Year the paper was published, or None if not available.
            pdf_url: URL of the paper's PDF, or None if not available.

        Returns:
            A new `Paper` instance with normalized fields.

        Raises:
            ValueError: If `title` is empty or whitespace-only.

        Example:
            ```pycon
            >>> from candidex.paper import Paper
            >>> paper = Paper.from_raw(title="Attention Is All You Need")
            >>> paper.title
            'Attention Is All You Need'
            >>> paper.authors is None
            True

            ```
        """
        title = normalize_unicode(title.strip())
        if not title:
            msg = "Paper title cannot be empty."
            raise ValueError(msg)

        return cls(
            title=title,
            authors=(
                tuple(normalize_unicode(a.strip()) for a in authors)
                if authors is not None
                else None
            ),
            venue=normalize_unicode(venue.strip()) if venue is not None else None,
            year=year,
            pdf_url=pdf_url.strip() if pdf_url is not None else None,
        )
