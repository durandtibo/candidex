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
        venue:   Venue where the paper was published (e.g. 'CVPR', 'NeurIPS'),
                 unicode-normalized and stripped.
        year:    Year the paper was published.
        pdf_url: URL of the paper's PDF, stripped of whitespace.
    """

    title: str
    authors: tuple[str, ...]
    venue: str
    year: int
    pdf_url: str

    def hash(self) -> str:
        """Return a stable BLAKE2b hex digest of the paper.

        Serialises all fields to a canonical JSON string before hashing,
        ensuring stability across Python sessions (unlike the built-in
        `__hash__` which is randomised by default).

        Returns:
            A 128-character lowercase hexadecimal BLAKE2b digest string.

        Example:
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
        """
        canonical = json.dumps(
            {
                "title": self.title,
                "authors": list(self.authors),
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
        authors: Sequence[str],
        venue: str,
        year: int,
        pdf_url: str,
    ) -> Paper:
        """Construct a `Paper` with normalized unicode and stripped
        whitespace.

        Normalizes unicode characters (e.g. 'é' → 'e') and strips leading/
        trailing whitespace from the title, venue, PDF URL, and each author
        name. Always use this constructor rather than the dataclass constructor
        directly to ensure consistent normalization.

        Args:
            title:   Title of the paper.
            authors: A sequence of author names.
            venue:   Venue where the paper was published (e.g. 'CVPR').
            year:    Year the paper was published.
            pdf_url: URL of the paper's PDF.

        Returns:
            A new `Paper` instance with normalized fields.

        Raises:
            ValueError: If `title` is empty or whitespace-only.
            ValueError: If `venue` is empty or whitespace-only.
            ValueError: If `pdf_url` is empty or whitespace-only.

        Example:
            >>> from candidex.paper import Paper
            >>> paper = Paper.from_raw(
            ...     title="Attention Is All You Need",
            ...     authors=["Ashish Vaswani", "Noam Shazeer"],
            ...     venue="NeurIPS",
            ...     year=2017,
            ...     pdf_url="https://arxiv.org/pdf/1706.03762",
            ... )
            >>> paper.authors
            ('Ashish Vaswani', 'Noam Shazeer')
        """
        title = normalize_unicode(title.strip())
        if not title:
            msg = "Paper title cannot be empty."
            raise ValueError(msg)

        venue = normalize_unicode(venue.strip())
        if not venue:
            msg = "Paper venue cannot be empty."
            raise ValueError(msg)

        pdf_url = pdf_url.strip()
        if not pdf_url:
            msg = "Paper PDF URL cannot be empty."
            raise ValueError(msg)

        return cls(
            title=title,
            authors=tuple(normalize_unicode(a.strip()) for a in authors),
            venue=venue,
            year=year,
            pdf_url=pdf_url,
        )
