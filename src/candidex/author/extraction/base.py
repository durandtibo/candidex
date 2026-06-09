r"""Contain the base class for extracting authors from a paper."""

from __future__ import annotations

__all__ = ["BaseAuthorExtractor"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from candidex.author.author import Author
    from candidex.paper.paper import Paper


class BaseAuthorExtractor(ABC):
    """Base class for extracting authors and their affiliations from a
    paper.

    Defines the interface that all author extractor implementations must
    follow. Subclasses are responsible for locating the paper's PDF,
    extracting its text, and parsing author names, affiliations, and
    emails — typically using a combination of a `BasePdfExtractor` and
    an LLM.

    Example:
        ```pycon
        >>> from candidex.author.extraction import BaseAuthorExtractor
        >>> from candidex.author import Author
        >>> from candidex.paper import Paper
        >>> class MyExtractor(BaseAuthorExtractor):
        ...     def extract(self, paper: Paper) -> list[Author] | None:
        ...         return []
        ...
        >>> extractor = MyExtractor()
        >>> paper = Paper.from_raw(
        ...     title="Attention Is All You Need",
        ...     authors=["Ashish Vaswani"],
        ...     venue="NeurIPS",
        ...     year=2017,
        ...     pdf_url="https://arxiv.org/pdf/1706.03762",
        ... )
        >>> authors = extractor.extract(paper)

        ```
    """

    @abstractmethod
    def extract(self, paper: Paper) -> list[Author] | None:
        """Extract the authors and their affiliations from a paper.

        Implementations should locate the paper's PDF, extract text from
        the first page where author affiliations are typically listed, and
        parse each author's name, affiliation, and email.

        Args:
            paper: The `Paper` object to extract authors from. Must have
                   a valid `pdf_url` or a cached PDF on disk.

        Returns:
            A list of `Author` objects, one per author found in the paper.
                Returns an empty list if the paper has no extractable authors.
                Returns None if extraction failed entirely (e.g. PDF not found,
                text extraction error, or LLM failure) — distinguishing
                infrastructure failures from legitimate empty results.
        """
