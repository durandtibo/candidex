r"""Define the base class for extracting text from PDF files."""

from __future__ import annotations

__all__ = ["BasePdfExtractor"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class BasePdfExtractor(ABC):
    """Base class for extracting text content from PDF files.

    Defines the interface that all PDF extractor implementations must
    follow. Subclasses are responsible for implementing the `extract`
    method using a specific extraction backend (e.g. pypdf, pdfplumber,
    pdfminer).

    Example:
        ```pycon
        >>> from pathlib import Path
        >>> from candidex.pdf.extraction import BasePdfExtractor
        >>> class MyExtractor(BasePdfExtractor):
        ...     def extract(self, pdf_path: Path) -> str:
        ...         return "extracted text"
        ...
        >>> extractor = MyExtractor()
        >>> extractor.extract(Path("paper.pdf"))
        'extracted text'

        ```
    """

    @abstractmethod
    def extract(self, pdf_path: Path) -> str:
        """Extract text content from a PDF file.

        Args:
            pdf_path: Path to the PDF file to extract text from.

        Returns:
            The extracted text content as a single string. Page
                separation is implementation-defined — subclasses should
                document their page separator convention.

        Raises:
            FileNotFoundError: If `pdf_path` does not exist.
            Exception: Subclass-specific exceptions may be raised
                               for malformed, encrypted, or unreadable PDFs.
        """
