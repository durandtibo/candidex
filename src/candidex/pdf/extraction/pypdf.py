r"""Contain utilities for working with PDFs."""

from __future__ import annotations

__all__ = ["PyPdfExtractor", "extract_text_pypdf"]

from typing import TYPE_CHECKING

from candidex.pdf.extraction.base import BasePdfExtractor
from candidex.utils.imports import check_pypdf, is_pypdf_available

if TYPE_CHECKING:
    from pathlib import Path

if is_pypdf_available():
    from pypdf import PdfReader
else:  # pragma: no cover
    from candidex.utils.fallback.pypdf import PdfReader


class PyPdfExtractor(BasePdfExtractor):
    r"""Extract text from digital PDFs using the `pypdf` library.

    Wraps `extract_text_pypdf` as a `BasePdfExtractor` implementation.
    Pages are separated by form feed characters (``\\f``), consistent
    with the plain-text page separator convention. Suitable for
    digitally-created PDFs where text is embedded as selectable
    characters. Not suitable for scanned PDFs — use an OCR-based
    extractor instead.

    Args:
        max_pages: Maximum number of pages to extract. If None, all
                   pages are extracted. Defaults to None.

    Raises:
        RuntimeError: If `pypdf` is not installed.

    Example:
        ```pycon
        >>> from pathlib import Path
        >>> from candidex.pdf.extraction import PyPdfExtractor
        >>> extractor = PyPdfExtractor(max_pages=1)
        >>> text = extractor.extract(Path("paper.pdf"))  # doctest: +SKIP
        >>> pages = text.split("\\f")  # doctest: +SKIP

        ```
    """

    def __init__(self, max_pages: int | None = None) -> None:
        check_pypdf()
        self._max_pages = max_pages

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(max_pages={self._max_pages!r})"

    def extract(self, pdf_path: Path) -> str:
        return extract_text_pypdf(pdf_path=pdf_path, max_pages=self._max_pages)


def extract_text_pypdf(pdf_path: Path, max_pages: int | None = None) -> str:
    r"""Extract text from a digital PDF using pypdf.

    Iterates through the pages of the PDF sequentially and extracts the
    text content of each page. Pages with no extractable text (e.g. scanned
    images) are silently skipped. Suitable for digitally-created PDFs where
    text is embedded as selectable characters.

    Pages are separated by a form feed character (\\f, ASCII 0x0C), which
    is the standard plain-text page separator. To split the result back
    into individual pages use `text.split("\\f")`.

    Args:
        pdf_path:  Path to the PDF file to extract text from.
        max_pages: Maximum number of pages to extract. If None, all pages
                   are extracted. Useful for extracting only the first page
                   (e.g. for affiliation extraction from paper headers).

    Returns:
        A single string containing the extracted text from all pages,
            separated by form feed characters (\\f). Returns an empty string
            if no text could be extracted.

    Raises:
        FileNotFoundError: If `pdf_path` does not exist.
        PdfReadError:      If the PDF is encrypted or malformed.

    Example:
        ```pycon
        >>> from pathlib import Path
        >>> from candidex.pdf.extraction import extract_text_pypdf
        >>> text = extract_text_pypdf(Path("paper.pdf"))
        >>> pages = text.split("\\f")
        >>> first_page = extract_text_pypdf(Path("paper.pdf"), max_pages=1)

        ```
    """
    check_pypdf()
    reader = PdfReader(pdf_path)
    pages = reader.pages[:max_pages] if max_pages is not None else reader.pages

    page_texts = [text for page in pages if (text := page.extract_text())]

    return "\f".join(page_texts)
