r"""Contain utilities for working with PDFs."""

from __future__ import annotations

__all__ = ["PyPdfium2Extractor", "extract_text_pypdfium2"]

from typing import TYPE_CHECKING

from candidex.pdf.extraction.base import BasePdfExtractor
from candidex.utils.imports import check_pypdfium2, is_pypdf_available

if TYPE_CHECKING:
    from pathlib import Path

if is_pypdf_available():
    import pypdfium2
else:  # pragma: no cover
    from candidex.utils.fallback.pypdfium2 import pypdfium2


class PyPdfium2Extractor(BasePdfExtractor):
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
        >>> from candidex.pdf.extraction import PyPdfium2Extractor
        >>> extractor = PyPdfium2Extractor(max_pages=1)
        >>> text = extractor.extract(Path("paper.pdf"))  # doctest: +SKIP
        >>> pages = text.split("\\f")  # doctest: +SKIP

        ```
    """

    def __init__(self, max_pages: int | None = None) -> None:
        check_pypdfium2()
        self._max_pages = max_pages

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(max_pages={self._max_pages!r})"

    def extract(self, pdf_path: Path) -> str:
        return extract_text_pypdfium2(pdf_path=pdf_path, max_pages=self._max_pages)


def extract_text_pypdfium2(
    pdf_path: Path,
    max_pages: int | None = None,
) -> str:
    r"""Extract text from a digital PDF using pypdfium2.

    Iterates through the pages of the PDF sequentially and extracts the
    text content of each page. Pages with no extractable text (e.g. scanned
    images) are silently skipped. Suitable for digitally-created PDFs where
    text is embedded as selectable characters.

    Pages are separated by a form feed character (\\f, ASCII 0x0C), which
    is the standard plain-text page separator. To split the result back
    into individual pages use ``text.split("\\f")``.

    Args:
        pdf_path: Path to the PDF file to extract text from.
        max_pages: Maximum number of pages to extract. If None, all pages
            are extracted. Useful for extracting only the first page
            (e.g. for affiliation extraction from paper headers).

    Returns:
        A single string containing the extracted text from all pages,
            separated by form feed characters (\\f). Returns an empty string
            if no text could be extracted.

    Raises:
        FileNotFoundError: If `pdf_path` does not exist.
        PdfiumError: If the PDF is malformed or unreadable.

    Example:
        ```pycon
        >>> from pathlib import Path
        >>> from candidex.pdf.extraction import extract_text_pypdfium2
        >>> text = extract_text_pypdfium2(Path("paper.pdf"))  # doctest: +SKIP

        ```
    """
    check_pypdfium2()
    pdf = pypdfium2.PdfDocument(pdf_path)
    page_indices = range(min(len(pdf), max_pages) if max_pages is not None else len(pdf))
    page_texts = []

    try:
        for page_idx in page_indices:
            page = pdf.get_page(page_idx)
            text_page = page.get_textpage()
            try:
                text = text_page.get_text_range().strip()
                if text:
                    page_texts.append(text)
            finally:
                text_page.close()
                page.close()
    finally:
        pdf.close()

    return "\f".join(page_texts)
