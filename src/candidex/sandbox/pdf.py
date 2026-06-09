r"""Contain utilities for working with PDFs."""

from __future__ import annotations

__all__ = ["PDFReadError", "extract_first_page_text", "remove_unreadable_pdfs"]

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import pdfplumber
import polars as pl

from candidex.columns import PAPER_STEM
from candidex.utils.progressbar import make_progressbar

if TYPE_CHECKING:
    from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


class PDFReadError(Exception):
    """Raised when a PDF file cannot be read or yields no extractable
    text.

    Attributes:
        pdf_path: Path to the PDF file that could not be read.
    """

    def __init__(self, pdf_path: Path, reason: str) -> None:
        self.pdf_path = pdf_path
        super().__init__(f"Cannot read PDF '{pdf_path}': {reason}")


def extract_first_page_text(pdf_path: Path) -> str:
    """Extract raw text from the first page of a PDF.

    Only the first page is read since author affiliations are always
    listed there, avoiding the overhead of parsing the full document.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Raw text content of the first page.

    Raises:
        FileNotFoundError: If the PDF does not exist at `pdf_path`.
        PDFReadError: If the file is not a valid PDF, has no pages,
                           or yields no extractable text on the first page
                           (e.g. scanned image without OCR).
    """
    logger.debug("Extracting first page text from %s.", pdf_path)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            first_page = pdf.pages[0] if pdf.pages else None
            text = first_page.extract_text() if first_page else None
    except Exception as e:
        raise PDFReadError(pdf_path, str(e)) from e

    if first_page is None:
        raise PDFReadError(pdf_path, "PDF has no pages.")
    if not text or not text.strip():
        raise PDFReadError(pdf_path, "First page yields no extractable text.")

    return text


def remove_unreadable_pdfs(
    papers: pl.DataFrame,
    pdf_dir: Path,
    max_workers: int = 4,
) -> pl.DataFrame:
    """Test all PDFs with `extract_first_page_text` and remove
    unreadable ones.

    Attempts to extract text from the first page of each PDF. PDFs that raise
    a `PDFReadError` (corrupt, scanned, or empty) are deleted from disk and
    removed from the returned DataFrame. PDFs that are missing from disk are
    also removed from the returned DataFrame without attempting deletion.

    Args:
        papers: Polars DataFrame produced by `scrape_cvpr_papers` or
                     equivalent. Must contain a column named by `PAPER_STEM`
                     with the PDF filename stem for each paper.
        pdf_dir: Directory where the PDF files are stored. Each PDF must
                     be named `{stem}.pdf`.
        max_workers: Maximum number of concurrent threads for PDF reading.
                     Defaults to 4.

    Returns:
        A filtered Polars DataFrame containing only the papers whose PDFs
        were successfully read, in the original row order.

    Example:
        >>> df = scrape_cvpr_papers("https://openaccess.thecvf.com/CVPR2024?day=all")
        >>> df = remove_unreadable_pdfs(df, Path("data/cvpr2024/pdfs"))
        >>> # df now contains only papers with readable PDFs
    """
    rows = list(papers.iter_rows(named=True))
    readable = [False] * len(rows)

    def _test(index: int, row: dict) -> tuple[int, bool]:
        stem = row[PAPER_STEM]
        pdf_path = pdf_dir / f"{stem}.pdf"

        if not pdf_path.is_file():
            logger.warning("PDF not found, skipping: %s.", pdf_path.name)
            return index, False

        try:
            extract_first_page_text(pdf_path)
            return index, True
        except PDFReadError as e:
            logger.warning("Removing unreadable PDF %s: %s", pdf_path.name, e)
            pdf_path.unlink()
            return index, False

    with make_progressbar() as progress:
        task = progress.add_task("Testing PDFs", total=len(rows))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_test, i, row): i for i, row in enumerate(rows)}
            for future in as_completed(futures):
                index, is_readable = future.result()
                readable[index] = is_readable
                progress.advance(task)

    removed = readable.count(False)
    logger.info(
        "PDF test complete. %d/%d readable, %d removed.",
        len(rows) - removed,
        len(rows),
        removed,
    )

    readable_stems = {rows[i][PAPER_STEM] for i, is_readable in enumerate(readable) if is_readable}
    return papers.filter(pl.col(PAPER_STEM).is_in(readable_stems))
