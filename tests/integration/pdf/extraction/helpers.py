from __future__ import annotations

__all__ = ["make_pdf"]


from typing import TYPE_CHECKING

from fpdf import FPDF

if TYPE_CHECKING:
    from pathlib import Path


def make_pdf(path: Path, texts: list[str]) -> Path:
    """Create a minimal single-column PDF with one text string per page.

    Intended for use in tests only. Each string in `texts` is written
    as a single cell on its own page using Helvetica 12pt.

    Args:
        path:  Destination path for the PDF file. Parent directory must
               exist.
        texts: List of strings to write, one per page.

    Returns:
        The `path` argument, for convenient chaining with the caller.

    Example:
        ```pycon
        >>> from pathlib import Path
        >>> pdf_path = make_pdf(Path("/tmp/test.pdf"), ["Hello", "World"])
        >>> pdf_path.exists()
        True

        ```
    """
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for text in texts:
        pdf.add_page()
        pdf.cell(0, 10, text)
    pdf.output(str(path))
    return path
