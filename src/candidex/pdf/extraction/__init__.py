r"""Contain utilities to extract content from digital PDF files."""

from __future__ import annotations

__all__ = [
    "BasePdfExtractor",
    "PdfPlumberExtractor",
    "PyPdfExtractor",
    "PyPdfium2Extractor",
    "extract_text_pdfplumber",
    "extract_text_pypdf",
    "extract_text_pypdfium2",
]

from candidex.pdf.extraction.base import BasePdfExtractor
from candidex.pdf.extraction.pdfplumber import (
    PdfPlumberExtractor,
    extract_text_pdfplumber,
)
from candidex.pdf.extraction.pypdf import PyPdfExtractor, extract_text_pypdf
from candidex.pdf.extraction.pypdfium2 import PyPdfium2Extractor, extract_text_pypdfium2
