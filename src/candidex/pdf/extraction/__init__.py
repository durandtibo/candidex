r"""Contain utilities to extract content from digital PDF files."""

from __future__ import annotations

__all__ = ["BasePdfExtractor", "PyPdfExtractor", "extract_text_pypdf"]

from candidex.pdf.extraction.base import BasePdfExtractor
from candidex.pdf.extraction.pypdf import PyPdfExtractor, extract_text_pypdf
