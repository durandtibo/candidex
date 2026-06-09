r"""Utilities for downloading and extracting text from digital PDF files.

Provides :class:`~candidex.pdf.extraction.BasePdfExtractor` and three
concrete backends — :class:`~candidex.pdf.extraction.PyPdfExtractor`,
:class:`~candidex.pdf.extraction.PdfPlumberExtractor`, and
:class:`~candidex.pdf.extraction.PyPdfium2Extractor` — as well as
:func:`~candidex.pdf.download.download_pdf` for fetching PDF files from URLs.
"""
