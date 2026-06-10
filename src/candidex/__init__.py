r"""Candidex — tools for indexing academic papers and their authors.

Provides utilities for scraping paper metadata from conference venues
(CVF, OpenReview), extracting author-affiliation information from PDFs
using LLMs, looking up OpenReview profiles, and assembling the results
into Polars DataFrames.
"""

from __future__ import annotations

__all__ = ["__version__"]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed, fallback if needed
    __version__ = "0.0.0"
