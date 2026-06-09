r"""Helpers for optional dependencies used by candidex.

The utilities in this package let callers:

- check whether an optional package is installed,
- fail early with a clear error when a package is required,
- gate function execution behind package availability.
"""

from __future__ import annotations

__all__ = [
    "check_colorlog",
    "check_pdfplumber",
    "check_pypdf",
    "colorlog_available",
    "is_colorlog_available",
    "is_pdfplumber_available",
    "is_pypdf_available",
    "pdfplumber_available",
    "pypdf_available",
    "raise_colorlog_missing_error",
    "raise_pdfplumber_missing_error",
    "raise_pypdf_missing_error",
]

from candidex.utils.imports.colorlog import (
    check_colorlog,
    colorlog_available,
    is_colorlog_available,
    raise_colorlog_missing_error,
)
from candidex.utils.imports.pdfplumber import (
    check_pdfplumber,
    is_pdfplumber_available,
    pdfplumber_available,
    raise_pdfplumber_missing_error,
)
from candidex.utils.imports.pypdf import (
    check_pypdf,
    is_pypdf_available,
    pypdf_available,
    raise_pypdf_missing_error,
)
