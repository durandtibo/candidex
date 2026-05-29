r"""Helpers for optional dependencies used by candidex.

The utilities in this package let callers:

- check whether an optional package is installed,
- fail early with a clear error when a package is required,
- gate function execution behind package availability.
"""

from __future__ import annotations

__all__ = [
    "check_colorlog",
    "colorlog_available",
    "is_colorlog_available",
    "raise_colorlog_missing_error",
]

from candidex.utils.imports.colorlog import (
    check_colorlog,
    colorlog_available,
    is_colorlog_available,
    raise_colorlog_missing_error,
)
