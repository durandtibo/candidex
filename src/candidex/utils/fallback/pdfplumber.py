r"""Contain fallback implementations used when ``pdfplumber`` dependency
is not available.
"""

from __future__ import annotations

__all__ = ["pdfplumber"]

from types import ModuleType
from typing import Any, NoReturn

from candidex.utils.imports import raise_pdfplumber_missing_error


def fake_function(*args: Any, **kwargs: Any) -> NoReturn:  # noqa: ARG001
    r"""Fake function that raises an error because pdfplumber is not
    installed.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Raises:
        RuntimeError: pdfplumber is required for this functionality.
    """
    raise_pdfplumber_missing_error()


# Create a fake pdfplumber package
pdfplumber: ModuleType = ModuleType("pdfplumber")
pdfplumber.open = fake_function
