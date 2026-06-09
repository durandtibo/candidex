r"""Contain fallback implementations used when ``pypdfium2`` dependency
is not available.
"""

from __future__ import annotations

__all__ = ["pypdfium2"]

from types import ModuleType
from typing import Any

from candidex.utils.imports import raise_pypdfium2_missing_error


class FakeClass:
    r"""Fake class that raises an error because pypdfium2 is not
    installed.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Raises:
        RuntimeError: pypdfium2 is required for this functionality.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        raise_pypdfium2_missing_error()


PdfDocument = FakeClass

# Create a fake pypdfium2 package
pypdfium2: ModuleType = ModuleType("pypdfium2")
pypdfium2.PdfDocument = PdfDocument
